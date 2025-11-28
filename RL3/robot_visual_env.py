"""
Визуальная среда для обучения робота Franka Panda управлению на основе изображений с камеры.
"""
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import cv2


class RobotArmEnv(gym.Env):
    """
    Среда для обучения роботизированного манипулятора Franka Panda
    с использованием визуального входа (изображения с камеры).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, use_gui=False, image_size=84, use_grayscale=False, 
                 frame_skip=4, frame_stack=4):
        """
        Args:
            render_mode: Режим рендеринга ('human' или 'rgb_array')
            use_gui: Использовать GUI PyBullet
            image_size: Размер изображения (по умолчанию 84x84)
            use_grayscale: Использовать ли grayscale вместо RGB
            frame_skip: Количество повторений одного действия
            frame_stack: Количество последних кадров для стекинга
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.image_size = image_size
        self.use_grayscale = use_grayscale
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        
        # Параметры камеры (Eye-to-hand: камера над столом)
        self.camera_distance = 1.0
        self.camera_yaw = 0
        self.camera_pitch = -45
        self.camera_target = [0.5, 0.0, 0.0]
        
        # Параметры награды
        self.prev_distance = None
        self.contact_reward = 100.0
        self.distance_weight = 10.0
        self.time_penalty = 0.01
        
        # Лимиты рабочего пространства
        self.workspace_limits = {
            'x': [0.3, 0.7],
            'y': [-0.3, 0.3],
            'z': [0.02, 0.4]
        }
        
        # Инициализация PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./60.)  # Уменьшено с 240 до 60 для скорости (в 4 раза быстрее)
        p.setPhysicsEngineParameter(numSolverIterations=5)  # Меньше итераций = быстрее
        
        # Observation space: изображения с камеры + proprioception (углы джоинтов)
        # Изображение: (frame_stack, height, width, channels)
        img_channels = 1 if use_grayscale else 3
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(frame_stack, image_size, image_size, img_channels),
                dtype=np.uint8
            ),
            'joints': spaces.Box(
                low=-np.pi, high=np.pi,
                shape=(7,),  # 7 джоинтов Franka Panda
                dtype=np.float32
            )
        })
        
        # Action space: смещение схвата (dx, dy, dz)
        self.action_space = spaces.Box(
            low=-0.05, high=0.05,
            shape=(3,),
            dtype=np.float32
        )
        
        # Буфер для frame stacking
        self.frame_buffer = []
        
        # Индексы управляемых джоинтов (определяем ДО загрузки моделей!)
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.end_effector_index = 11  # Индекс схвата
        
        # Начальные углы джоинтов
        self.reset_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Загрузка моделей
        self._load_models()
        
        # Счетчики
        self.step_count = 0
        self.max_steps = 100  # Уменьшено со 200 для более быстрых эпизодов
        
    def _load_models(self):
        """Загрузка моделей робота, стола и объекта"""
        # Загрузка плоскости
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Загрузка стола
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.2])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.2], 
                                          rgbaColor=[0.5, 0.5, 0.5, 1])
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0.0, -0.2]
        )
        
        # Загрузка робота Franka Panda
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Установка начальных углов джоинтов
        for i, joint_index in enumerate(self.arm_joints):
            p.resetJointState(self.robot_id, joint_index, self.reset_joint_positions[i])
        
        # Создание целевого объекта (куб)
        self.object_id = None
        
    def _create_target_object(self):
        """Создание целевого объекта в случайной позиции"""
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Случайная позиция в рабочем пространстве
        x = np.random.uniform(*self.workspace_limits['x'])
        y = np.random.uniform(*self.workspace_limits['y'])
        z = 0.025  # На поверхности стола
        
        # Создание куба
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025],
                                          rgbaColor=[1, 0, 0, 1])
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z]
        )
        
        return np.array([x, y, z])
    
    def _get_camera_image(self):
        """Получение изображения с камеры"""
        # Параметры камеры
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.0
        )
        
        # Получение изображения
        (_, _, px, _, _) = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Преобразование в RGB (px уже массив, reshape нужен)
        rgb_array = np.reshape(px, (self.image_size, self.image_size, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)  # Убираем альфа-канал
        
        # Преобразование в grayscale если нужно
        if self.use_grayscale:
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            return gray[:, :, np.newaxis]  # Добавляем измерение канала
        
        return rgb_array
    
    def _get_joint_states(self):
        """Получение углов джоинтов (proprioception)"""
        joint_states = p.getJointStates(self.robot_id, self.arm_joints)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        return joint_positions
    
    def _get_end_effector_pos(self):
        """Получение позиции схвата"""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(state[0])  # Позиция
    
    def _get_object_pos(self):
        """Получение позиции объекта"""
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos)
    
    def _check_contact(self):
        """Проверка контакта схвата с объектом"""
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.object_id
        )
        return len(contact_points) > 0
    
    def _update_frame_buffer(self, image):
        """Обновление буфера кадров для frame stacking"""
        self.frame_buffer.append(image)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_frames(self):
        """Получение стека кадров"""
        # Если буфер не заполнен, дублируем последний кадр
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.append(self.frame_buffer[-1] if self.frame_buffer else 
                                    np.zeros((self.image_size, self.image_size, 
                                            1 if self.use_grayscale else 3), dtype=np.uint8))
        
        return np.array(self.frame_buffer, dtype=np.uint8)
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        
        # Сброс робота
        for i, joint_index in enumerate(self.arm_joints):
            p.resetJointState(self.robot_id, joint_index, self.reset_joint_positions[i])
        
        # Создание нового целевого объекта
        self.target_pos = self._create_target_object()
        
        # Сброс счетчиков
        self.step_count = 0
        self.prev_distance = None
        
        # Очистка буфера кадров
        self.frame_buffer = []
        
        # Получение начального наблюдения
        for _ in range(self.frame_stack):
            image = self._get_camera_image()
            self._update_frame_buffer(image)
        
        observation = {
            'image': self._get_stacked_frames(),
            'joints': self._get_joint_states()
        }
        
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Выполнение действия"""
        # Повторяем действие frame_skip раз
        for _ in range(self.frame_skip):
            # Получение текущей позиции схвата
            current_pos = self._get_end_effector_pos()
            
            # Новая целевая позиция
            target_pos = current_pos + action
            
            # Ограничение целевой позиции рабочим пространством
            target_pos[0] = np.clip(target_pos[0], *self.workspace_limits['x'])
            target_pos[1] = np.clip(target_pos[1], *self.workspace_limits['y'])
            target_pos[2] = np.clip(target_pos[2], *self.workspace_limits['z'])
            
            # Инверсная кинематика
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_pos,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            # Установка целевых позиций джоинтов
            for i, joint_index in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500
                )
            
            # Шаг симуляции
            p.stepSimulation()
        
        # Получение нового наблюдения
        image = self._get_camera_image()
        self._update_frame_buffer(image)
        
        observation = {
            'image': self._get_stacked_frames(),
            'joints': self._get_joint_states()
        }
        
        # Вычисление награды
        ee_pos = self._get_end_effector_pos()
        obj_pos = self._get_object_pos()
        distance = np.linalg.norm(ee_pos - obj_pos)
        
        # Dense reward: штраф за расстояние (усилен для быстрого обучения)
        reward = -self.distance_weight * distance
        
        # Бонус за приближение
        if self.prev_distance is not None:
            improvement = self.prev_distance - distance
            reward += improvement * 20.0  # Награда за приближение
        self.prev_distance = distance
        
        # Sparse reward: большая награда за контакт
        if self._check_contact():
            reward += self.contact_reward
            
        # Бонус за близость (дополнительная мотивация)
        if distance < 0.1:
            reward += 50.0
        elif distance < 0.2:
            reward += 20.0
        
        # Time penalty: штраф за каждый шаг
        reward -= self.time_penalty
        
        # Проверка завершения эпизода
        self.step_count += 1
        terminated = self._check_contact()  # Успех при контакте
        truncated = self.step_count >= self.max_steps  # Превышение лимита шагов
        
        info = {
            'distance': distance,
            'contact': self._check_contact(),
            'ee_pos': ee_pos,
            'obj_pos': obj_pos
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Рендеринг среды"""
        if self.render_mode == "rgb_array":
            return self._get_camera_image()
        return None
    
    def close(self):
        """Закрытие среды"""
        p.disconnect(self.physics_client)
