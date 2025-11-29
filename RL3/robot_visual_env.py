"""
Визуальная среда для обучения робота Franka Panda управлению на основе изображений с камеры.

Поддерживает 3 режима обучения (curriculum learning):
  - "reach": Дотянуться до объекта (самая простая задача)
  - "grasp": Дотянуться и схватить объект  
  - "transfer": Схватить и перенести объект к цели (полная задача)
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
    
    Режимы задач:
      - "reach": Только дотянуться до объекта (для начального обучения)
      - "grasp": Дотянуться и схватить объект
      - "transfer": Полная задача - схватить и перенести к цели
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, use_gui=False, image_size=84, use_grayscale=False, 
                 frame_skip=4, frame_stack=4, task="reach"):
        """
        Args:
            render_mode: Режим рендеринга ('human' или 'rgb_array')
            use_gui: Использовать GUI PyBullet
            image_size: Размер изображения (по умолчанию 84x84)
            use_grayscale: Использовать ли grayscale вместо RGB
            frame_skip: Количество повторений одного действия
            frame_stack: Количество последних кадров для стекинга
            task: Режим задачи ("reach", "grasp", "transfer")
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.image_size = image_size
        self.use_grayscale = use_grayscale
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.task = task  # Режим задачи
        
        # Параметры камеры (Eye-to-hand: камера над столом, хороший обзор)
        self.camera_distance = 1.0  # Ближе к сцене
        self.camera_yaw = 45  # Угол для лучшего 3D восприятия
        self.camera_pitch = -45  # Смотрим сверху-сбоку
        self.camera_target = [0.5, 0.0, 0.15]  # Центр рабочей зоны, чуть выше стола
        
        # Кэширование матриц камеры (вычисляем один раз!)
        self._view_matrix = None
        self._proj_matrix = None
        
        # Параметры награды
        self.prev_distance_to_obj = None
        self.prev_distance_obj_to_goal = None
        
        # Состояние задачи
        self.object_grasped = False
        self.grasp_constraint = None
        
        # Лимиты рабочего пространства
        self.workspace_limits = {
            'x': [0.3, 0.7],
            'y': [-0.3, 0.3],
            'z': [0.05, 0.4]  # Минимум 0.05 - не даём опускаться к полу!
        }
        
        # Инициализация PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # ===== ОПТИМИЗАЦИЯ ФИЗИКИ =====
        # Увеличиваем timestep для меньшего числа вычислений
        # 1/60 вместо 1/240 = в 4 раза меньше шагов физики
        p.setTimeStep(1./60.)
        
        # Минимум итераций солвера (достаточно для манипулятора)
        p.setPhysicsEngineParameter(
            numSolverIterations=5,      # Уменьшено с 50 по умолчанию
            numSubSteps=1,              # Без подшагов
            useSplitImpulse=True,       # Быстрее для контактов
            splitImpulsePenetrationThreshold=-0.04
        )
        
        # Отключаем лишний рендеринг в DIRECT режиме
        if not self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        
        # Observation space: изображения с камеры + proprioception (углы джоинтов)
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
        
        # Action space: смещение схвата (dx, dy, dz) + захват (gripper)
        # gripper: -1 = открыть, +1 = закрыть
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1.0]),
            high=np.array([0.05, 0.05, 0.05, 1.0]),
            dtype=np.float32
        )
        
        # Буфер для frame stacking
        self.frame_buffer = []
        
        # Индексы управляемых джоинтов
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]  # Пальцы схвата
        self.end_effector_index = 11  # Индекс схвата
        
        # Начальные углы джоинтов
        self.reset_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        # Загрузка моделей
        self._load_models()
        
        # Счетчики
        self.step_count = 0
        self.max_steps = 150  # Больше шагов для задачи переноса
        
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
        
        # Открываем схват
        for joint_index in self.gripper_joints:
            p.resetJointState(self.robot_id, joint_index, 0.04)
        
        # Объекты создаются при reset
        self.object_id = None
        self.goal_marker_id = None
        
    def _create_target_object(self):
        """Создание объекта для переноса в случайной позиции"""
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Случайная позиция объекта (левая часть стола)
        x = np.random.uniform(0.35, 0.55)
        y = np.random.uniform(-0.25, 0.0)
        z = 0.025
        
        # Создание куба (красный - объект для переноса)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02],
                                          rgbaColor=[1, 0, 0, 1])
        self.object_id = p.createMultiBody(
            baseMass=0.05,  # Лёгкий объект для захвата
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z]
        )
        
        # Увеличиваем трение для лучшего захвата
        p.changeDynamics(self.object_id, -1, lateralFriction=1.0)
        
        return np.array([x, y, z])
    
    def _create_goal_marker(self):
        """Создание маркера целевой позиции (зелёный)"""
        if self.goal_marker_id is not None:
            p.removeBody(self.goal_marker_id)
        
        # Случайная позиция цели (правая часть стола, отдельно от объекта)
        x = np.random.uniform(0.45, 0.65)
        y = np.random.uniform(0.0, 0.25)
        z = 0.01  # Чуть выше стола (маркер на столе)
        
        # Создание маркера (зелёный плоский диск)
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=0.04, 
            length=0.005,
            rgbaColor=[0, 1, 0, 0.7]  # Полупрозрачный зелёный
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0,  # Статичный маркер
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z]
        )
        
        self.goal_pos = np.array([x, y, 0.025])  # Позиция куда нести (высота куба)
        return self.goal_pos
    
    def _get_camera_image(self):
        """Получение изображения с камеры (оптимизировано)"""
        # Кэшируем матрицы камеры (вычисляем только один раз)
        if self._view_matrix is None:
            self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_target,
                distance=self.camera_distance,
                yaw=self.camera_yaw,
                pitch=self.camera_pitch,
                roll=0,
                upAxisIndex=2
            )
            self._proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=3.0
            )
        
        # Получение изображения (используем TINY_RENDERER - быстрее на CPU)
        renderer = p.ER_TINY_RENDERER if not self.use_gui else p.ER_BULLET_HARDWARE_OPENGL
        
        (_, _, px, _, _) = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK  # Не нужна сегментация - экономим время
        )
        
        # Преобразование в RGB
        rgb_array = np.reshape(px, (self.image_size, self.image_size, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        
        if self.use_grayscale:
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            return gray[:, :, np.newaxis]
        
        return rgb_array
    
    def _get_joint_states(self):
        """Получение углов джоинтов (proprioception)"""
        joint_states = p.getJointStates(self.robot_id, self.arm_joints)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        return joint_positions
    
    def _get_end_effector_pos(self):
        """Получение позиции схвата"""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(state[0])
    
    def _get_object_pos(self):
        """Получение позиции объекта"""
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos)
    
    def _check_grasp(self):
        """Проверка захвата объекта (контакт обоих пальцев)"""
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.object_id)
        
        # Проверяем, что есть контакты с пальцами схвата
        finger_contacts = 0
        for contact in contacts:
            link_index = contact[3]  # Индекс линка робота
            if link_index in self.gripper_joints or link_index == self.end_effector_index:
                finger_contacts += 1
        
        return finger_contacts >= 1
    
    def _attach_object(self):
        """Прикрепление объекта к схвату при захвате"""
        if self.grasp_constraint is None and self._check_grasp():
            # Создаём фиксированное соединение
            ee_pos = self._get_end_effector_pos()
            obj_pos = self._get_object_pos()
            
            self.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.end_effector_index,
                childBodyUniqueId=self.object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.05],
                childFramePosition=[0, 0, 0]
            )
            self.object_grasped = True
            return True
        return False
    
    def _release_object(self):
        """Отпускание объекта"""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            self.object_grasped = False
    
    def _control_gripper(self, gripper_action):
        """Управление схватом: положительное = закрыть, отрицательное = открыть"""
        if gripper_action > 0.5:  # Закрыть
            target_pos = 0.01  # Почти закрыт
            self._attach_object()
        else:  # Открыть
            target_pos = 0.04  # Открыт
            self._release_object()
        
        for joint_index in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=50
            )
    
    def _update_frame_buffer(self, image):
        """Обновление буфера кадров для frame stacking"""
        self.frame_buffer.append(image)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_stacked_frames(self):
        """Получение стека кадров"""
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
        
        # Открываем схват
        for joint_index in self.gripper_joints:
            p.resetJointState(self.robot_id, joint_index, 0.04)
        
        # Сброс состояния захвата
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
        self.object_grasped = False
        
        # Создание объекта и цели
        self.object_start_pos = self._create_target_object()
        self.goal_pos = self._create_goal_marker()
        
        # Сброс счетчиков
        self.step_count = 0
        self.prev_distance_to_obj = None
        self.prev_distance_obj_to_goal = None
        
        # Очистка буфера кадров
        self.frame_buffer = []
        
        # Несколько шагов симуляции для стабилизации
        for _ in range(10):
            p.stepSimulation()
        
        # Получение начального наблюдения
        for _ in range(self.frame_stack):
            image = self._get_camera_image()
            self._update_frame_buffer(image)
        
        observation = {
            'image': self._get_stacked_frames(),
            'joints': self._get_joint_states()
        }
        
        return observation, {}
    
    def step(self, action):
        """Выполнение действия"""
        # Разделяем действие: движение (3D) + захват
        move_action = action[:3]
        gripper_action = action[3]
        
        # Управление схватом
        self._control_gripper(gripper_action)
        
        # Повторяем движение frame_skip раз
        for _ in range(self.frame_skip):
            current_pos = self._get_end_effector_pos()
            target_pos = current_pos + move_action
            
            # Ограничение рабочим пространством
            target_pos[0] = np.clip(target_pos[0], *self.workspace_limits['x'])
            target_pos[1] = np.clip(target_pos[1], *self.workspace_limits['y'])
            target_pos[2] = np.clip(target_pos[2], *self.workspace_limits['z'])
            
            # Инверсная кинематика (оптимизировано: меньше итераций)
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_index,
                target_pos,
                maxNumIterations=50,  # Уменьшено со 100
                residualThreshold=1e-4  # Менее строгий порог
            )
            
            # Установка целевых позиций
            for i, joint_index in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500
                )
            
            p.stepSimulation()
        
        # Получение наблюдения
        image = self._get_camera_image()
        self._update_frame_buffer(image)
        
        observation = {
            'image': self._get_stacked_frames(),
            'joints': self._get_joint_states()
        }
        
        # Вычисление награды для задачи ПЕРЕНОСА
        reward, terminated, info = self._compute_reward()
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self):
        """
        Вычисление награды в зависимости от режима задачи.
        
        Режимы:
          - "reach": Дотянуться до объекта (простая задача для начала)
          - "grasp": Дотянуться и схватить объект
          - "transfer": Схватить и перенести объект к цели
        """
        ee_pos = self._get_end_effector_pos()
        obj_pos = self._get_object_pos()
        
        reward = 0.0
        terminated = False
        
        # Базовые расстояния
        dist_3d = np.linalg.norm(ee_pos - obj_pos)  # 3D расстояние до объекта
        dist_xy = np.linalg.norm(ee_pos[:2] - obj_pos[:2])  # XY расстояние
        dist_obj_to_goal = np.linalg.norm(obj_pos[:2] - self.goal_pos[:2])
        
        # Проверка контакта
        contact = self._check_grasp()
        
        if self.task == "reach":
            # ========== ЗАДАЧА 1: REACH (дотянуться) ==========
            # Простейшая задача - просто коснуться объекта
            
            # Dense reward: приближение к объекту
            reward = -dist_3d * 10.0
            
            # Бонус за улучшение
            if self.prev_distance_to_obj is not None:
                improvement = self.prev_distance_to_obj - dist_3d
                reward += improvement * 50.0
            self.prev_distance_to_obj = dist_3d
            
            # Бонусы за близость
            if dist_3d < 0.15:
                reward += 5.0
            if dist_3d < 0.1:
                reward += 10.0
            if dist_3d < 0.05:
                reward += 20.0
            
            # УСПЕХ: касание объекта!
            if contact:
                reward += 100.0
                terminated = True
                
        elif self.task == "grasp":
            # ========== ЗАДАЧА 2: GRASP (схватить) ==========
            # Дотянуться СВЕРХУ и схватить объект
            
            if not self.object_grasped:
                # Целевая высота - чуть выше объекта
                target_height = obj_pos[2] + 0.06
                height_error = abs(ee_pos[2] - target_height)
                
                # Награда за XY приближение
                reward -= dist_xy * 5.0
                
                # Награда за правильную высоту
                reward -= height_error * 3.0
                
                # Бонус за улучшение XY
                if self.prev_distance_to_obj is not None:
                    improvement = self.prev_distance_to_obj - dist_xy
                    reward += improvement * 30.0
                self.prev_distance_to_obj = dist_xy
                
                # Бонусы за хорошую позицию
                if dist_xy < 0.1 and height_error < 0.1:
                    reward += 10.0
                if dist_xy < 0.05 and height_error < 0.05:
                    reward += 25.0
                
                # УСПЕХ: объект схвачен!
                if contact:
                    reward += 150.0
                    terminated = True
            else:
                # Объект уже схвачен - успех
                reward += 50.0
                terminated = True
                
        else:  # task == "transfer"
            # ========== ЗАДАЧА 3: TRANSFER (перенести) ==========
            # Полная задача: схватить и перенести к цели
            
            if not self.object_grasped:
                # Фаза 1: Приближение к объекту
                target_height = obj_pos[2] + 0.06
                height_error = abs(ee_pos[2] - target_height)
                
                reward -= dist_xy * 3.0
                reward -= height_error * 2.0
                
                if self.prev_distance_to_obj is not None:
                    improvement = self.prev_distance_to_obj - dist_xy
                    reward += improvement * 20.0
                self.prev_distance_to_obj = dist_xy
                
                if dist_xy < 0.05 and height_error < 0.05:
                    reward += 15.0
                
                if contact:
                    reward += 100.0
            else:
                # Фаза 2: Перенос к цели
                reward -= dist_obj_to_goal * 5.0
                
                if self.prev_distance_obj_to_goal is not None:
                    improvement = self.prev_distance_obj_to_goal - dist_obj_to_goal
                    reward += improvement * 30.0
                self.prev_distance_obj_to_goal = dist_obj_to_goal
                
                reward += 2.0  # Бонус за удержание
                
                if dist_obj_to_goal < 0.1:
                    reward += 15.0
                if dist_obj_to_goal < 0.05:
                    reward += 40.0
                
                # УСПЕХ: объект доставлен!
                if dist_obj_to_goal < 0.04:
                    reward += 200.0
                    terminated = True
        
        # Маленький штраф за время
        reward -= 0.02
        
        info = {
            'distance': dist_3d,  # Для совместимости со старым кодом
            'distance_to_object': dist_3d,
            'distance_to_goal': dist_obj_to_goal,
            'contact': contact,
            'object_grasped': self.object_grasped,
            'ee_pos': ee_pos,
            'obj_pos': obj_pos,
            'goal_pos': self.goal_pos,
            'task': self.task,
            'success': terminated
        }
        
        return reward, terminated, info
    
    def render(self):
        """Рендеринг среды"""
        if self.render_mode == "rgb_array":
            return self._get_camera_image()
        return None
    
    def close(self):
        """Закрытие среды"""
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        p.disconnect(self.physics_client)
