"""
Финальная среда для обучения робота Franka Panda.
Задача: схватить объект и перенести к цели.

Вход: ТОЛЬКО пиксели (64x64 grayscale) - никаких координат!
Выход: управление всеми 7 джоинтами + gripper
Камеры: 1 или 2 камеры (на манипуляторе и/или сбоку)
"""
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import cv2


class RobotEnv(gym.Env):
    """
    Среда для задачи pick-and-place.
    
    Особенности:
    - Вход: только изображения с камер (без координат!)
    - Выход: углы всех джоинтов напрямую (не IK)
    - 1 или 2 камеры для разных ракурсов
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self, 
        render_mode=None, 
        use_gui=False, 
        image_size=64,
        frame_stack=4,
        camera_mode="side",  # "side", "wrist", "both"
        max_steps=200
    ):
        """
        Args:
            render_mode: Режим рендеринга
            use_gui: Показывать GUI PyBullet
            image_size: Размер изображения (64x64)
            frame_stack: Количество кадров в стеке (для понимания движения)
            camera_mode: Режим камеры ("side", "wrist", "both")
            max_steps: Максимум шагов в эпизоде
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.camera_mode = camera_mode
        self.max_steps = max_steps
        
        # PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            # Отключаем лишние GUI элементы (preview окна камер, тени и т.д.)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # Индексы джоинтов Franka Panda
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]  # 7 джоинтов руки
        self.gripper_joints = [9, 10]  # 2 пальца
        self.end_effector_index = 11
        
        # Лимиты джоинтов (из URDF)
        self.joint_limits_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        
        # Начальные углы (рука над столом)
        self.home_position = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # Размер мини-камеры (depth или wrist)
        self.mini_cam_size = 32  # 32x32 для вторичной камеры
        
        # Observation space: только изображения!
        # side+depth / side+wrist: Dict с RGB (64x64) и вторичная камера (8x8)
        if camera_mode in ("side+depth", "side+wrist"):
            n_cameras = 1  # Основной канал RGB
            self.n_cameras = n_cameras
            secondary_key = "depth" if camera_mode == "side+depth" else "wrist"
            self.secondary_key = secondary_key
            # Dict space: RGB 64x64 + secondary 32x32
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=255,
                    shape=(frame_stack, image_size, image_size, 1),
                    dtype=np.uint8
                ),
                secondary_key: spaces.Box(
                    low=0, high=255,
                    shape=(frame_stack, self.mini_cam_size, self.mini_cam_size, 1),
                    dtype=np.uint8
                )
            })
        else:  # side only
            n_cameras = 1
            self.n_cameras = n_cameras
            self.secondary_key = None
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(frame_stack, image_size, image_size, n_cameras),
                dtype=np.uint8
            )
        
        # Action space: delta углов для 7 джоинтов + gripper
        # Маленькие изменения углов за шаг для плавности
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(8,),  # 7 joints + 1 gripper
            dtype=np.float32
        )
        
        # Масштаб действий (радианы за шаг)
        self.action_scale = 0.05  # ~3 градуса за шаг
        
        # Буфер кадров
        self.frame_buffer = []
        
        # Кэш матриц камер
        self._side_view_matrix = None
        self._side_proj_matrix = None
        self._wrist_view_matrix = None
        self._wrist_proj_matrix = None
        
        # Загрузка моделей
        self._load_models()
        
        # Состояние
        self.step_count = 0
        self.object_grasped = False
        self.grasp_constraint = None
        self.prev_obj_pos = None
        
    def _load_models(self):
        """Загрузка моделей"""
        # Плоскость
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Стол
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.5, 0.2])
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.5, 0.2],
            rgbaColor=[0.6, 0.6, 0.6, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0.0, -0.2]
        )
        
        # Робот
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Начальная позиция
        for i, joint_idx in enumerate(self.arm_joints):
            p.resetJointState(self.robot_id, joint_idx, self.home_position[i])
        for joint_idx in self.gripper_joints:
            p.resetJointState(self.robot_id, joint_idx, 0.04)
        
        self.object_id = None
        self.goal_id = None
        
    def _create_object(self):
        """Создание объекта для переноса"""
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Случайная позиция на столе (обе стороны, не близко к краям)
        x = np.random.uniform(0.25, 0.65)
        # Выбираем левую или правую сторону, не близко к центру
        if np.random.random() < 0.5:
            y = np.random.uniform(-0.35, -0.1)  # левая сторона
        else:
            y = np.random.uniform(0.1, 0.35)    # правая сторона
        z = 0.025
        
        # Белый цилиндр (хорошо виден в grayscale)
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.05)
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.02, length=0.05,
            rgbaColor=[1.0, 1.0, 1.0, 1]  # Белый
        )
        self.object_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z]
        )
        p.changeDynamics(self.object_id, -1, lateralFriction=1.0)
        
        self.object_start_pos = np.array([x, y, z])
        return self.object_start_pos
    
    def _create_goal(self):
        """Создание целевой зоны (на противоположной стороне от объекта)"""
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        
        # Генерируем позицию пока она не будет достаточно далеко от объекта
        min_distance = 0.2
        for _ in range(50):  # Максимум 50 попыток
            x = np.random.uniform(0.25, 0.65)
            # Выбираем сторону, противоположную объекту
            if self.object_start_pos[1] < 0:
                y = np.random.uniform(0.1, 0.35)   # правая сторона
            else:
                y = np.random.uniform(-0.35, -0.1) # левая сторона
            
            # Проверяем расстояние от объекта
            dist = np.sqrt((x - self.object_start_pos[0])**2 + (y - self.object_start_pos[1])**2)
            if dist >= min_distance:
                break
        
        z = 0.005
        
        # Тёмный квадрат (хорошо контрастирует с белым объектом в grayscale)
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.04, 0.005],
            rgbaColor=[0.2, 0.2, 0.2, 0.9]  # Тёмно-серый
        )
        self.goal_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z]
        )
        
        self.goal_pos = np.array([x, y, 0.025])
        return self.goal_pos
    
    def _get_side_camera_image(self):
        """Изображение с фронтальной камеры (вид спереди манипулятора)"""
        if self._side_view_matrix is None:
            self._side_view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.45, 0.0, 0.05],
                distance=1.0,
                yaw=90,   # Спереди (смотрит вдоль оси X)
                pitch=-25,
                roll=0,
                upAxisIndex=2
            )
            self._side_proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=2.0
            )
        
        _, _, px, _, _ = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=self._side_view_matrix,
            projectionMatrix=self._side_proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        # RGB -> Grayscale (конвертируем в uint8 для OpenCV)
        rgb = np.reshape(px, (self.image_size, self.image_size, 4))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray
    
    def _get_wrist_camera_image(self):
        """Изображение с камеры на запястье"""
        # Получаем позицию и ориентацию end-effector
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = np.array(state[0])
        ee_orn = state[1]
        
        # Матрица вращения из кватерниона
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        
        # Камера смотрит вниз (в локальной системе координат схвата)
        forward = rot_matrix @ np.array([0, 0, 1])
        up = rot_matrix @ np.array([0, -1, 0])
        
        target = ee_pos + forward * 0.3
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=ee_pos,
            cameraTargetPosition=target,
            cameraUpVector=up
        )
        
        if self._wrist_proj_matrix is None:
            self._wrist_proj_matrix = p.computeProjectionMatrixFOV(
                fov=80, aspect=1.0, nearVal=0.02, farVal=1.0
            )
        
        _, _, px, _, _ = p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=self._wrist_proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        rgb = np.reshape(px, (self.image_size, self.image_size, 4))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray
    
    def _get_depth_mini_camera_image(self, upscale=False):
        """
        Мини-камера 8x8 с угла стола под 45°.
        Видит ту же сцену что и side камера, но с другого ракурса и в низком разрешении.
        
        Args:
            upscale: если True, возвращает апскейленное изображение (для визуализации)
                     если False, возвращает 8x8 (для нейросети)
        """
        # Камера с противоположной стороны от основной (зеркально)
        # Основная side: yaw=90 (спереди), эта: yaw=-90 (сзади)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.45, 0.0, 0.05],  # Центр рабочей зоны (как у side)
            distance=1.0,
            yaw=180,   # С противоположной стороны от основной камеры
            pitch=-25, # Как у основной камеры
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=2.0
        )
        
        # Получаем RGB изображение в низком разрешении (8x8)
        _, _, px, _, _ = p.getCameraImage(
            width=self.mini_cam_size,
            height=self.mini_cam_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        # RGB -> Grayscale
        rgb = np.reshape(px, (self.mini_cam_size, self.mini_cam_size, 4))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        if upscale:
            # Апскейлим для визуализации
            return cv2.resize(
                gray, 
                (self.image_size, self.image_size), 
                interpolation=cv2.INTER_NEAREST
            )
        
        return gray  # 8x8 для нейросети
    
    def _get_wrist_mini_camera_image(self, upscale=False):
        """
        Мини-камера на клешне 8x8.
        Видит что прямо под схватом - объект или цель.
        
        Args:
            upscale: если True, возвращает апскейленное изображение (для визуализации)
                     если False, возвращает 8x8 (для нейросети)
        """
        # Получаем позицию и ориентацию end-effector
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        ee_pos = np.array(state[0])
        ee_orn = state[1]
        
        # Матрица вращения из кватерниона
        rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        
        # Камера смотрит вниз
        forward = rot_matrix @ np.array([0, 0, 1])
        up = rot_matrix @ np.array([0, -1, 0])
        target = ee_pos + forward * 0.3
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=ee_pos,
            cameraTargetPosition=target,
            cameraUpVector=up
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=80, aspect=1.0, nearVal=0.02, farVal=1.0
        )
        
        # 8x8 изображение
        _, _, px, _, _ = p.getCameraImage(
            width=self.mini_cam_size,
            height=self.mini_cam_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK
        )
        
        rgb = np.reshape(px, (self.mini_cam_size, self.mini_cam_size, 4))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        if upscale:
            return cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return gray  # 8x8 для нейросети
    
    def _get_camera_images(self):
        """
        Получение изображений с камер.
        Для side+depth/side+wrist возвращает dict.
        """
        if self.camera_mode == "side":
            return self._get_side_camera_image()[:, :, np.newaxis]
        elif self.camera_mode == "side+depth":
            # Dict: RGB 64x64 + depth 8x8
            side = self._get_side_camera_image()[:, :, np.newaxis]
            depth = self._get_depth_mini_camera_image(upscale=False)[:, :, np.newaxis]
            return {'image': side, 'depth': depth}
        elif self.camera_mode == "side+wrist":
            # Dict: RGB 64x64 + wrist 8x8
            side = self._get_side_camera_image()[:, :, np.newaxis]
            wrist = self._get_wrist_mini_camera_image(upscale=False)[:, :, np.newaxis]
            return {'image': side, 'wrist': wrist}
        else:
            # Fallback to side only
            return self._get_side_camera_image()[:, :, np.newaxis]
    
    def _update_frame_buffer(self, images):
        """Обновление буфера кадров"""
        self.frame_buffer.append(images)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
    
    def _get_observation(self):
        """Получение наблюдения (стек кадров)"""
        if self.camera_mode in ("side+depth", "side+wrist"):
            # Dict observation
            sec_key = self.secondary_key  # 'depth' или 'wrist'
            while len(self.frame_buffer) < self.frame_stack:
                if self.frame_buffer:
                    self.frame_buffer.append({
                        'image': self.frame_buffer[-1]['image'].copy(),
                        sec_key: self.frame_buffer[-1][sec_key].copy()
                    })
                else:
                    self.frame_buffer.append({
                        'image': np.zeros((self.image_size, self.image_size, 1), dtype=np.uint8),
                        sec_key: np.zeros((self.mini_cam_size, self.mini_cam_size, 1), dtype=np.uint8)
                    })
            
            # Stack frames
            images = np.array([f['image'] for f in self.frame_buffer], dtype=np.uint8)
            secondary = np.array([f[sec_key] for f in self.frame_buffer], dtype=np.uint8)
            return {'image': images, sec_key: secondary}
        else:
            # Box observation
            while len(self.frame_buffer) < self.frame_stack:
                self.frame_buffer.append(self.frame_buffer[-1].copy() if self.frame_buffer 
                                         else np.zeros((self.image_size, self.image_size, self.n_cameras), dtype=np.uint8))
            return np.array(self.frame_buffer, dtype=np.uint8)
    
    def _get_ee_pos(self):
        """Позиция end-effector"""
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        return np.array(state[0])
    
    def _get_object_pos(self):
        """Позиция объекта"""
        pos, _ = p.getBasePositionAndOrientation(self.object_id)
        return np.array(pos)
    
    def _check_grasp(self):
        """Проверка контакта с объектом"""
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.object_id)
        # Любой контакт с пальцами или рукой (links 8, 9, 10)
        gripper_links = [8, 9, 10]  # hand, leftfinger, rightfinger
        for c in contacts:
            if c[3] in gripper_links:
                return True
        return False
    
    def _can_attach_object(self):
        """Проверка можно ли прикрепить объект (объект близко и схват закрыт)"""
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        dist = np.linalg.norm(ee_pos - obj_pos)
        
        # Проверяем положение пальцев (закрыты ли)
        finger_pos = p.getJointState(self.robot_id, self.gripper_joints[0])[0]
        gripper_closed = finger_pos < 0.03  # Пальцы достаточно закрыты
        
        # Объект близко к end-effector и есть контакт
        close_enough = dist < 0.08
        has_contact = self._check_grasp()
        
        return close_enough and gripper_closed and has_contact
    
    def _attach_object(self):
        """Прикрепление объекта к схвату"""
        if self.grasp_constraint is None and self._can_attach_object():
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
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        
        # Сброс constraint
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
        self.object_grasped = False
        
        # Сброс робота в home позицию с небольшим шумом
        for i, joint_idx in enumerate(self.arm_joints):
            noise = np.random.uniform(-0.05, 0.05)
            target_pos = self.home_position[i] + noise
            # Сбрасываем состояние джоинта (позиция и скорость)
            p.resetJointState(self.robot_id, joint_idx, target_pos, targetVelocity=0)
            # ВАЖНО: Сбрасываем motor controller на эту же позицию!
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=240
            )
        
        # Gripper
        for joint_idx in self.gripper_joints:
            p.resetJointState(self.robot_id, joint_idx, 0.04, targetVelocity=0)
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0.04,
                force=50
            )
        
        # Создание объекта и цели
        self._create_object()
        self._create_goal()
        
        # Стабилизация (меньше шагов, т.к. контроллеры уже на месте)
        for _ in range(20):
            p.stepSimulation()
        
        # Инвалидация кэша wrist камеры
        self._wrist_view_matrix = None
        
        # Сброс буфера
        self.frame_buffer = []
        images = self._get_camera_images()
        for _ in range(self.frame_stack):
            self._update_frame_buffer(images.copy())
        
        self.step_count = 0
        self.prev_obj_pos = self._get_object_pos().copy()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Выполнение действия"""
        # Разделяем действие: 7 джоинтов + gripper
        joint_actions = action[:7] * self.action_scale
        gripper_action = action[7]
        
        # Получаем текущие углы
        current_joints = np.array([p.getJointState(self.robot_id, j)[0] for j in self.arm_joints])
        
        # Вычисляем новые углы
        new_joints = current_joints + joint_actions
        
        # Ограничиваем пределами
        new_joints = np.clip(new_joints, self.joint_limits_low, self.joint_limits_high)
        
        # Применяем к джоинтам
        for i, joint_idx in enumerate(self.arm_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=new_joints[i],
                force=240
            )
        
        # Gripper
        was_grasped = self.object_grasped
        if gripper_action > 0:  # Закрыть
            gripper_pos = 0.01
            self._attach_object()
        else:  # Открыть
            gripper_pos = 0.04
            self._release_object()
        
        # Бонус за успешный захват (момент когда схватили)
        just_grasped = self.object_grasped and not was_grasped
        
        for joint_idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                force=50
            )
        
        # Симуляция
        for _ in range(4):  # 4 шага физики
            p.stepSimulation()
        
        # Наблюдение
        images = self._get_camera_images()
        self._update_frame_buffer(images)
        obs = self._get_observation()
        
        # Награда
        reward, terminated, info = self._compute_reward()
        
        # Большой бонус за момент захвата
        if just_grasped:
            reward += 30.0
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self):
        """
        Вычисление награды для задачи pick-and-place.
        
        Стратегия: dense reward с фазами
        1. Приближение к объекту
        2. Захват объекта
        3. Перенос к цели
        4. Размещение на цели
        """
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_object_pos()
        goal_pos = self.goal_pos
        
        reward = 0.0
        terminated = False
        
        # Расстояния
        dist_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        dist_obj_to_goal = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
        
        # Проверка падения объекта
        if obj_pos[2] < -0.1:
            reward = -50.0
            terminated = True
            info = {'success': False, 'reason': 'object_fell'}
            return reward, terminated, info
        
        if not self.object_grasped:
            # === ФАЗА 1: Приближение к объекту ===
            
            # Dense reward за приближение
            approach_reward = np.exp(-5.0 * dist_ee_to_obj)
            reward += approach_reward * 2.0
            
            # Высота: поощряем подход сверху
            ideal_height = obj_pos[2] + 0.08
            height_diff = abs(ee_pos[2] - ideal_height)
            height_reward = np.exp(-10.0 * height_diff)
            reward += height_reward * 0.5
            
            # Бонус за контакт (даже без полного захвата)
            if self._check_grasp():
                reward += 5.0
                
            # Большой бонус когда объект схвачен
            # (это произойдёт после _attach_object в step)
                
        else:
            # === ФАЗА 2: Перенос к цели ===
            
            # Dense reward за приближение к цели
            transport_reward = np.exp(-5.0 * dist_obj_to_goal)
            reward += transport_reward * 3.0
            
            # Бонус за удержание
            reward += 0.5
            
            # Проверка движения к цели
            if self.prev_obj_pos is not None:
                prev_dist = np.linalg.norm(self.prev_obj_pos[:2] - goal_pos[:2])
                curr_dist = dist_obj_to_goal
                if curr_dist < prev_dist:
                    reward += (prev_dist - curr_dist) * 50.0
            
            # === УСПЕХ: объект на цели ===
            if dist_obj_to_goal < 0.04:
                reward += 100.0
                terminated = True
                info = {'success': True, 'reason': 'goal_reached'}
                return reward, terminated, info
        
        self.prev_obj_pos = obj_pos.copy()
        
        # Штраф за время
        reward -= 0.01
        
        info = {
            'success': False,
            'dist_ee_to_obj': dist_ee_to_obj,
            'dist_obj_to_goal': dist_obj_to_goal,
            'object_grasped': self.object_grasped
        }
        
        return reward, terminated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_side_camera_image()
        return None
    
    def close(self):
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        p.disconnect(self.physics_client)
