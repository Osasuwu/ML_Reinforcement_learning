import pybullet as p
import pybullet_data
import numpy as np
import time
import math


class RobotEnv:
    """
    Среда для обучения двухколёсного робота R2D2 навигации к цели.
    """
    
    def __init__(self, gui=False):
        self.gui = gui
        self.max_steps = 150
        self.goal_threshold = 0.5
        self.max_velocity = 20.0 
        self.max_force = 10.0
        
        # Инициализация PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(1./240.)
        
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.robot_id = None
        self.target_id = None
        
        self.robot_pos = None
        self.robot_yaw = None
        self.target_pos = None
        self.prev_distance = None
        self.current_step = 0
        self.prev_action = None
        self.stuck_counter = 0
        
    def reset(self):
        # Удаление старых объектов
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        if self.target_id is not None:
            p.removeBody(self.target_id)
        
        # Позиции
        robot_x = np.random.uniform(-2, 2)
        robot_y = np.random.uniform(-2, 2)
        target_x = np.random.uniform(-2, 2)
        target_y = np.random.uniform(-2, 2)
        
        while np.hypot(robot_x - target_x, robot_y - target_y) < 0.5:
            target_x = np.random.uniform(-2, 2)
            target_y = np.random.uniform(-2, 2)
        
        # Робот R2D2
        robot_orient = p.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])
        self.robot_id = p.loadURDF("r2d2.urdf", 
                                    [robot_x, robot_y, 0.3],
                                    robot_orient,
                                    globalScaling=0.5)
        
        # Найти колёса (у R2D2: 2 правых + 2 левых = 4 колеса)
        # Правые: индексы 2 (переднее), 3 (заднее)
        # Левые: индексы 6 (переднее), 7 (заднее)
        num_joints = p.getNumJoints(self.robot_id)
        
        # Для дифференциального привода используем передние колёса
        self.right_wheel = 2  # right_front_wheel_joint
        self.left_wheel = 6   # left_front_wheel_joint
        
        # Цель
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.3, 
                                                     rgbaColor=[1, 0, 0, 0.7]),
            basePosition=[target_x, target_y, 0.3])
        
        # Обновление состояния
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.robot_pos = np.array(pos[:2])
        # Без коррекции
        self.robot_yaw = p.getEulerFromQuaternion(orn)[2]
        self.target_pos = np.array([target_x, target_y])
        self.prev_distance = np.linalg.norm(self.robot_pos - self.target_pos)
        self.current_step = 0
        
        # Стабилизация
        for _ in range(50):
            p.stepSimulation()
        
        return self._get_state()
    
    def step(self, action):
        """
        Контроль над каждым колесом отдельно с выбором скорости.
        
        Действия (27 возможных комбинаций):
        action = speed_level * 9 + left_wheel_action * 3 + right_wheel_action
        
        Каждое колесо может быть в одном из трёх состояний:
        0: Назад (forward movement, т.к. отрицательная скорость)
        1: Стоп
        2: Вперёд (backward movement, т.к. положительная скорость)
        
        Уровни скорости:
        0: Медленная (0.4x max_velocity)
        1: Средняя (0.7x max_velocity)
        2: Быстрая (1.0x max_velocity)
        
        Примеры действий:
        0-8: Медленная скорость, различные комбинации колёс
        9-17: Средняя скорость, различные комбинации колёс
        18-26: Быстрая скорость, различные комбинации колёс
        """
        
        # Декодируем действие
        speed_level = action // 9
        wheel_combo = action % 9
        left_action = wheel_combo // 3
        right_action = wheel_combo % 3
        
        # Определяем множитель скорости
        if speed_level == 0:
            speed_mult = 0.4  # Медленная
        elif speed_level == 1:
            speed_mult = 0.7  # Средняя
        else:  # speed_level == 2
            speed_mult = 1.0  # Быстрая
        
        # Преобразуем действие в скорость колеса
        # 0 -> -max_velocity (вперёд), 1 -> 0 (стоп), 2 -> +max_velocity (назад)
        def action_to_velocity(wheel_action, speed_multiplier):
            if wheel_action == 0:
                return -self.max_velocity * speed_multiplier
            elif wheel_action == 1:
                return 0.0
            else:  # wheel_action == 2
                return self.max_velocity * speed_multiplier
        
        vl = action_to_velocity(left_action, speed_mult)
        vr = action_to_velocity(right_action, speed_mult)
        
        # Левые колёса: 6 (переднее), 7 (заднее)
        p.setJointMotorControl2(self.robot_id, 6, p.VELOCITY_CONTROL,
                               targetVelocity=vl, force=self.max_force)
        p.setJointMotorControl2(self.robot_id, 7, p.VELOCITY_CONTROL,
                               targetVelocity=vl, force=self.max_force)
        
        # Правые колёса: 2 (переднее), 3 (заднее)
        p.setJointMotorControl2(self.robot_id, 2, p.VELOCITY_CONTROL,
                               targetVelocity=vr, force=self.max_force)
        p.setJointMotorControl2(self.robot_id, 3, p.VELOCITY_CONTROL,
                               targetVelocity=vr, force=self.max_force)
        
        # Симуляция
        for _ in range(10):
            p.stepSimulation()
        
        # Обновление позиции
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.robot_pos = np.array(pos[:2])
        self.robot_yaw = p.getEulerFromQuaternion(orn)[2]
        
        # Вычисление расстояния до цели
        current_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        
        # Получаем состояние для вычисления угла к цели
        state = self._get_state()
        angle_to_target = state[3]
        
        # Упрощённая система наград (в сотых долях)
        # 1. Награда за приближение к цели
        distance_reward = (self.prev_distance - current_dist) * 1
        
        # 2. Штраф за каждый шаг (мотивация быстрее достигать цели)
        step_penalty = -0.01
        
        # 3. Награда за выравнивание на цель (приоритет движения по прямой)
        angle_abs = abs(angle_to_target)
        if angle_abs < np.pi / 18:  # < 10° - смотрит прямо на цель
            alignment_reward = 0.05
        elif angle_abs < np.pi / 9:  # < 20° - почти прямо
            alignment_reward = 0.02
        else:
            alignment_reward = 0.0
        
        # 4. Награда за достижение цели
        goal_reward = 0.0
        done = False
        
        if current_dist < self.goal_threshold:
            goal_reward = 10.0
            done = True
        
        # Суммарная награда
        reward = distance_reward + step_penalty + alignment_reward + goal_reward
        
        self.prev_distance = current_dist
        self.current_step += 1
        
        # Проверка лимита шагов
        if self.current_step >= self.max_steps:
            done = True
        
        # Проверка выхода за границы
        if abs(self.robot_pos[0]) > 5 or abs(self.robot_pos[1]) > 5:
            done = True
        
        return self._get_state(), reward, done, {}
    
    def render(self, sleep_time=0.01):
        if self.gui:
            p.addUserDebugLine([self.robot_pos[0], self.robot_pos[1], 0.1],
                              [self.target_pos[0], self.target_pos[1], 0.3],
                              [0, 1, 0], 1, lifeTime=0.1)
            time.sleep(sleep_time)
    
    def close(self):
        p.disconnect(self.physics_client)
    
    def _get_state(self):
        dx = self.target_pos[0] - self.robot_pos[0]
        dy = self.target_pos[1] - self.robot_pos[1]
        distance = np.linalg.norm([dx, dy])
        
        # Сырой yaw без нормализации (как координаты)
        yaw = self.robot_yaw
        
        # При отрицательных скоростях колес (-V/-V), робот едет "вперёд"
        # При yaw=0° робот едет в направлении +90° (на север, вдоль +Y)
        # Это означает: направление движения = yaw + 90°
        actual_forward_direction = yaw + np.pi/2
        
        # Угол к цели относительно направления движения робота
        angle_to_target_global = math.atan2(dy, dx)
        angle_to_target = angle_to_target_global - actual_forward_direction
        
        # Нормализация угла к цели в диапазон [-π, π]
        while angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        while angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        
        return np.array([self.robot_pos[0], self.robot_pos[1], yaw, 
                        angle_to_target, distance])
    
    def _get_distance(self):
        return np.linalg.norm(self.robot_pos - self.target_pos)
