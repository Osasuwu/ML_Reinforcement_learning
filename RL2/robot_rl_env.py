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
        self.max_steps = 150  # Уменьшено для мотивации быстрее достигать цель
        self.goal_threshold = 0.5
        self.max_velocity = 20.0  # Снижена для плавности
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
        # Расширенная система действий с контролем скорости
        # При отрицательных скоростях робот едет вперед
        
        # Действия 0-2: Движение вперёд с разной скоростью
        if action == 0:  # Вперёд быстро
            vl, vr = -self.max_velocity, -self.max_velocity
        elif action == 1:  # Вперёд медленно (более стабильно)
            vl, vr = -self.max_velocity * 0.5, -self.max_velocity * 0.5
        elif action == 2:  # Вперёд очень медленно (точная коррекция)
            vl, vr = -self.max_velocity * 0.3, -self.max_velocity * 0.3
        
        # Действия 3-4: Движение назад
        elif action == 3:  # Назад медленно
            vl, vr = self.max_velocity * 0.4, self.max_velocity * 0.4
        elif action == 4:  # Назад быстро
            vl, vr = self.max_velocity * 0.6, self.max_velocity * 0.6
        
        # Действия 5-6: Плавные повороты влево (дуга)
        elif action == 5:  # Плавный поворот влево (едет и поворачивает)
            vl, vr = -self.max_velocity * 0.4, -self.max_velocity * 0.9
        elif action == 6:  # Резкий поворот влево на месте
            vl, vr = self.max_velocity * 0.6, -self.max_velocity * 0.6
        
        # Действия 7-8: Плавные повороты вправо (дуга)
        elif action == 7:  # Плавный поворот вправо (едет и поворачивает)
            vl, vr = -self.max_velocity * 0.9, -self.max_velocity * 0.4
        elif action == 8:  # Резкий поворот вправо на месте
            vl, vr = -self.max_velocity * 0.6, self.max_velocity * 0.6
        
        else:  # action == 9: Стоять
            vl, vr = 0, 0
        
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
        
        # Получаем текущее состояние для расчета награды
        state = self._get_state()
        angle_to_target = state[3]  # Угол к цели
        
        # Награда за приближение к цели
        current_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        distance_reward = (self.prev_distance - current_dist) * 20.0
        
        # Штраф за каждый шаг (мотивация быстрее достигать цель)
        step_penalty = -0.15
        
        # УБИРАЕМ все штрафы за скорость - робот должен ехать БЫСТРО!
        speed_penalty = 0.0
        
        # Симметричная награда: робот может ехать вперёд ИЛИ назад к цели
        # Награда за направление - смотрит на цель (0°) или от цели (180°)
        angle_abs = abs(angle_to_target)
        
        # Вперёд к цели (угол близок к 0°) ИЛИ назад к цели (угол близок к 180°)
        if angle_abs < np.pi / 18:  # Менее 10° - смотрит ПРЯМО на цель
            alignment_reward = 0.3
        elif angle_abs > 35 * np.pi / 18:  # Более 170° - смотрит НАЗАД на цель (может ехать назад!)
            alignment_reward = 0.3  # Та же награда!
        elif angle_abs < np.pi / 12:  # 10-15° - хорошо
            alignment_reward = 0.15
        elif angle_abs > 11 * np.pi / 12:  # 170-175° - тоже хорошо для движения назад
            alignment_reward = 0.15
        elif angle_abs < np.pi / 6 or angle_abs > 5 * np.pi / 6:  # 10-30° или 150-170° - нормально
            alignment_reward = 0.0
        else:  # Смотрит боком - нужно поворачивать
            alignment_reward = -0.1
        
        # Бонус за правильные действия (учитываем ЗНАК угла!)
        action_bonus = 0.0
        
        # Если смотрит на цель (угол близок к 0) - ехать вперёд БЫСТРО!
        if angle_abs < np.pi / 6:  # -30° до +30° - направлен к цели
            if action == 0:  # Быстро вперёд - ОТЛИЧНО! Максимальный бонус!
                action_bonus = 0.5
            elif action == 1:  # Средне вперёд - нормально
                action_bonus = 0.2
            elif action == 2:  # Медленно вперёд - плохо, надо быстрее!
                action_bonus = 0.0
        
        # Если смотрит от цели (угол близок к ±180°) - можно ехать НАЗАД (экономим на развороте)
        elif angle_abs > 5 * np.pi / 6:  # >150° - смотрит назад на цель
            if action == 4:  # Назад быстро - эффективно!
                action_bonus = 0.4
            elif action == 3:  # Назад медленно - лучше чем стоять
                action_bonus = 0.1
        
        # Если смотрит боком - нужно поворачивать В ПРАВИЛЬНУЮ СТОРОНУ
        else:
            # Цель слева (положительный угол) - поворачивать ВЛЕВО
            if angle_to_target > 0:
                if action in [5, 6]:  # Поворот влево - ПРАВИЛЬНО!
                    action_bonus = 0.25 if action == 6 else 0.15  # Резкий быстрее
            # Цель справа (отрицательный угол) - поворачивать ВПРАВО
            else:
                if action in [7, 8]:  # Поворот вправо - ПРАВИЛЬНО!
                    action_bonus = 0.25 if action == 8 else 0.15  # Резкий быстрее
        
        # Суммарная награда
        reward = distance_reward + step_penalty + alignment_reward + action_bonus + speed_penalty
        
        self.prev_action = action
        
        # Проверка завершения
        done = False
        if current_dist < self.goal_threshold:
            # Большой бонус за достижение цели + бонус за скорость
            time_bonus = max(0, (self.max_steps - self.current_step) * 1.0)
            reward += 100.0 + time_bonus
            done = True
        
        self.prev_distance = current_dist
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            done = True
        
        if abs(self.robot_pos[0]) > 5 or abs(self.robot_pos[1]) > 5:
            reward -= 5.0
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
        
        # Нормализация yaw
        yaw = self.robot_yaw
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi
        
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
