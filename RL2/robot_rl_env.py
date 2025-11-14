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
        self.max_steps = 200
        self.goal_threshold = 0.5
        self.max_velocity = 30.0
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
        # Скорости колёс
        if action == 0:  # Вперёд
            vl, vr = self.max_velocity, self.max_velocity
        elif action == 1:  # Назад
            vl, vr = -self.max_velocity * 0.7, -self.max_velocity * 0.7
        elif action == 2:  # Влево
            vl, vr = -self.max_velocity * 0.6, self.max_velocity * 0.6
        elif action == 3:  # Вправо
            vl, vr = self.max_velocity * 0.6, -self.max_velocity * 0.6
        else:  # Стоять
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
        # Без коррекции - используем raw yaw напрямую
        self.robot_yaw = p.getEulerFromQuaternion(orn)[2]
        
        # Награда за приближение к цели
        current_dist = np.linalg.norm(self.robot_pos - self.target_pos)
        distance_reward = (self.prev_distance - current_dist) * 15.0 
        reward = distance_reward - 0.05
        
        self.prev_action = action
        
        # Проверка завершения
        done = False
        if current_dist < self.goal_threshold:
            reward += 50.0
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
        
        # Raw yaw нормализация
        yaw = self.robot_yaw
        while yaw > np.pi:
            yaw -= 2 * np.pi
        while yaw < -np.pi:
            yaw += 2 * np.pi
        
        # ИСПРАВЛЕНИЕ: Робот физически едет перпендикулярно своему yaw
        # Когда yaw=0, робот едет вдоль Y- (направление -π/2)
        # Поэтому реальное направление движения = yaw - π/2
        actual_forward_direction = yaw - np.pi/2
        
        # Угол к цели относительно РЕАЛЬНОГО направления движения
        angle_to_target_global = math.atan2(dy, dx)
        angle_to_target = angle_to_target_global - actual_forward_direction
        
        while angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        while angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        
        return np.array([self.robot_pos[0], self.robot_pos[1], yaw, 
                        angle_to_target, distance])
    
    def _get_distance(self):
        return np.linalg.norm(self.robot_pos - self.target_pos)
