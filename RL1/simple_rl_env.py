import pybullet as p
import pybullet_data
import numpy as np
import time


class SimpleEnv:
    """
    Простая среда с роботом-кубиком и целевой сферой в PyBullet.
    Агент обучается двигаться к цели методом Q-learning.
    """
    
    def __init__(self, gui=False):
        self.gui = gui
        self.max_steps = 200
        self.goal_threshold = 0.5
        self.action_force = 100.0
        
        # Инициализация PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Загрузка плоскости
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Создание агента (кубик)
        self.agent_id = None
        self.target_id = None
        
        # Переменные состояния
        self.agent_pos = None
        self.target_pos = None
        self.prev_distance = None
        self.current_step = 0
        self.prev_action = None
        
    def reset(self):
        """
        Сброс среды: размещение агента и цели на случайных позициях.
        """
        # Удаление старых объектов
        if self.agent_id is not None:
            p.removeBody(self.agent_id)
        if self.target_id is not None:
            p.removeBody(self.target_id)
        
        # Случайная позиция агента
        agent_start_x = np.random.uniform(-2, 2)
        agent_start_pos = [agent_start_x, 0, 0.5]
        
        # Случайная позиция цели
        target_x = np.random.uniform(-2, 2)
        target_pos = [target_x, 0, 0.5]
        
        # Убедимся, что агент и цель не слишком близко
        while abs(agent_start_x - target_x) < 0.5:
            target_x = np.random.uniform(-2, 2)
            target_pos = [target_x, 0, 0.5]
        
        # Создание агента (кубик)
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], 
                                          rgbaColor=[0.0, 0.0, 1.0, 1.0])
        self.agent_id = p.createMultiBody(baseMass=1.0,
                                         baseCollisionShapeIndex=collision_shape,
                                         baseVisualShapeIndex=visual_shape,
                                         basePosition=agent_start_pos)
        
        # Добавление трения для контроля инерции
        p.changeDynamics(self.agent_id, -1, linearDamping=0.8, angularDamping=0.9)
        
        # Создание цели (сфера)
        target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3,
                                           rgbaColor=[1.0, 0.0, 0.0, 0.5])
        self.target_id = p.createMultiBody(baseMass=0,
                                          baseCollisionShapeIndex=target_collision,
                                          baseVisualShapeIndex=target_visual,
                                          basePosition=target_pos)
        
        # Обновление состояния
        self.agent_pos = np.array(agent_start_pos[:2])
        self.target_pos = np.array(target_pos[:2])
        self.prev_distance = self._get_distance()
        self.current_step = 0
        self.prev_action = None
        
        # Стабилизация физики
        for _ in range(10):
            p.stepSimulation()
        
        return self._get_state()
    
    def step(self, action):
        """
        Выполнение действия.
        action: 0 - движение влево, 1 - движение вправо
        """
        # Применение силы
        if action == 0:  # Влево
            force = [-self.action_force, 0, 0]
        elif action == 1:  # Вправо
            force = [self.action_force, 0, 0]
        else:
            raise ValueError(f"Неизвестное действие: {action}")
        
        p.applyExternalForce(self.agent_id, -1, force, [0, 0, 0], p.LINK_FRAME)
        
        # Симуляция физики
        for _ in range(10):
            p.stepSimulation()
        
        # Обновление позиции агента
        pos, _ = p.getBasePositionAndOrientation(self.agent_id)
        self.agent_pos = np.array(pos[:2])
        
        # Вычисление награды
        current_distance = self._get_distance()
        reward = -(current_distance - self.prev_distance) * 10.0
        
        # Штраф за каждый шаг для стимулирования быстрого достижения
        reward -= 0.1
        
        # Штраф за смену направления (колебания)
        if self.prev_action is not None and self.prev_action != action:
            reward -= 0.5
        
        # Бонус за достижение цели
        done = False
        if current_distance < self.goal_threshold:
            reward += 50.0
            done = True
        
        self.prev_distance = current_distance
        self.prev_action = action
        self.current_step += 1
        
        # Проверка лимита шагов
        if self.current_step >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done, {}
    
    def render(self, sleep_time=0.01):
        """
        Рендеринг среды (только в GUI режиме).
        """
        if self.gui:
            time.sleep(sleep_time)
    
    def close(self):
        """
        Закрытие соединения с PyBullet.
        """
        p.disconnect(self.physics_client)
    
    def _get_state(self):
        """
        Получение текущего состояния: [x_agent, x_target].
        """
        return np.array([self.agent_pos[0], self.target_pos[0]])
    
    def _get_distance(self):
        """
        Вычисление расстояния между агентом и целью.
        """
        return np.linalg.norm(self.agent_pos - self.target_pos)