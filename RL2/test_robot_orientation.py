"""
Тест для проверки правильности ориентации робота и направления движения.
"""
import numpy as np
import math
from robot_rl_env import RobotEnv
import pybullet as p
import time


def test_orientation_and_movement():
    """
    Проверка ориентации робота и направления движения.
    """
    env = RobotEnv(gui=True)
    
    print("=" * 80)
    print("ТЕСТ ОРИЕНТАЦИИ И ДВИЖЕНИЯ РОБОТА")
    print("=" * 80)
    print()
    print("Этот тест проверяет:")
    print("1. Как робот повёрнут при разных yaw")
    print("2. Куда он едет при команде 'вперёд' (L:Назад, R:Назад)")
    print("3. Правильность расчёта angle_to_target")
    print("=" * 80)
    print()
    
    test_cases = [
        # (robot_x, robot_y, robot_yaw, target_x, target_y, описание)
        (0, 0, 0, 2, 0, "yaw=0°: робот должен смотреть ВПРАВО (+X, восток)"),
        (0, 0, np.pi/2, 0, 2, "yaw=90°: робот должен смотреть ВВЕРХ (+Y, север)"),
        (0, 0, np.pi, -2, 0, "yaw=180°: робот должен смотреть ВЛЕВО (-X, запад)"),
        (0, 0, -np.pi/2, 0, -2, "yaw=-90°: робот должен смотреть ВНИЗ (-Y, юг)"),
    ]
    
    for i, (rx, ry, r_yaw, tx, ty, description) in enumerate(test_cases):
        print(f"\n{'═' * 80}")
        print(f"ТЕСТ {i+1}: {description}")
        print(f"{'═' * 80}")
        
        # Удаляем старые объекты
        if env.robot_id is not None:
            p.removeBody(env.robot_id)
        if env.target_id is not None:
            p.removeBody(env.target_id)
        
        # Создаём робота в заданной позиции и ориентации
        robot_orient = p.getQuaternionFromEuler([0, 0, r_yaw])
        env.robot_id = p.loadURDF("r2d2.urdf", 
                                    [rx, ry, 0.3],
                                    robot_orient,
                                    globalScaling=0.5)
        
        # Создаём цель
        env.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.3, 
                                                     rgbaColor=[1, 0, 0, 0.7]),
            basePosition=[tx, ty, 0.3])
        
        # Обновляем состояние
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        env.robot_pos = np.array(pos[:2])
        env.robot_yaw = p.getEulerFromQuaternion(orn)[2]
        env.target_pos = np.array([tx, ty])
        
        # Стабилизация
        for _ in range(50):
            p.stepSimulation()
        
        # Получаем состояние
        state = env._get_state()
        x, y, angle_to_target, distance = state
        
        print(f"\n📍 Начальное состояние:")
        print(f"  Позиция робота:     ({x:.2f}, {y:.2f})")
        print(f"  Yaw робота:         {env.robot_yaw:.4f} rad ({math.degrees(env.robot_yaw):.1f}°)")
        print(f"  Позиция цели:       ({tx:.2f}, {ty:.2f})")
        print(f"  Вектор к цели:      ({tx-rx:.2f}, {ty-ry:.2f})")
        
        # Вычисляем ожидаемый angle_to_target
        dx = tx - rx
        dy = ty - ry
        angle_to_target_global = math.atan2(dy, dx)
        expected_angle = angle_to_target_global - env.robot_yaw
        
        # Нормализация
        while expected_angle > np.pi:
            expected_angle -= 2 * np.pi
        while expected_angle < -np.pi:
            expected_angle += 2 * np.pi
        
        print(f"\n🎯 Анализ углов:")
        print(f"  Глобальный угол к цели: {angle_to_target_global:.4f} rad ({math.degrees(angle_to_target_global):.1f}°)")
        print(f"  Вычисленный angle_to_target: {angle_to_target:.4f} rad ({math.degrees(angle_to_target):.1f}°)")
        print(f"  Ожидаемый angle:             {expected_angle:.4f} rad ({math.degrees(expected_angle):.1f}°)")
        
        # Проверка
        if abs(angle_to_target - expected_angle) < 0.01:
            print(f"  ✓ УГЛЫ СОВПАДАЮТ!")
        else:
            print(f"  ✗ ОШИБКА! Углы НЕ совпадают!")
        
        # Интерпретация
        angle_deg = math.degrees(angle_to_target)
        print(f"\n📊 Интерпретация (angle_to_target = {angle_deg:.1f}°):")
        if -10 < angle_deg < 10:
            print(f"  ✓ Цель ПРЯМО ВПЕРЕДИ - робот смотрит правильно!")
        elif 10 <= angle_deg < 80:
            print(f"  ⚠ Цель ВПЕРЕДИ СЛЕВА - нужно повернуть налево")
        elif 80 <= angle_deg <= 100:
            print(f"  ⚠ Цель СЛЕВА - нужно сильно повернуть налево")
        elif 100 < angle_deg < 170:
            print(f"  ⚠ Цель СЗАДИ СЛЕВА - нужно развернуться")
        elif angle_deg >= 170 or angle_deg <= -170:
            print(f"  ⚠ Цель СЗАДИ - нужно развернуться на 180°")
        elif -170 < angle_deg < -100:
            print(f"  ⚠ Цель СЗАДИ СПРАВА - нужно развернуться")
        elif -100 <= angle_deg < -80:
            print(f"  ⚠ Цель СПРАВА - нужно сильно повернуть направо")
        elif -80 < angle_deg <= -10:
            print(f"  ⚠ Цель ВПЕРЕДИ СПРАВА - нужно повернуть направо")
        
        print(f"\n🚗 ТЕСТ ДВИЖЕНИЯ: едем ВПЕРЁД (action 18: Быстро L:Назад, R:Назад)")
        print(f"  Начальная позиция: ({env.robot_pos[0]:.2f}, {env.robot_pos[1]:.2f})")
        
        # Запоминаем начальную позицию и высоту
        start_pos = env.robot_pos.copy()
        start_height = pos[2]
        
        # Инициализируем переменные среды для step()
        env.prev_distance = distance
        env.current_step = 0
        env.prev_angle_to_target = abs(angle_to_target)
        
        # Едем вперёд 30 шагов
        for step in range(30):
            env.step(18)  # action 18 = Быстро вперёд
            
            # Проверяем высоту (падение)
            current_pos = p.getBasePositionAndOrientation(env.robot_id)[0]
            if current_pos[2] < 0.15:  # Если высота меньше 15см
                print(f"  ⚠ РОБОТ УПАЛ на шаге {step}! Высота: {current_pos[2]:.3f}м")
                break
        
        # Обновляем позицию
        pos, _ = p.getBasePositionAndOrientation(env.robot_id)
        env.robot_pos = np.array(pos[:2])
        end_pos = env.robot_pos
        final_height = pos[2]
        
        print(f"  Конечная позиция:  ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
        print(f"  Высота: начальная={start_height:.3f}м, конечная={final_height:.3f}м")
        
        # Вектор движения
        movement = end_pos - start_pos
        movement_distance = np.linalg.norm(movement)
        
        if movement_distance > 0.1:
            movement_angle = math.atan2(movement[1], movement[0])
            print(f"  Вектор движения:   ({movement[0]:.2f}, {movement[1]:.2f})")
            print(f"  Дистанция:         {movement_distance:.2f}м")
            print(f"  Угол движения:     {movement_angle:.4f} rad ({math.degrees(movement_angle):.1f}°)")
            print(f"  Yaw робота был:    {r_yaw:.4f} rad ({math.degrees(r_yaw):.1f}°)")
            
            # Проверка соответствия
            angle_diff = abs(movement_angle - r_yaw)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if angle_diff < 0.2:  # ~11 градусов
                print(f"  ✓ ПРАВИЛЬНО! Робот едет вперёд в направлении yaw")
            else:
                print(f"  ✗ ОШИБКА! Робот едет не туда (разница {math.degrees(angle_diff):.1f}°)")
                print(f"  Возможно, нужно скорректировать направление движения")
        else:
            print(f"  ⚠ Робот почти не двигался (дистанция: {movement_distance:.3f}м)")
            if final_height < 0.2:
                print(f"  ⚠ Возможно, робот УПАЛ!")
        
        input(f"\n{'─' * 80}\nНажмите Enter для следующего теста...\n{'─' * 80}\n")
    
    print("\n" + "=" * 80)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)
    print("\nВыводы:")
    print("1. Если angle_to_target = 0° когда цель впереди - расчёт правильный ✓")
    print("2. Если робот едет в направлении своего yaw - ориентация правильная ✓")
    print("3. Если робот падает - нужно уменьшить скорость или силу")
    print("=" * 80)
    
    env.close()


if __name__ == "__main__":
    test_orientation_and_movement()
