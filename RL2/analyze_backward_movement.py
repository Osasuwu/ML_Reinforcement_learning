import numpy as np

print("=" * 80)
print("АНАЛИЗ: КОГДА РОБОТ ДОЛЖЕН ЕХАТЬ НАЗАД")
print("=" * 80)

def should_go_backward(angle_deg):
    """Проверяет, должен ли робот ехать назад при данном угле"""
    angle_rad = np.radians(angle_deg)
    angle_abs = abs(angle_rad)
    
    # Текущее условие из robot_rl_env.py
    return angle_abs > 5 * np.pi / 6  # >150°

print("\nТестирование углов из последнего запуска:")
print()

# Из тестовых эпизодов
test_cases = [
    ("Эпизод 2", 0.15, -1.18, 0.03, -0.27),  # Едет назад но не должен!
    ("Эпизод 3", 0.64, -1.80, 0.89, -0.88),  # Едет назад но не должен!
    ("Эпизод 5", -0.46, 0.24, 0.29, 1.18),   # Едет назад но не должен!
]

for name, rx, ry, gx, gy in test_cases:
    # Предположим что робот смотрит вверх (yaw=π/2) в начале
    robot_pos = np.array([rx, ry])
    goal_pos = np.array([gx, gy])
    
    # Направление к цели
    delta = goal_pos - robot_pos
    target_angle = np.arctan2(delta[1], delta[0])
    
    # Пробуем разные углы робота
    for yaw_deg in [0, 45, 90, 135, 180, -135, -90, -45]:
        yaw_rad = np.radians(yaw_deg)
        actual_forward = yaw_rad + np.pi / 2  # Физика робота
        
        # Угол к цели
        angle_to_target = target_angle - actual_forward
        
        # Нормализация в диапазон [-π, π]
        while angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        while angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        
        angle_deg_val = np.degrees(angle_to_target)
        
        backward = should_go_backward(angle_deg_val)
        
        if abs(angle_deg_val) > 30:  # Только интересные случаи
            action = "НАЗАД" if backward else "ВПЕРЁД/ПОВОРОТ"
            print(f"{name}: yaw={yaw_deg:+4.0f}° → angle_to_target={angle_deg_val:+6.1f}° → {action}")

print("\n" + "=" * 80)
print("ПРОБЛЕМА:")
print("=" * 80)
print("Условие 'angle_abs > 150°' корректное, НО робот застревает в локальных минимумах!")
print()
print("Возможные причины:")
print("1. Робот не научился КОМБИНИРОВАТЬ повороты + движение")
print("2. Слишком большая награда за движение назад → робот выбирает только его")
print("3. Недостаточно штрафа за застревание в одном действии")
print()
print("=" * 80)
print("РЕШЕНИЕ:")
print("=" * 80)
print("1. Уменьшить бонус за движение назад (0.4 → 0.2)")
print("2. Увеличить бонус за движение вперёд (0.5 → 0.7)")
print("3. Добавить штраф за повторение одного действия много раз подряд")
print("4. Увеличить step_penalty чтобы мотивировать ЛЮБОЕ движение к цели")
