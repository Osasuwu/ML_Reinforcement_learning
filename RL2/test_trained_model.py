import numpy as np
from robot_rl_env import RobotEnv
from train_robot import QLearningAgent
import time


def test_trained_agent(model_path="q_table_robot.pkl", n_episodes=10, gui=True, render_delay=0.03):
    """
    Тестирование обученного агента на новых задачах.
    
    Args:
        model_path: Путь к сохраненной Q-таблице
        n_episodes: Количество тестовых эпизодов
        gui: Показывать ли визуализацию
        render_delay: Задержка между кадрами (секунды)
    """
    # Создание среды
    env = RobotEnv(gui=gui)
    
    # Создание и загрузка агента
    agent = QLearningAgent()
    agent.load(model_path)
    
    # Отключаем исследование (используем только обученную политику)
    agent.epsilon = 0.0
    
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 80)
    print(f"Модель загружена из: {model_path}")
    print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
    print(f"Количество тестов: {n_episodes}")
    print("=" * 80)
    
    def get_action_name(action):
        """Получить название действия по его номеру"""
        speed_level = action // 9
        wheel_combo = action % 9
        left = wheel_combo // 3
        right = wheel_combo % 3
        
        speed_names = ['Медленно', 'Средне', 'Быстро']
        wheel_names = ['Назад', 'Стоп', 'Вперёд']
        
        return f"{speed_names[speed_level]}: L:{wheel_names[left]}, R:{wheel_names[right]}"
    
    results = []
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        action_counts = [0] * 27  # 27 действий вместо 10
        
        initial_distance = env._get_distance()
        
        print(f"\n{'='*80}")
        print(f"ТЕСТ ЭПИЗОД {episode + 1}/{n_episodes}")
        print(f"{'='*80}")
        print(f"Начальная позиция робота: ({state[0]:.2f}, {state[1]:.2f})")
        print(f"Позиция цели: ({env.target_pos[0]:.2f}, {env.target_pos[1]:.2f})")
        print(f"Начальное расстояние: {initial_distance:.2f}м")
        print(f"Начальный угол к цели: {np.degrees(state[3]):.1f}°")
        print("-" * 80)
        
        # Запуск эпизода
        while not done:
            # Выбор действия (без исследования)
            discrete_state = agent.discretize_state(state)
            action = np.argmax(agent.q_table[discrete_state])
            
            # Выполнение действия
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
            action_counts[action] += 1
            
            # Визуализация
            if gui:
                env.render(sleep_time=render_delay)
            
            # Вывод прогресса каждые 20 шагов
            if step_count % 20 == 0:
                current_dist = env._get_distance()
                print(f"Шаг {step_count:3d}: расстояние={current_dist:.2f}м, "
                      f"угол={np.degrees(next_state[3]):+6.1f}°, "
                      f"действие={get_action_name(action)}")
            
            state = next_state
        
        final_distance = env._get_distance()
        success = final_distance < env.goal_threshold
        
        if success:
            success_count += 1
        
        results.append({
            'episode': episode + 1,
            'success': success,
            'steps': step_count,
            'reward': total_reward,
            'initial_distance': initial_distance,
            'final_distance': final_distance
        })
        
        total_rewards.append(total_reward)
        total_steps.append(step_count)
        
        # Вывод результатов эпизода
        print("-" * 80)
        print(f"РЕЗУЛЬТАТ ЭПИЗОДА {episode + 1}:")
        print(f"  Статус: {'✓ УСПЕХ - ЦЕЛЬ ДОСТИГНУТА!' if success else '✗ Неудача - цель не достигнута'}")
        print(f"  Количество шагов: {step_count}")
        print(f"  Общая награда: {total_reward:.2f}")
        print(f"  Начальное расстояние: {initial_distance:.2f}м")
        print(f"  Финальное расстояние: {final_distance:.2f}м")
        print(f"  Улучшение: {initial_distance - final_distance:.2f}м ({(initial_distance - final_distance)/initial_distance*100:.1f}%)")
        print(f"\n  Использование действий (топ-5):")
        
        # Показываем только топ-5 наиболее используемых действий
        action_usage = [(i, count) for i, count in enumerate(action_counts) if count > 0]
        action_usage.sort(key=lambda x: x[1], reverse=True)
        
        for i, (action, count) in enumerate(action_usage[:5]):
            percentage = count / step_count * 100
            print(f"    {get_action_name(action):30s}: {count:3d} раз ({percentage:5.1f}%)")
    
    # Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего тестов: {n_episodes}")
    print(f"Успешных: {success_count} ({success_count/n_episodes*100:.1f}%)")
    print(f"Неудачных: {n_episodes - success_count} ({(n_episodes - success_count)/n_episodes*100:.1f}%)")
    print(f"\nСредняя награда: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
    print(f"Средняя длина эпизода: {np.mean(total_steps):.1f} (±{np.std(total_steps):.1f})")
    
    if success_count > 0:
        successful_steps = [r['steps'] for r in results if r['success']]
        print(f"\nУспешные эпизоды:")
        print(f"  Среднее количество шагов: {np.mean(successful_steps):.1f}")
        print(f"  Минимальное: {np.min(successful_steps)}")
        print(f"  Максимальное: {np.max(successful_steps)}")
    
    print("=" * 80)
    
    env.close()
    
    return results


def quick_demo(model_path="q_table_robot.pkl"):
    """
    Быстрая демонстрация работы обученного агента (1 эпизод).
    """
    print("Запуск быстрой демонстрации...\n")
    test_trained_agent(model_path=model_path, n_episodes=1, gui=True, render_delay=0.03)


if __name__ == "__main__":
    import sys
    import os
    
    # Проверка наличия модели
    model_file = r"q_table_robot.pkl"
    
    if not os.path.exists(model_file):
        print("=" * 80)
        print("ОШИБКА: Файл с обученной моделью не найден!")
        print("=" * 80)
        print(f"Ожидаемый файл: {model_file}")
        print("\nСначала запустите обучение:")
        print("  python train_robot.py")
        print("\nИли укажите путь к файлу модели:")
        print("  python test_trained_model.py <путь_к_модели>")
        print("=" * 80)
        sys.exit(1)
    
    # Определение пути к модели
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = model_file
    
    # Запуск тестирования
    print(f"\nИспользуется модель: {model_path}\n")
    
    # Можно изменить параметры здесь:
    # - n_episodes: количество тестовых эпизодов
    # - gui: True для визуализации, False для быстрого тестирования
    # - render_delay: задержка между кадрами (меньше = быстрее)
    
    results = test_trained_agent(
        model_path=model_path,
        n_episodes=5,  # Запустить 5 тестовых эпизодов
        gui=True,      # Показывать визуализацию
        render_delay=0.03  # Задержка 30мс между кадрами
    )
    
    print("\nТестирование завершено!")
