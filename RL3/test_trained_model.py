"""
Скрипт для тестирования обученной модели.
Визуализирует работу агента в GUI и сохраняет статистику.
"""
import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from robot_visual_env import RobotArmEnv
from feature_extractor import MobileNetFeatureExtractor
import matplotlib.pyplot as plt
import time


def load_model_config(model_path):
    """
    Загрузка конфигурации модели из JSON файла
    """
    config_path = model_path.replace('.zip', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def test_model(model_path, n_episodes=10, use_gui=True, record_video=False):
    """
    Тестирование обученной модели
    
    Args:
        model_path: Путь к сохраненной модели
        n_episodes: Количество эпизодов для тестирования
        use_gui: Показывать GUI PyBullet
        record_video: Записывать видео (пока не реализовано)
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)
    
    # Проверка существования модели
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("\nДоступные модели в RL3/models/:")
        if os.path.exists("RL3/models"):
            models = [f for f in os.listdir("RL3/models") if f.endswith('.zip')]
            for m in models:
                print(f"  - {m}")
        return
    
    print(f"✓ Загрузка модели из {model_path}")
    
    # Загрузка конфигурации модели
    config = load_model_config(model_path)
    
    if config:
        print(f"✓ Конфигурация модели загружена:")
        print(f"  - Эксперимент: {config.get('experiment_name', 'N/A')}")
        print(f"  - Image size: {config.get('image_size', 84)}x{config.get('image_size', 84)}")
        print(f"  - Image mode: {'Grayscale' if config.get('use_grayscale', False) else 'RGB'}")
        print(f"  - Frame skip: {config.get('frame_skip', 4)}")
        print(f"  - Frame stack: {config.get('frame_stack', 4)}")
        print(f"  - Total timesteps: {config.get('total_timesteps', 'N/A'):,}")
        
        # Использование параметров из конфигурации
        image_size = config.get('image_size', 84)
        use_grayscale = config.get('use_grayscale', False)
        frame_skip = config.get('frame_skip', 4)
        frame_stack = config.get('frame_stack', 4)
    else:
        print("⚠ Конфигурация не найдена, используются параметры по умолчанию:")
        print("  - Image size: 84x84")
        print("  - Image mode: RGB")
        print("  - Frame skip: 4")
        print("  - Frame stack: 4")
        image_size = 84
        use_grayscale = False
        frame_skip = 4
        frame_stack = 4
    
    # Создание среды с параметрами из конфигурации
    env = RobotArmEnv(
        use_gui=use_gui,
        image_size=image_size,
        use_grayscale=use_grayscale,
        frame_skip=frame_skip,
        frame_stack=frame_stack
    )
    
    # Загрузка модели
    model = PPO.load(model_path, env=env)
    
    print(f"✓ Модель загружена")
    print(f"\nЗапуск {n_episodes} тестовых эпизодов...\n")
    
    # Статистика
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    min_distances = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        min_dist = float('inf')
        
        print(f"Эпизод {episode + 1}/{n_episodes}")
        
        while not done:
            # Получение действия от модели
            action, _states = model.predict(obs, deterministic=True)
            
            # Выполнение действия
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Отслеживание минимального расстояния
            if info['distance'] < min_dist:
                min_dist = info['distance']
            
            # Вывод информации каждые 20 шагов
            if episode_length % 20 == 0 or done:
                print(f"  Шаг {episode_length}: "
                      f"Reward={reward:.2f}, "
                      f"Distance={info['distance']:.4f}, "
                      f"Contact={info['contact']}")
            
            if use_gui:
                time.sleep(0.01)  # Замедление для визуализации
        
        # Проверка успеха
        if info['contact']:
            success_count += 1
            print(f"  ✓ УСПЕХ: Контакт достигнут!")
        else:
            print(f"  ✗ Контакт не достигнут. Min distance: {min_dist:.4f}")
        
        print(f"  Итого: Reward={episode_reward:.2f}, Length={episode_length}\n")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        min_distances.append(min_dist)
    
    env.close()
    
    # Статистика
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    success_rate = (success_count / n_episodes) * 100
    
    print(f"\nОбщая статистика:")
    print(f"  - Успешных эпизодов: {success_count}/{n_episodes} ({success_rate:.1f}%)")
    print(f"  - Средняя награда: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  - Средняя длина: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  - Среднее мин. расстояние: {np.mean(min_distances):.4f} ± {np.std(min_distances):.4f}")
    print(f"  - Лучшая награда: {max(episode_rewards):.2f}")
    print(f"  - Худшая награда: {min(episode_rewards):.2f}")
    
    # Построение графиков
    print("\n⏳ Построение графиков...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1: Награды по эпизодам
    axes[0, 0].plot(episode_rewards, marker='o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: Длина эпизодов
    axes[0, 1].plot(episode_lengths, marker='s', linewidth=2, markersize=6, color='orange')
    axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--',
                       label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: Минимальные расстояния
    axes[1, 0].plot(min_distances, marker='^', linewidth=2, markersize=6, color='green')
    axes[1, 0].axhline(y=np.mean(min_distances), color='r', linestyle='--',
                       label=f'Mean: {np.mean(min_distances):.4f}')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Minimum Distance')
    axes[1, 0].set_title('Minimum Distance to Target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: Статистика успеха
    success_data = [success_count, n_episodes - success_count]
    axes[1, 1].pie(success_data, labels=['Success', 'Failure'], 
                   autopct='%1.1f%%', colors=['#4CAF50', '#F44336'],
                   startangle=90)
    axes[1, 1].set_title(f'Success Rate: {success_rate:.1f}%')
    
    plt.tight_layout()
    plt.savefig('RL3/test_results.png', dpi=150)
    print("✓ Графики сохранены в RL3/test_results.png")
    plt.close()
    
    # Сохранение детальной статистики
    stats_path = 'RL3/test_statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ОБУЧЕННОЙ МОДЕЛИ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Модель: {model_path}\n")
        f.write(f"Количество эпизодов: {n_episodes}\n\n")
        
        f.write("ОБЩАЯ СТАТИСТИКА:\n")
        f.write(f"  Успешных эпизодов: {success_count}/{n_episodes} ({success_rate:.1f}%)\n")
        f.write(f"  Средняя награда: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}\n")
        f.write(f"  Средняя длина: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}\n")
        f.write(f"  Среднее мин. расстояние: {np.mean(min_distances):.4f} ± {np.std(min_distances):.4f}\n\n")
        
        f.write("ДЕТАЛИ ПО ЭПИЗОДАМ:\n")
        for i, (r, l, d) in enumerate(zip(episode_rewards, episode_lengths, min_distances)):
            success_mark = "✓" if i < success_count else "✗"
            f.write(f"  Эпизод {i+1}: {success_mark} Reward={r:.2f}, Length={l}, MinDist={d:.4f}\n")
    
    print(f"✓ Детальная статистика сохранена в {stats_path}")
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


def test_best_model(n_episodes=10, use_gui=True):
    """Тестирование лучшей модели"""
    best_model_path = "RL3/models/best_model.zip"
    if os.path.exists(best_model_path):
        test_model(best_model_path, n_episodes, use_gui)
    else:
        print(f"❌ Лучшая модель не найдена: {best_model_path}")
        print("\nПопробуйте использовать финальную модель:")
        final_model_path = "RL3/models/ppo_robot_visual_final.zip"
        if os.path.exists(final_model_path):
            test_model(final_model_path, n_episodes, use_gui)
        else:
            print(f"❌ Финальная модель также не найдена: {final_model_path}")
            print("\nСначала обучите модель, запустив:")
            print("  python RL3/train_visual_robot.py")


def interactive_test():
    """Интерактивный режим тестирования"""
    print("=" * 60)
    print("ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 60)
    
    # Список доступных моделей
    models_dir = "RL3/models"
    if not os.path.exists(models_dir):
        print(f"❌ Директория моделей не найдена: {models_dir}")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    
    if not models:
        print("❌ Модели не найдены. Сначала обучите модель.")
        return
    
    print("\nДоступные модели:")
    for i, model in enumerate(models):
        print(f"  {i+1}. {model}")
    
    # Выбор модели
    choice = input("\nВыберите модель (номер или путь): ")
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            model_path = os.path.join(models_dir, models[idx])
        else:
            print("❌ Неверный номер")
            return
    except ValueError:
        model_path = choice
    
    # Параметры тестирования
    n_episodes = int(input("Количество эпизодов (по умолчанию 10): ") or "10")
    use_gui = input("Показывать GUI? (y/n, по умолчанию y): ").lower() != 'n'
    
    # Запуск теста
    test_model(model_path, n_episodes, use_gui)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Если передан аргумент - путь к модели
        model_path = sys.argv[1]
        n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        test_model(model_path, n_episodes, use_gui=True)
    else:
        # Интерактивный режим
        interactive_test()
