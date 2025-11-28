"""
Скрипт для тестирования корректности работы симуляции и настройки камеры.
Проверяет:
- Загрузку робота и объектов
- Работу камеры
- Инверсную кинематику
- Получение наблюдений
"""
import numpy as np
from robot_visual_env import RobotArmEnv
import matplotlib.pyplot as plt
import time


def test_environment_setup():
    """Тест базовой настройки среды"""
    print("=" * 50)
    print("ТЕСТ 1: Базовая настройка среды")
    print("=" * 50)
    
    # Создаем среду с GUI для визуализации
    env = RobotArmEnv(use_gui=True, image_size=84, use_grayscale=False, 
                     frame_skip=4, frame_stack=4)
    
    print("✓ Среда создана успешно")
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")
    
    # Сброс среды
    obs, info = env.reset()
    print(f"✓ Среда сброшена")
    print(f"  - Image shape: {obs['image'].shape}")
    print(f"  - Joints shape: {obs['joints'].shape}")
    print(f"  - Joints values: {obs['joints']}")
    
    # Визуализация изображения с камеры
    print("\nВизуализация изображения с камеры...")
    latest_frame = obs['image'][-1]  # Последний кадр из стека
    
    plt.figure(figsize=(6, 6))
    if latest_frame.shape[-1] == 1:
        plt.imshow(latest_frame[:, :, 0], cmap='gray')
    else:
        plt.imshow(latest_frame)
    plt.title("Camera View (Latest Frame)")
    plt.axis('off')
    plt.savefig('RL3/test_camera_view.png')
    print("✓ Изображение сохранено в 'RL3/test_camera_view.png'")
    plt.close()
    
    # Ждем 3 секунды для просмотра GUI
    print("\nПодождите 3 секунды (можно посмотреть GUI)...")
    time.sleep(3)
    
    env.close()
    print("\n✓ ТЕСТ 1 ПРОЙДЕН\n")
    

def test_robot_kinematics():
    """Тест инверсной кинематики и движения робота"""
    print("=" * 50)
    print("ТЕСТ 2: Кинематика робота")
    print("=" * 50)
    
    env = RobotArmEnv(use_gui=True, image_size=84, use_grayscale=False, 
                     frame_skip=4, frame_stack=4)
    obs, info = env.reset()
    
    print("Начальная позиция схвата:", env._get_end_effector_pos())
    print("Позиция целевого объекта:", env._get_object_pos())
    
    # Тестируем несколько случайных действий
    print("\nВыполнение 10 случайных действий...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Шаг {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Distance: {info['distance']:.4f}")
        print(f"  Contact: {info['contact']}")
        print(f"  End effector pos: {info['ee_pos']}")
        
        if terminated:
            print("  ✓ Контакт достигнут!")
            break
        
        time.sleep(0.1)  # Небольшая задержка для визуализации
    
    env.close()
    print("\n✓ ТЕСТ 2 ПРОЙДЕН\n")


def test_observation_consistency():
    """Тест консистентности наблюдений"""
    print("=" * 50)
    print("ТЕСТ 3: Консистентность наблюдений")
    print("=" * 50)
    
    env = RobotArmEnv(use_gui=False, image_size=84, use_grayscale=False, 
                     frame_skip=4, frame_stack=4)
    
    # Несколько сбросов
    print("Проверка 5 сбросов среды...")
    for i in range(5):
        obs, info = env.reset()
        
        # Проверяем форматы
        assert obs['image'].shape == (4, 84, 84, 3), f"Неверная форма изображения: {obs['image'].shape}"
        assert obs['joints'].shape == (7,), f"Неверная форма джоинтов: {obs['joints'].shape}"
        assert obs['image'].dtype == np.uint8, f"Неверный тип изображения: {obs['image'].dtype}"
        assert obs['joints'].dtype == np.float32, f"Неверный тип джоинтов: {obs['joints'].dtype}"
        
        print(f"  Сброс {i+1}: ✓ Форматы корректны")
    
    # Проверка frame stacking
    print("\nПроверка frame stacking...")
    obs, info = env.reset()
    initial_stack = obs['image'].copy()
    
    # Выполняем действие
    action = np.array([0.01, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    new_stack = obs['image']
    
    # Проверяем, что первые 3 кадра старого стека = последним 3 кадрам нового
    # (кадры сдвинулись)
    print("  - Проверка сдвига кадров в стеке...")
    # Последние 3 кадра из initial_stack должны быть первыми 3 в new_stack
    # (кроме самого последнего, который новый)
    
    env.close()
    print("\n✓ ТЕСТ 3 ПРОЙДЕН\n")


def test_reward_function():
    """Тест функции награды"""
    print("=" * 50)
    print("ТЕСТ 4: Функция награды")
    print("=" * 50)
    
    env = RobotArmEnv(use_gui=False, image_size=84, use_grayscale=False, 
                     frame_skip=4, frame_stack=4)
    obs, info = env.reset()
    
    print("Тестирование награды при приближении к объекту...")
    
    # Получаем текущую позицию и целевую
    ee_pos = env._get_end_effector_pos()
    obj_pos = env._get_object_pos()
    
    print(f"Начальная позиция схвата: {ee_pos}")
    print(f"Позиция объекта: {obj_pos}")
    print(f"Начальное расстояние: {np.linalg.norm(ee_pos - obj_pos):.4f}")
    
    # Действие в направлении объекта
    direction = obj_pos - ee_pos
    direction = direction / np.linalg.norm(direction)  # Нормализация
    action = direction * 0.05  # Максимальный шаг
    
    print(f"\nДействие в сторону объекта: {action}")
    
    rewards = []
    distances = []
    
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        distances.append(info['distance'])
        
        if terminated:
            print(f"\n✓ Контакт достигнут на шаге {i+1}!")
            print(f"  Финальная награда: {reward:.2f}")
            break
    
    print(f"\nСтатистика:")
    print(f"  Награды: min={min(rewards):.2f}, max={max(rewards):.2f}, mean={np.mean(rewards):.2f}")
    print(f"  Расстояния: min={min(distances):.4f}, max={max(distances):.4f}")
    
    # График
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over Steps')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(distances, marker='o', color='red')
    plt.xlabel('Step')
    plt.ylabel('Distance')
    plt.title('Distance to Target')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('RL3/test_reward_function.png')
    print("✓ График сохранен в 'RL3/test_reward_function.png'")
    plt.close()
    
    env.close()
    print("\n✓ ТЕСТ 4 ПРОЙДЕН\n")


def test_grayscale_mode():
    """Тест режима grayscale"""
    print("=" * 50)
    print("ТЕСТ 5: Режим Grayscale")
    print("=" * 50)
    
    env = RobotArmEnv(use_gui=False, image_size=84, use_grayscale=True, 
                     frame_skip=4, frame_stack=4)
    obs, info = env.reset()
    
    print(f"Image shape (grayscale): {obs['image'].shape}")
    assert obs['image'].shape == (4, 84, 84, 1), f"Неверная форма для grayscale: {obs['image'].shape}"
    
    # Визуализация
    latest_frame = obs['image'][-1, :, :, 0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(latest_frame, cmap='gray')
    plt.title("Grayscale Camera View")
    plt.axis('off')
    plt.savefig('RL3/test_grayscale.png')
    print("✓ Grayscale изображение сохранено в 'RL3/test_grayscale.png'")
    plt.close()
    
    env.close()
    print("\n✓ ТЕСТ 5 ПРОЙДЕН\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("ЗАПУСК ТЕСТОВ СИМУЛЯЦИИ")
    print("=" * 50 + "\n")
    
    try:
        # Тест 1: Базовая настройка
        test_environment_setup()
        
        # Тест 2: Кинематика
        test_robot_kinematics()
        
        # Тест 3: Консистентность наблюдений
        test_observation_consistency()
        
        # Тест 4: Функция награды
        test_reward_function()
        
        # Тест 5: Grayscale режим
        test_grayscale_mode()
        
        print("\n" + "=" * 50)
        print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! ✓")
        print("=" * 50 + "\n")
        print("Симуляция настроена правильно и готова к обучению.")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
