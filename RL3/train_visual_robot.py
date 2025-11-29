"""
Скрипт для обучения агента с использованием Transfer Learning и Stable-Baselines3.
Использует PPO с MobileNetV3 feature extractor.

Поддерживает 3 режима обучения (curriculum learning):
  - "reach": Дотянуться до объекта (самая простая задача, начните с неё!)
  - "grasp": Дотянуться и схватить объект  
  - "transfer": Схватить и перенести объект к цели (полная задача)
"""
import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from robot_visual_env import RobotArmEnv
from feature_extractor import CustomCNNFeatureExtractor


def save_model_config(model_path, config):
    """
    Сохранение конфигурации модели в JSON файл
    """
    config_path = model_path.replace('.zip', '_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ Конфигурация сохранена в {config_path}")
    return config_path


class EarlyStoppingCallback(BaseCallback):
    """
    Остановка обучения если нет прогресса в течение patience обновлений.
    """
    def __init__(self, patience=20, min_improvement=0.01, verbose=1):
        super().__init__(verbose)
        self.patience = patience  # Количество оценок без улучшения до остановки
        self.min_improvement = min_improvement  # Минимальное улучшение
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        # Проверяем каждые 10000 шагов
        if self.n_calls % 10000 == 0 and self.n_calls > 0:
            # Получаем последние награды из логов
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards)
                
                if mean_reward > self.best_mean_reward + self.min_improvement:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"\n📈 Новый лучший результат: {mean_reward:.2f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"\n⏳ Нет улучшения ({self.no_improvement_count}/{self.patience}), "
                              f"лучший: {self.best_mean_reward:.2f}, текущий: {mean_reward:.2f}")
                
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        print(f"\n🛑 EARLY STOPPING: нет улучшения {self.patience} раз подряд")
                    return False
        return True


def make_env(rank=0, image_size=84, use_grayscale=False, frame_skip=4, frame_stack=4, task="reach"):
    """
    Создание обернутой среды для обучения
    """
    def _init():
        env = RobotArmEnv(
            use_gui=False,
            image_size=image_size,
            use_grayscale=use_grayscale,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            task=task
        )
        env = Monitor(env)
        return env
    return _init


def plot_training_results(log_dir, save_path):
    """
    Построение графиков обучения
    """
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    results = load_results(log_dir)
    
    # Скользящее среднее
    def moving_average(values, window):
        # Убедиться что values это numpy array с числовым типом
        values = np.array(values, dtype=np.float64)
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')
    
    x, y = ts2xy(results, 'timesteps')
    
    # Преобразовать в numpy arrays
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График 1: Награда
    ax1.plot(x, y, alpha=0.3, label='Raw')
    if len(y) >= 50:
        y_smooth = moving_average(y, 50)
        ax1.plot(x[len(x)-len(y_smooth):], y_smooth, label='Moving Average (50 episodes)')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress: Episode Reward')
    ax1.legend()
    ax1.grid(True)
    
    # График 2: Длина эпизода
    x_ep, y_ep = ts2xy(results, 'episodes')
    ep_lengths = np.array(results['l'].values, dtype=np.float64)
    
    ax2.plot(range(len(ep_lengths)), ep_lengths, alpha=0.3, label='Raw')
    if len(ep_lengths) >= 50:
        ep_smooth = moving_average(ep_lengths, 50)
        ax2.plot(range(len(ep_lengths)-len(ep_smooth), len(ep_lengths)), ep_smooth, 
                label='Moving Average (50 episodes)')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress: Episode Length')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ График обучения сохранен в {save_path}")


def train():
    """
    Основная функция обучения
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ ВИЗУАЛЬНОГО УПРАВЛЕНИЯ РОБОТОМ FRANKA PANDA")
    print("=" * 60)
    
    # ========== ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ==========
    # 🔥 МЕНЯЙТЕ ТОЛЬКО ЭТИ ПАРАМЕТРЫ! 🔥
    
    # РЕЖИМ ЗАДАЧИ (curriculum learning - начинайте с простой!)
    # "reach"    - Только дотянуться до объекта (НАЧНИТЕ С ЭТОГО!)
    # "grasp"    - Дотянуться и схватить объект
    # "transfer" - Полная задача: схватить и перенести к цели
    TASK = "reach"  # <-- МЕНЯЙТЕ ЗДЕСЬ!
    
    # ДООБУЧЕНИЕ: загрузить предыдущую модель или начать с нуля?
    # None = обучение с нуля
    # "RL3/models/reach_rgb84_200k_best.zip" = дообучение этой модели
    PRETRAINED_MODEL = None  # <-- Путь к модели для дообучения или None
    
    # Параметры среды
    IMAGE_SIZE = 84  # 64 (быстро), 84 (стандарт), 128 (медленно)
    USE_GRAYSCALE = False  # True = Grayscale, False = RGB
    FRAME_SKIP = 4  # Количество повторений действия
    FRAME_STACK = 4  # Количество кадров в стеке
    
    # Параметры обучения
    TOTAL_TIMESTEPS = 200_000  # reach: 100-200K, grasp: 200-300K, transfer: 500K+
    N_ENVS = 4  # Параллельные среды
    USE_SUBPROC = True  # Параллельные процессы
    
    # Early stopping (остановка при отсутствии прогресса)
    USE_EARLY_STOPPING = True
    PATIENCE = 15  # Остановка после N проверок без улучшения
    
    # PPO параметры
    N_STEPS = 1024
    
    # Автоматическое округление timesteps
    steps_per_update = N_STEPS * N_ENVS
    actual_timesteps = ((TOTAL_TIMESTEPS + steps_per_update - 1) // steps_per_update) * steps_per_update
    if actual_timesteps != TOTAL_TIMESTEPS:
        print(f"\n⚠ TOTAL_TIMESTEPS округлено: {TOTAL_TIMESTEPS:,} → {actual_timesteps:,}")
        print(f"  (PPO тренирует блоками по {steps_per_update:,} шагов)")
        TOTAL_TIMESTEPS = actual_timesteps
    
    # Автоматическая генерация имени эксперимента
    mode = "gray" if USE_GRAYSCALE else "rgb"
    timesteps_k = TOTAL_TIMESTEPS // 1000
    EXPERIMENT_NAME = f"{TASK}_{mode}{IMAGE_SIZE}_{timesteps_k}k"
    
    # Директории
    models_dir = "RL3/models"
    logs_dir = f"RL3/logs/{EXPERIMENT_NAME}"
    tensorboard_dir = f"RL3/tensorboard/{EXPERIMENT_NAME}"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Конфигурация модели для сохранения
    model_config = {
        "experiment_name": EXPERIMENT_NAME,
        "task": TASK,
        "image_size": IMAGE_SIZE,
        "use_grayscale": USE_GRAYSCALE,
        "frame_skip": FRAME_SKIP,
        "frame_stack": FRAME_STACK,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_envs": N_ENVS,
        "use_subproc": USE_SUBPROC,
        "algorithm": "PPO",
        "feature_extractor": "MobileNetV3-Small"
    }
    
    # Описание задач
    task_descriptions = {
        "reach": "🎯 ЗАДАЧА: REACH - дотянуться до объекта (касание = успех)",
        "grasp": "🎯 ЗАДАЧА: GRASP - дотянуться и схватить объект",
        "transfer": "🎯 ЗАДАЧА: TRANSFER - схватить и перенести объект к цели"
    }
    
    print(f"\n📊 ЭКСПЕРИМЕНТ: {EXPERIMENT_NAME}")
    print(f"\n{task_descriptions.get(TASK, TASK)}")
    print(f"\nПараметры среды:")
    print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  - Image mode: {'Grayscale' if USE_GRAYSCALE else 'RGB'}")
    print(f"  - Frame skip: {FRAME_SKIP}")
    print(f"  - Frame stack: {FRAME_STACK}")
    print(f"\nПараметры обучения:")
    print(f"  - Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  - Parallel environments: {N_ENVS}")
    print(f"  - Early stopping: {'Да (patience={})'.format(PATIENCE) if USE_EARLY_STOPPING else 'Нет'}")
    print(f"  - Feature extractor: CustomCNN (обучается с нуля)")
    print(f"  - Algorithm: PPO")
    
    # Проверка CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Создание векторизованной среды
    print("\n⏳ Создание среды...")
    
    if USE_SUBPROC and N_ENVS > 1:
        env = SubprocVecEnv([make_env(i, IMAGE_SIZE, USE_GRAYSCALE, FRAME_SKIP, FRAME_STACK, TASK) for i in range(N_ENVS)])
        print(f"✓ SubprocVecEnv ({N_ENVS} процессов)")
    else:
        env = DummyVecEnv([make_env(i, IMAGE_SIZE, USE_GRAYSCALE, FRAME_SKIP, FRAME_STACK, TASK) for i in range(N_ENVS)])
        print(f"✓ DummyVecEnv")
    
    env = VecMonitor(env, logs_dir)
    
    # Среда для валидации
    eval_env = DummyVecEnv([make_env(0, IMAGE_SIZE, USE_GRAYSCALE, FRAME_SKIP, FRAME_STACK, TASK)])
    eval_env = VecMonitor(eval_env, logs_dir)
    
    print("✓ Среда создана")
    
    # Настройка политики с CustomCNN feature extractor
    # CustomCNN обучается с нуля - лучше для простой графики PyBullet
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 128], vf=[256, 128])  # Меньше, т.к. CNN обучаемая
    )
    
    # Создание или загрузка модели
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        print(f"\n⏳ Загрузка предобученной модели: {PRETRAINED_MODEL}")
        model = PPO.load(
            PRETRAINED_MODEL,
            env=env,
            device=device,
            tensorboard_log=tensorboard_dir,
            # Можно изменить learning rate для fine-tuning
            learning_rate=1e-4,  # Меньше для дообучения
        )
        print(f"✓ Модель загружена для дообучения")
        print(f"  Режим: Fine-tuning (learning_rate=1e-4)")
    else:
        if PRETRAINED_MODEL:
            print(f"\n⚠ Предобученная модель не найдена: {PRETRAINED_MODEL}")
            print("  Начинаем обучение с нуля...")
        
        print(f"\n⏳ Создание новой модели PPO...")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=512,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device=device
        )
        print("✓ Новая модель создана")
    
    print(f"\nАрхитектура политики:")
    print(f"  - Features extractor: CustomCNN (обучаемая с нуля)")
    print(f"  - Features dim: 256")
    print(f"  - Policy network: [256, 128]")
    print(f"  - Value network: [256, 128]")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=models_dir,
        name_prefix=EXPERIMENT_NAME
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    
    # Wrapper для переименования best_model в нужное имя
    
    class BestModelRenameCallback(BaseCallback):
        def __init__(self, eval_callback, experiment_name, models_dir, model_config, verbose=0):
            super().__init__(verbose)
            self.eval_callback = eval_callback
            self.experiment_name = experiment_name
            self.models_dir = models_dir
            self.model_config = model_config
            self.last_mean_reward = -np.inf
            
        def _on_step(self) -> bool:
            # Проверяем, обновилась ли лучшая модель
            if hasattr(self.eval_callback, 'best_mean_reward'):
                if self.eval_callback.best_mean_reward > self.last_mean_reward:
                    self.last_mean_reward = self.eval_callback.best_mean_reward
                    # Переименовываем best_model.zip в наше имя
                    default_path = os.path.join(self.models_dir, "best_model.zip")
                    new_path = os.path.join(self.models_dir, f"{self.experiment_name}_best.zip")
                    if os.path.exists(default_path):
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        os.rename(default_path, new_path)
                        save_model_config(new_path, self.model_config)
                        if self.verbose > 0:
                            print(f"✓ Лучшая модель сохранена: {new_path}")
            return True
    
    rename_callback = BestModelRenameCallback(eval_callback, EXPERIMENT_NAME, models_dir, model_config)
    
    # Список callbacks
    callbacks = [checkpoint_callback, eval_callback, rename_callback]
    
    # Добавляем early stopping если включен
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStoppingCallback(patience=PATIENCE, min_improvement=1.0, verbose=1)
        callbacks.append(early_stopping)
    
    callback_list = CallbackList(callbacks)
    
    # Обучение
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Логи TensorBoard: tensorboard --logdir={tensorboard_dir}")
    if USE_EARLY_STOPPING:
        print(f"Early stopping: остановка после {PATIENCE} проверок без улучшения")
    print("=" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            progress_bar=True
        )
        
        # Сохранение финальной модели
        final_model_path = os.path.join(models_dir, f"{EXPERIMENT_NAME}_final.zip")
        model.save(final_model_path)
        save_model_config(final_model_path, model_config)
        print(f"\n✓ Финальная модель сохранена в {final_model_path}")
        
        # Сохранение конфигурации для best_model если она существует
        best_model_path = os.path.join(models_dir, f"{EXPERIMENT_NAME}_best.zip")
        if os.path.exists(best_model_path):
            save_model_config(best_model_path, model_config)
        print(f"\n✓ Финальная модель сохранена в {final_model_path}")
        
        # Построение графиков
        print("\n⏳ Построение графиков обучения...")
        plot_path = f"./RL3/{EXPERIMENT_NAME}_training.png"
        plot_training_results(logs_dir, plot_path)
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        print(f"\nРезультаты сохранены в:")
        print(f"  - Модели: {models_dir}")
        print(f"  - Логи: {logs_dir}")
        print(f"  - TensorBoard: {tensorboard_dir}")
        print(f"  - График: ./RL3/training_results.png")
        
        print("\nДля просмотра логов TensorBoard выполните:")
        print(f"  tensorboard --logdir={tensorboard_dir}")
        
        print("\nДля тестирования обученной модели запустите:")
        print("  python ./RL3/test_trained_model.py")
        
    except KeyboardInterrupt:
        print("\n⚠ Обучение прервано пользователем")
        interrupted_path = os.path.join(models_dir, f"{EXPERIMENT_NAME}_interrupted.zip")
        model.save(interrupted_path)
        save_model_config(interrupted_path, model_config)
        print("✓ Промежуточная модель сохранена")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    # Установка seed для воспроизводимости
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train()
