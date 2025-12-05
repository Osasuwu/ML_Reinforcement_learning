"""
Скрипт для продолжения обучения прерванной модели.

Использование:
    python RL3/continue_training.py                                    # Последняя interrupted модель
    python RL3/continue_training.py --model models/some_model.zip      # Конкретная модель
    python RL3/continue_training.py --steps 2000000                    # Добавить 2M шагов
"""
import os
import json
import sys
import argparse
import glob

# Добавляем путь к RL3 для импортов
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList
from robot_env import RobotEnv
from train import PeriodicRenderCallback, EarlyStoppingCallback, make_env


def find_latest_model(models_dir):
    """Находит последнюю interrupted или лучшую модель"""
    # Сначала ищем interrupted
    pattern = os.path.join(models_dir, "*_interrupted.zip")
    interrupted = glob.glob(pattern)
    if interrupted:
        # Берём самую свежую по времени модификации
        return max(interrupted, key=os.path.getmtime)
    
    # Если нет interrupted, ищем модели с числом шагов
    pattern = os.path.join(models_dir, "*_*k.zip")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        # Сортируем по числу шагов
        def get_steps(path):
            name = os.path.basename(path)
            import re
            match = re.search(r'_(\d+)k\.zip$', name)
            return int(match.group(1)) if match else 0
        return max(checkpoints, key=get_steps)
    
    return None


def find_config(model_path):
    """Находит конфиг для модели"""
    import re
    
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # Вариант 1: точное совпадение
    config_path = model_path.replace('.zip', '_config.json')
    if os.path.exists(config_path):
        return config_path
    
    # Вариант 2: общий конфиг эксперимента
    base_path = re.sub(r'_\d+k\.zip$', '_config.json', model_path)
    if base_path != model_path and os.path.exists(base_path):
        return base_path
    
    # Вариант 3: для interrupted/continued
    base_path = re.sub(r'_(interrupted|continued|final|best)\.zip$', '_config.json', model_path)
    if base_path != model_path and os.path.exists(base_path):
        return base_path
    
    # Вариант 4: ищем любой подходящий config
    match = re.match(r'(.+?)_(\d+k|interrupted|continued|final|best)', model_name)
    if match:
        exp_base = match.group(1)
        for fname in os.listdir(model_dir):
            if fname.startswith(exp_base) and fname.endswith('_config.json'):
                return os.path.join(model_dir, fname)
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: latest interrupted)')
    parser.add_argument('--steps', type=int, default=5_000_000,
                       help='Additional steps to train (default: 5M)')
    parser.add_argument('--n-envs', type=int, default=8,
                       help='Number of parallel environments (default: 8)')
    args = parser.parse_args()
    
    models_dir = os.path.join(script_dir, "models")
    logs_dir = os.path.join(script_dir, "logs")
    
    print("=" * 60)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("=" * 60)
    
    # Находим модель
    if args.model:
        model_path = args.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
    else:
        model_path = find_latest_model(models_dir)
    
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("\nAvailable models:")
        for f in sorted(os.listdir(models_dir)):
            if f.endswith('.zip'):
                print(f"  {f}")
        return
    
    print(f"\n[OK] Model: {os.path.basename(model_path)}")
    
    # Загружаем конфиг
    config_path = find_config(model_path)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"[OK] Config: {os.path.basename(config_path)}")
    else:
        print(f"[WARN] Config not found, using defaults")
        config = {}
    
    # Строим env_kwargs из конфига или используем defaults
    env_kwargs = config.get('env_kwargs', None)
    if env_kwargs is None:
        # Строим из отдельных параметров конфига
        camera_mode = config.get('camera_mode', 'side')
        image_size = config.get('image_size', 64)
        env_kwargs = {
            'image_size': image_size,
            'frame_stack': 4,
            'camera_mode': camera_mode,
            'max_steps': 200,
        }
    
    print(f"\nEnvironment config:")
    for k, v in env_kwargs.items():
        print(f"  {k}: {v}")
    
    # Создаём среды
    n_envs = args.n_envs
    print(f"\nCreating {n_envs} parallel environments...")
    
    envs = SubprocVecEnv([make_env(i, False, **env_kwargs) for i in range(n_envs)])
    envs = VecMonitor(envs, logs_dir)
    
    # Определяем алгоритм по имени файла
    algo_class = PPO
    if 'sac' in os.path.basename(model_path).lower():
        algo_class = SAC
    
    # Загружаем модель
    print(f"\nLoading model...")
    model = algo_class.load(model_path, env=envs)
    
    # Проверяем сколько уже обучено
    current_steps = model.num_timesteps
    print(f"Current timesteps: {current_steps:,}")
    print(f"Additional steps: {args.steps:,}")
    print(f"Total after training: {current_steps + args.steps:,}")
    
    # Определяем имя эксперимента из пути модели
    import re
    model_name = os.path.basename(model_path)
    match = re.match(r'(.+?)_(\d+k|interrupted|continued)', model_name)
    if match:
        exp_name = match.group(1)
    else:
        exp_name = model_name.replace('.zip', '')
    
    print(f"Experiment name: {exp_name}")
    
    # Callbacks
    callbacks = []
    
    # Periodic render callback
    render_callback = PeriodicRenderCallback(
        models_dir=models_dir,
        exp_name=exp_name,
        env_kwargs=env_kwargs,
        total_steps=current_steps + args.steps,
        render_episodes=2,
        verbose=1
    )
    # Смещаем индекс чтобы не повторять уже пройденные checkpoints
    while (render_callback.next_save_idx < len(render_callback.save_checkpoints) and 
           render_callback.save_checkpoints[render_callback.next_save_idx] <= current_steps):
        render_callback.next_save_idx += 1
    while (render_callback.next_render_idx < len(render_callback.render_checkpoints) and 
           render_callback.render_checkpoints[render_callback.next_render_idx] <= current_steps):
        render_callback.next_render_idx += 1
    
    callbacks.append(render_callback)
    
    # Примечание: EarlyStoppingCallback требует success_callback,
    # который создаётся в train.py. Для continue training пропускаем его.
    
    callback_list = CallbackList(callbacks)
    
    # Запускаем обучение
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback_list,
            reset_num_timesteps=False,  # Продолжаем счётчик
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user")
    finally:
        # Сохраняем финальную модель
        final_path = os.path.join(models_dir, f"{exp_name}_continued.zip")
        model.save(final_path)
        print(f"\n[OK] Model saved: {final_path}")
        print(f"Total timesteps: {model.num_timesteps:,}")
        
        envs.close()


if __name__ == "__main__":
    main()
