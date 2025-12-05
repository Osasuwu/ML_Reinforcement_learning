"""
Скрипт для продолжения обучения прерванной модели.
Просто запустите: python RL3/continue_training.py

Модель будет загружена и обучение продолжится с того же места.
"""
import os
import json
import sys

# Добавляем путь к RL3 для импортов
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from robot_env import RobotEnv
from train import (PeriodicRenderCallback, EarlyStopCallback, 
                   CurriculumCallback, make_env)

# ============ НАСТРОЙКИ ============
# Путь к прерванной модели (измените если нужно)
MODEL_PATH = os.path.join(script_dir, "models", "pickplace_mobilenet_side+wrist_ppo_10000k_interrupted.zip")
CONFIG_PATH = os.path.join(script_dir, "models", "pickplace_mobilenet_side+wrist_ppo_10000k_config.json")

# Сколько ещё шагов обучать (добавьте нужное количество)
ADDITIONAL_STEPS = 5_000_000

# ===================================

def main():
    print("=" * 60)
    print("CONTINUING TRAINING FROM CHECKPOINT")
    print("=" * 60)
    
    # Проверяем существование файлов
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("\nAvailable models:")
        models_dir = os.path.join(script_dir, "models")
        for f in sorted(os.listdir(models_dir)):
            if f.endswith('.zip'):
                print(f"  {f}")
        return
    
    # Загружаем конфиг
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"[OK] Config loaded: {CONFIG_PATH}")
    else:
        print(f"[WARN] Config not found, using defaults")
        config = {
            'env_kwargs': {
                'render_mode': None,
                'image_size': 64,
                'frame_stack': 4,
                'camera_mode': 'side+wrist',
                'secondary_size': 32,
                'curriculum': True
            }
        }
    
    env_kwargs = config.get('env_kwargs', {})
    print(f"\nEnvironment config:")
    for k, v in env_kwargs.items():
        print(f"  {k}: {v}")
    
    # Создаём среды
    n_envs = 4
    print(f"\nCreating {n_envs} parallel environments...")
    
    envs = SubprocVecEnv([make_env(i, env_kwargs) for i in range(n_envs)])
    envs = VecMonitor(envs, os.path.join(script_dir, "logs"))
    
    # Загружаем модель
    print(f"\nLoading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=envs)
    
    # Проверяем сколько уже обучено
    current_steps = model.num_timesteps
    print(f"Current timesteps: {current_steps:,}")
    print(f"Will train for additional: {ADDITIONAL_STEPS:,}")
    print(f"Total after training: {current_steps + ADDITIONAL_STEPS:,}")
    
    # Callbacks
    exp_name = "pickplace_mobilenet_side+wrist_ppo_10000k"
    models_dir = os.path.join(script_dir, "models")
    logs_dir = os.path.join(script_dir, "logs")
    
    callbacks = []
    
    # Periodic render callback
    render_callback = PeriodicRenderCallback(
        models_dir=models_dir,
        exp_name=exp_name,
        env_kwargs=env_kwargs,
        total_steps=current_steps + ADDITIONAL_STEPS,
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
    
    # Curriculum callback если включён
    if env_kwargs.get('curriculum', False):
        callbacks.append(CurriculumCallback(verbose=1))
    
    # Early stop callback
    callbacks.append(EarlyStopCallback(
        target_success_rate=80.0,
        patience=500_000,
        min_timesteps=max(1_000_000, current_steps),
        models_dir=models_dir,
        exp_name=exp_name,
        verbose=1
    ))
    
    callback_list = CallbackList(callbacks)
    
    # Запускаем обучение
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=ADDITIONAL_STEPS,
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
