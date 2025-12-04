"""
Скрипт обучения для задачи pick-and-place.

Вход: только пиксели (64x64 grayscale)
Модель: MobileNetV3-Small с заморозкой слоёв
Цель: 1-2M шагов для достижения успеха
"""
import os
import json
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

# Добавляем путь к RL3 для импортов
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from robot_env import RobotEnv
from feature_extractor import MobileNetExtractor, EfficientNetExtractor, SimpleCNNExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Train pick-and-place robot')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total timesteps (default: 1M)')
    parser.add_argument('--network', type=str, default='mobilenet',
                       choices=['mobilenet', 'efficientnet', 'cnn'],
                       help='Network architecture')
    parser.add_argument('--algo', type=str, default='ppo',
                       choices=['ppo', 'sac'],
                       help='RL algorithm')
    parser.add_argument('--camera', type=str, default='side',
                       choices=['side', 'wrist', 'both'],
                       help='Camera mode')
    parser.add_argument('--freeze', type=int, default=8,
                       help='Number of layers to freeze in pretrained model')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size (64 recommended)')
    parser.add_argument('--eval_freq', type=int, default=25000,
                       help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--render', type=int, default=2,
                       help='Episodes to render at each checkpoint (0 = disable)')
    return parser.parse_args()


class ProgressCallback(BaseCallback):
    """Показывает прогресс и статистику"""
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_success_rate = 0
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Собираем информацию об успехах
        for info in self.locals.get('infos', []):
            if 'success' in info:
                self.episode_successes.append(info['success'])
        
        if self.n_calls % self.check_freq == 0 and len(self.model.ep_info_buffer) > 0:
            rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            
            # Успешность за последние эпизоды
            recent_successes = self.episode_successes[-100:] if self.episode_successes else []
            success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
            
            marker = ""
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                marker = " 🎯 NEW BEST!"
            
            print(f"\nStep {self.n_calls:,}: "
                  f"Mean R = {mean_reward:.2f}, "
                  f"Success = {success_rate:.1f}%{marker}")
        
        return True


class SuccessRateCallback(BaseCallback):
    """Логирует success rate для TensorBoard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'success' in info:
                self.successes.append(float(info['success']))
        
        if len(self.successes) >= 100:
            success_rate = np.mean(self.successes[-100:])
            self.logger.record('rollout/success_rate', success_rate)
        
        return True


class PeriodicRenderCallback(BaseCallback):
    """
    Периодически сохраняет модель и запускает рендер-тест.
    Checkpoints: каждые 10k до 100k, каждые 50k до 500k, каждые 200k после.
    """
    def __init__(self, models_dir, exp_name, env_kwargs, total_steps, 
                 render_episodes=2, verbose=1):
        super().__init__(verbose)
        self.models_dir = models_dir
        self.exp_name = exp_name
        self.env_kwargs = env_kwargs
        self.total_steps = total_steps
        self.render_episodes = render_episodes
        self.render_process = None
        
        # Генерируем список checkpoints заранее
        self.checkpoints = self._generate_checkpoints(total_steps)
        self.next_checkpoint_idx = 0
        
    def _generate_checkpoints(self, total_steps):
        """Генерирует список checkpoints"""
        checkpoints = []
        
        # Каждые 10k до 100k
        for step in range(10000, min(100001, total_steps + 1), 10000):
            checkpoints.append(step)
        
        # Каждые 50k от 100k до 500k  
        for step in range(150000, min(500001, total_steps + 1), 50000):
            checkpoints.append(step)
        
        # Каждые 200k после 500k
        for step in range(600000, total_steps + 1, 200000):
            checkpoints.append(step)
        
        return sorted(set(checkpoints))
    
    def _on_step(self) -> bool:
        # Проверяем достигли ли следующего checkpoint
        if self.next_checkpoint_idx >= len(self.checkpoints):
            return True
            
        next_checkpoint = self.checkpoints[self.next_checkpoint_idx]
        
        if self.num_timesteps >= next_checkpoint:
            self.next_checkpoint_idx += 1
            
            # Сохраняем модель (абсолютный путь)
            save_path = os.path.abspath(os.path.join(
                self.models_dir, 
                f"{self.exp_name}_{next_checkpoint//1000}k.zip"
            ))
            self.model.save(save_path)
            
            if self.verbose:
                print(f"\n[CHECKPOINT {next_checkpoint//1000}k]")
                print(f"  Model saved: {save_path}")
                print(f"  Starting render test ({self.render_episodes} episodes)...")
            
            # Запускаем рендер в отдельном процессе
            self._start_render_test(save_path)
        
        return True
    
    def _start_render_test(self, model_path):
        """Запускает тест модели в отдельном процессе с GUI окном"""
        import subprocess
        import sys
        
        # Завершаем предыдущий процесс если ещё работает
        if self.render_process is not None:
            try:
                self.render_process.terminate()
                self.render_process.wait(timeout=2)
            except:
                pass
        
        # Получаем абсолютный путь к test.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_script = os.path.join(script_dir, "test.py")
        
        # Запускаем новый процесс рендера в отдельном окне консоли
        # start с заголовком окна выводит его на передний план
        cmd = f'start "=== ROBOT RENDER TEST ===" cmd /c "python \\"{test_script}\\" \\"{model_path}\\" --episodes {self.render_episodes} --gui && pause"'
        
        try:
            # Используем shell=True с start для создания активного окна
            self.render_process = subprocess.Popen(
                cmd,
                shell=True
            )
            if self.verbose:
                print(f"  [OK] Render window opened")
        except Exception as e:
            if self.verbose:
                print(f"  [!] Failed to start render: {e}")
    
    def _on_training_end(self):
        """Ждем пока пользователь закроет окно рендера"""
        if self.render_process is not None:
            try:
                # Ждем пока процесс завершится (пользователь закроет окно)
                self.render_process.wait()
            except:
                pass


def make_env(rank, use_gui=False, **env_kwargs):
    def _init():
        env = RobotEnv(use_gui=use_gui, **env_kwargs)
        env = Monitor(env)
        return env
    return _init


def train(args):
    print("=" * 60)
    print("PICK-AND-PLACE ROBOT TRAINING")
    print("=" * 60)
    
    # Experiment name
    exp_name = f"pickplace_{args.network}_{args.camera}_{args.algo}_{args.steps//1000}k"
    
    # Directories (абсолютные пути относительно RL3)
    rl3_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(rl3_dir, "models")
    logs_dir = os.path.join(rl3_dir, "logs", exp_name)
    tensorboard_dir = os.path.join(rl3_dir, "tensorboard", exp_name)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Environment kwargs
    env_kwargs = {
        'image_size': args.image_size,
        'frame_stack': 4,
        'camera_mode': args.camera,
        'max_steps': 200
    }
    
    # Config
    config = {
        "experiment_name": exp_name,
        "network": args.network,
        "algorithm": args.algo,
        "camera_mode": args.camera,
        "image_size": args.image_size,
        "freeze_layers": args.freeze,
        "total_timesteps": args.steps,
        "n_envs": args.n_envs,
        "seed": args.seed
    }
    
    print(f"\nExperiment: {exp_name}")
    print(f"\nSettings:")
    print(f"  - Network: {args.network}")
    print(f"  - Algorithm: {args.algo}")
    print(f"  - Camera: {args.camera}")
    print(f"  - Image size: {args.image_size}x{args.image_size}")
    print(f"  - Freeze layers: {args.freeze}")
    print(f"  - Total steps: {args.steps:,}")
    print(f"  - Parallel envs: {args.n_envs}")
    if args.render > 0:
        print(f"  - Periodic render: {args.render} episodes at checkpoints")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    
    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environments
    print("\nCreating environments...")
    
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, use_gui=False, **env_kwargs) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0, use_gui=False, **env_kwargs)])
    env = VecMonitor(env, logs_dir)
    
    eval_env = DummyVecEnv([make_env(0, use_gui=False, **env_kwargs)])
    eval_env = VecMonitor(eval_env, logs_dir)
    
    print("[OK] Environments created")
    
    # Select feature extractor
    if args.network == "mobilenet":
        extractor_class = MobileNetExtractor
        extractor_kwargs = {"features_dim": 256, "freeze_layers": args.freeze}
    elif args.network == "efficientnet":
        extractor_class = EfficientNetExtractor
        extractor_kwargs = {"features_dim": 256, "freeze_layers": args.freeze}
    else:
        extractor_class = SimpleCNNExtractor
        extractor_kwargs = {"features_dim": 256}
    
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )
    
    # Create model
    print(f"\nCreating {args.algo.upper()} model...")
    
    if args.algo == "ppo":
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device=device,
            seed=args.seed
        )
    else:  # SAC
        model = SAC(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            device=device,
            seed=args.seed
        )
    
    print("[OK] Model created")
    
    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    progress_cb = ProgressCallback(check_freq=10000)
    success_cb = SuccessRateCallback()
    
    callbacks_list = [eval_cb, progress_cb, success_cb]
    
    # Добавляем PeriodicRenderCallback если нужен рендер
    if args.render > 0:
        render_cb = PeriodicRenderCallback(
            models_dir=models_dir,
            exp_name=exp_name,
            env_kwargs=env_kwargs,
            total_steps=args.steps,
            render_episodes=args.render
        )
        callbacks_list.append(render_cb)
    
    callbacks = CallbackList(callbacks_list)
    
    # Training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    if args.render > 0:
        print(f"  Periodic render: {args.render} episodes at checkpoints")
        print(f"  Save intervals: 10k -> 50k -> 200k steps")
    print("=" * 60)
    print(f"TensorBoard: tensorboard --logdir={tensorboard_dir}")
    print("=" * 60 + "\n")
    
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(models_dir, f"{exp_name}_final.zip")
        model.save(final_path)
        
        # Save config
        config_path = final_path.replace('.zip', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n[OK] Final model saved: {final_path}")
        
        # Rename best_model
        best_src = os.path.join(models_dir, "best_model.zip")
        best_dst = os.path.join(models_dir, f"{exp_name}_best.zip")
        if os.path.exists(best_src):
            if os.path.exists(best_dst):
                os.remove(best_dst)
            os.rename(best_src, best_dst)
            
            config_dst = best_dst.replace('.zip', '_config.json')
            with open(config_dst, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"[OK] Best model saved: {best_dst}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted")
        interrupted_path = os.path.join(models_dir, f"{exp_name}_interrupted.zip")
        model.save(interrupted_path)
        print(f"[OK] Model saved: {interrupted_path}")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
