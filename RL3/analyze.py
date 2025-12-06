"""
Скрипт для анализа обучения и построения графиков.

Функции:
1. Построение графиков награды и успешности из monitor.csv
2. Парсинг TensorBoard логов
3. Анализ распределения действий
4. Анализ траекторий эндэффектора
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Для TensorBoard логов
try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available, some features disabled")


def load_monitor_csv(log_dir):
    """Загружает данные из monitor.csv файлов"""
    monitor_files = list(Path(log_dir).rglob("monitor.csv"))
    
    if not monitor_files:
        print(f"No monitor.csv files found in {log_dir}")
        return None
    
    all_data = []
    for f in monitor_files:
        try:
            # Пропускаем первую строку с метаданными
            df = pd.read_csv(f, skiprows=1)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not all_data:
        return None
    
    # Объединяем все данные
    combined = pd.concat(all_data, ignore_index=True)
    
    # Добавляем кумулятивные шаги
    combined['cumulative_steps'] = combined['l'].cumsum()
    
    return combined


def load_tensorboard_logs(log_dir):
    """Загружает данные из TensorBoard логов"""
    if not TENSORBOARD_AVAILABLE:
        return None
    
    # Ищем event файлы
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"No TensorBoard events found in {log_dir}")
        return None
    
    all_scalars = {}
    
    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            for tag in ea.Tags()['scalars']:
                if tag not in all_scalars:
                    all_scalars[tag] = {'steps': [], 'values': []}
                
                for event in ea.Scalars(tag):
                    all_scalars[tag]['steps'].append(event.step)
                    all_scalars[tag]['values'].append(event.value)
        except Exception as e:
            print(f"Error loading {event_file}: {e}")
    
    return all_scalars


def smooth(values, window=50):
    """Скользящее среднее"""
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid')


def plot_training_progress(log_dir, output_path=None, show=True):
    """
    Построение графиков прогресса обучения.
    
    Графики:
    1. Episode Reward (raw + smoothed)
    2. Episode Length
    3. Success Rate (если есть)
    4. Policy Loss / Value Loss
    """
    print(f"\nAnalyzing training logs from: {log_dir}")
    
    # Загружаем данные
    monitor_data = load_monitor_csv(log_dir)
    tb_data = load_tensorboard_logs(log_dir)
    
    if monitor_data is None and tb_data is None:
        print("No data found!")
        return
    
    # Создаём фигуру
    n_plots = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. Episode Reward
    if monitor_data is not None and 'r' in monitor_data.columns:
        ax = axes[plot_idx]
        rewards = monitor_data['r'].values
        steps = monitor_data['cumulative_steps'].values
        
        ax.plot(steps, rewards, alpha=0.3, color='blue', label='Raw')
        
        if len(rewards) >= 50:
            smoothed = smooth(rewards, 50)
            ax.plot(steps[len(steps)-len(smoothed):], smoothed, 
                   color='blue', linewidth=2, label='Smoothed (50 ep)')
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Training Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистику
        ax.axhline(y=np.mean(rewards), color='r', linestyle='--', alpha=0.5,
                  label=f'Mean: {np.mean(rewards):.2f}')
        
        plot_idx += 1
    
    # 2. Episode Length
    if monitor_data is not None and 'l' in monitor_data.columns:
        ax = axes[plot_idx]
        lengths = monitor_data['l'].values
        
        ax.plot(lengths, alpha=0.3, color='green')
        
        if len(lengths) >= 50:
            smoothed = smooth(lengths, 50)
            ax.plot(range(len(lengths)-len(smoothed), len(lengths)), smoothed,
                   color='green', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length (lower = faster success OR faster failure)')
        ax.grid(True, alpha=0.3)
        
        # Добавляем процентили
        ax.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Max steps')
        ax.legend()
        
        plot_idx += 1
    
    # 3. Success Rate (из TensorBoard)
    if tb_data and 'rollout/success_rate' in tb_data:
        ax = axes[plot_idx]
        data = tb_data['rollout/success_rate']
        
        ax.plot(data['steps'], data['values'], color='purple', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Добавляем целевую линию
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.legend()
        
        plot_idx += 1
    
    # 4. Losses (из TensorBoard)
    if tb_data:
        ax = axes[plot_idx]
        
        plotted = False
        for loss_tag in ['train/policy_gradient_loss', 'train/value_loss', 
                        'train/loss', 'train/entropy_loss']:
            if loss_tag in tb_data:
                data = tb_data[loss_tag]
                label = loss_tag.split('/')[-1]
                ax.plot(data['steps'], data['values'], label=label, alpha=0.7)
                plotted = True
        
        if plotted:
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Убираем пустые графики
    for i in range(plot_idx, n_plots):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Сохраняем
    if output_path is None:
        output_path = os.path.join(log_dir, 'training_analysis.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return monitor_data, tb_data


def analyze_action_distribution(model_path, n_episodes=3):
    """
    Анализирует распределение действий модели.
    Помогает понять, не "застряла" ли модель в определённых действиях.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from robot_env import RobotEnv
    
    print(f"\nAnalyzing action distribution for: {model_path}")
    
    # Создаём среду без обёрток для прямого доступа
    raw_env = RobotEnv(use_gui=False, image_size=64, camera_mode='side+depth')
    vec_env = DummyVecEnv([lambda: RobotEnv(use_gui=False, image_size=64, camera_mode='side+depth')])
    
    model = PPO.load(model_path, env=vec_env)
    
    # Собираем действия
    all_actions = []
    all_positions = []
    
    for ep in range(n_episodes):
        obs, _ = raw_env.reset()
        # Нужен формат для vec_env
        obs_dict = {k: np.expand_dims(v, 0) for k, v in obs.items()} if isinstance(obs, dict) else np.expand_dims(obs, 0)
        
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            action, _ = model.predict(obs_dict, deterministic=True)
            action = action[0]  # Убираем batch dimension
            all_actions.append(action.copy())
            
            # Сохраняем позицию end-effector
            ee_pos = raw_env._get_ee_pos()
            all_positions.append(ee_pos.copy())
            
            obs, _, term, trunc, _ = raw_env.step(action)
            obs_dict = {k: np.expand_dims(v, 0) for k, v in obs.items()} if isinstance(obs, dict) else np.expand_dims(obs, 0)
            
            done = term or trunc
            step_count += 1
        
        print(f"  Episode {ep+1}/{n_episodes}: {step_count} steps")
    
    raw_env.close()
    vec_env.close()
    
    all_actions = np.array(all_actions)
    all_positions = np.array(all_positions)
    
    # Строим графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение действий для каждого джоинта
    ax = axes[0, 0]
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'Grip']
    
    bp = ax.boxplot([all_actions[:, i] for i in range(8)], labels=joint_names)
    ax.set_ylabel('Action Value')
    ax.set_title('Action Distribution per Joint')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # 2. Гистограмма gripper
    ax = axes[0, 1]
    ax.hist(all_actions[:, 7], bins=50, edgecolor='black')
    ax.set_xlabel('Gripper Action')
    ax.set_ylabel('Count')
    ax.set_title('Gripper Action Distribution\n(< 0 = open, > 0 = close)')
    ax.axvline(x=0, color='r', linestyle='--')
    
    # 3. Траектория XY (вид сверху)
    ax = axes[1, 0]
    ax.scatter(all_positions[:, 0], all_positions[:, 1], c=range(len(all_positions)),
              cmap='viridis', alpha=0.5, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('End-Effector XY Trajectory (color = time)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Добавляем рабочее пространство
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.45, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Траектория XZ (вид сбоку)
    ax = axes[1, 1]
    ax.scatter(all_positions[:, 0], all_positions[:, 2], c=range(len(all_positions)),
              cmap='viridis', alpha=0.5, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('End-Effector XZ Trajectory (color = time)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = model_path.replace('.zip', '_action_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved action analysis to: {output_path}")
    
    plt.show()
    
    # Статистика
    print("\nAction Statistics:")
    print("-" * 50)
    for i, name in enumerate(joint_names):
        mean = np.mean(all_actions[:, i])
        std = np.std(all_actions[:, i])
        min_val = np.min(all_actions[:, i])
        max_val = np.max(all_actions[:, i])
        print(f"{name:6s}: mean={mean:7.3f}, std={std:6.3f}, "
              f"range=[{min_val:6.3f}, {max_val:6.3f}]")
    
    # Проверка на "застревание"
    print("\n⚠️ Potential Issues:")
    for i, name in enumerate(joint_names):
        std = np.std(all_actions[:, i])
        if std < 0.05:
            print(f"  - {name}: Very low variance ({std:.4f}) - model may be stuck!")
    
    return all_actions, all_positions


def compare_models(model_paths, n_episodes=5):
    """Сравнивает несколько моделей"""
    from stable_baselines3 import PPO
    from robot_env import RobotEnv
    
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.zip', '')
        print(f"\nTesting: {model_name}")
        
        env = RobotEnv(use_gui=False, image_size=64, camera_mode='side+depth')
        model = PPO.load(model_path, env=env)
        
        rewards = []
        lengths = []
        min_dists = []
        grasps = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            min_dist = float('inf')
            grasped = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
                total_reward += reward
                steps += 1
                
                if 'distance_to_object' in info:
                    min_dist = min(min_dist, info['distance_to_object'])
                if info.get('object_grasped', False):
                    grasped = True
            
            rewards.append(total_reward)
            lengths.append(steps)
            min_dists.append(min_dist)
            grasps.append(grasped)
        
        env.close()
        
        results[model_name] = {
            'mean_reward': np.mean(rewards),
            'mean_length': np.mean(lengths),
            'mean_min_dist': np.mean(min_dists),
            'grasp_rate': np.mean(grasps) * 100
        }
    
    # Выводим таблицу
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<40} {'Reward':>10} {'Length':>8} {'MinDist':>10} {'Grasp%':>8}")
    print("-" * 70)
    
    for name, stats in results.items():
        print(f"{name:<40} {stats['mean_reward']:>10.2f} {stats['mean_length']:>8.1f} "
              f"{stats['mean_min_dist']:>10.4f} {stats['grasp_rate']:>7.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument('--log_dir', type=str, 
                       default='RL3/logs/pickplace_mobilenet_side+depth_ppo_500k',
                       help='Directory with training logs')
    parser.add_argument('--tensorboard_dir', type=str,
                       default='RL3/tensorboard/pickplace_mobilenet_side+depth_ppo_500k',
                       help='TensorBoard log directory')
    parser.add_argument('--model', type=str, default=None,
                       help='Model to analyze actions')
    parser.add_argument('--compare', nargs='+', default=None,
                       help='List of models to compare')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t show plots interactively')
    
    args = parser.parse_args()
    
    # Анализ логов обучения
    if args.log_dir and os.path.exists(args.log_dir):
        print("\n" + "=" * 60)
        print("TRAINING PROGRESS ANALYSIS")
        print("=" * 60)
        
        # Ищем monitor.csv в log_dir или tensorboard_dir
        monitor_data, tb_data = plot_training_progress(
            args.tensorboard_dir if os.path.exists(args.tensorboard_dir) else args.log_dir,
            show=not args.no_show
        )
        
        if monitor_data is not None:
            print("\nTraining Statistics:")
            print("-" * 40)
            print(f"Total episodes: {len(monitor_data)}")
            print(f"Total steps: {monitor_data['l'].sum():,}")
            print(f"Mean reward: {monitor_data['r'].mean():.2f}")
            print(f"Max reward: {monitor_data['r'].max():.2f}")
            print(f"Min reward: {monitor_data['r'].min():.2f}")
            print(f"Mean episode length: {monitor_data['l'].mean():.1f}")
    
    # Анализ действий модели
    if args.model:
        print("\n" + "=" * 60)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 60)
        analyze_action_distribution(args.model)
    
    # Сравнение моделей
    if args.compare:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        compare_models(args.compare)


if __name__ == '__main__':
    main()
