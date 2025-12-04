"""
Тестирование обученной модели pick-and-place.
"""
import os
import sys
import json
import argparse
import numpy as np
import time
from stable_baselines3 import PPO, SAC

# Добавляем путь к RL3 для импортов
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from robot_env import RobotEnv
from feature_extractor import MobileNetExtractor, EfficientNetExtractor, SimpleCNNExtractor


# Устанавливаем заголовок окна консоли (Windows)
if sys.platform == 'win32':
    os.system('title Pick-and-Place Robot Test')


def load_config(model_path):
    config_path = model_path.replace('.zip', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def test_model(model_path, n_episodes=10, use_gui=True, slow=True):
    print("=" * 60)
    print("TESTING PICK-AND-PLACE MODEL")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    config = load_config(model_path)
    
    if config:
        print(f"[OK] Config loaded:")
        print(f"  - Experiment: {config.get('experiment_name', 'N/A')}")
        print(f"  - Network: {config.get('network', 'mobilenet')}")
        print(f"  - Algorithm: {config.get('algorithm', 'ppo')}")
        print(f"  - Camera: {config.get('camera_mode', 'side')}")
        
        env_kwargs = {
            'image_size': config.get('image_size', 64),
            'frame_stack': 4,
            'camera_mode': config.get('camera_mode', 'side'),
            'max_steps': 200
        }
        algo = config.get('algorithm', 'ppo')
        network = config.get('network', 'mobilenet')
    else:
        print("[!] Config not found, using defaults")
        env_kwargs = {
            'image_size': 64,
            'frame_stack': 4,
            'camera_mode': 'side',
            'max_steps': 200
        }
        algo = 'ppo'
        network = 'mobilenet'
    
    # Create environment
    env = RobotEnv(use_gui=use_gui, **env_kwargs)
    
    # Select extractor
    if network == "mobilenet":
        extractor_class = MobileNetExtractor
    elif network == "efficientnet":
        extractor_class = EfficientNetExtractor
    else:
        extractor_class = SimpleCNNExtractor
    
    # Load model
    try:
        if algo == 'ppo':
            model = PPO.load(model_path, env=env, custom_objects={
                "MobileNetExtractor": MobileNetExtractor,
                "EfficientNetExtractor": EfficientNetExtractor,
                "SimpleCNNExtractor": SimpleCNNExtractor
            })
        else:
            model = SAC.load(model_path, env=env, custom_objects={
                "MobileNetExtractor": MobileNetExtractor,
                "EfficientNetExtractor": EfficientNetExtractor,
                "SimpleCNNExtractor": SimpleCNNExtractor
            })
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        env.close()
        return
    
    print(f"\nRunning {n_episodes} episodes...\n")
    
    # Statistics
    successes = []
    rewards = []
    lengths = []
    grasp_count = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        grasped = False
        
        print(f"Episode {ep + 1}/{n_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_length += 1
            
            if info.get('object_grasped', False):
                grasped = True
            
            if ep_length % 40 == 0 or done:
                g_str = "+" if info.get('object_grasped', False) else "-"
                print(f"  Step {ep_length}: R={reward:.2f}, "
                      f"dist_obj={info.get('dist_ee_to_obj', 0):.3f}, "
                      f"dist_goal={info.get('dist_obj_to_goal', 0):.3f}, "
                      f"grasped={g_str}")
            
            if use_gui and slow:
                time.sleep(0.02)
        
        success = info.get('success', False)
        successes.append(success)
        rewards.append(ep_reward)
        lengths.append(ep_length)
        if grasped:
            grasp_count += 1
        
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {status} | R={ep_reward:.2f}, L={ep_length}, Grasped={grasped}\n")
    
    env.close()
    
    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    success_rate = np.mean(successes) * 100
    grasp_rate = grasp_count / n_episodes * 100
    
    print(f"\nSuccess rate: {sum(successes)}/{n_episodes} ({success_rate:.1f}%)")
    print(f"Grasp rate: {grasp_count}/{n_episodes} ({grasp_rate:.1f}%)")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Best reward: {max(rewards):.2f}")


def list_models():
    print("\n📁 Available models:")
    for dir_name in ["RL3/models_v2", "RL3/models"]:
        if os.path.exists(dir_name):
            models = [f for f in os.listdir(dir_name) if f.endswith('.zip')]
            if models:
                print(f"\n  {dir_name}/")
                for m in sorted(models):
                    print(f"    - {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', default=None, help='Path to model')
    parser.add_argument('-n', '--episodes', type=int, default=10)
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI')
    parser.add_argument('--gui', action='store_true', help='Enable GUI (default)')
    parser.add_argument('--fast', action='store_true', help='Fast mode (no delay)')
    parser.add_argument('--list', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.model:
        use_gui = not args.no_gui  # GUI по умолчанию, --no-gui отключает
        test_model(args.model, args.episodes, use_gui, not args.fast)
    else:
        list_models()
        print("\nUsage: python test.py <model_path> [-n N] [--no-gui] [--fast]")
