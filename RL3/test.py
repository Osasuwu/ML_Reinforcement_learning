"""
Тестирование обученной модели pick-and-place.
"""
import os
import sys
import json
import argparse
import numpy as np
import time
import cv2
from stable_baselines3 import PPO, SAC

# Добавляем путь к RL3 для импортов
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from robot_env import RobotEnv
from feature_extractor import (MobileNetExtractor, EfficientNetExtractor, 
                               SimpleCNNExtractor, MobileNetWithDepthExtractor)


# Устанавливаем заголовок окна консоли (Windows)
if sys.platform == 'win32':
    os.system('title Pick-and-Place Robot Test')


def visualize_observation(obs, window_name="Neural Network Input", scale=2, camera_mode="side"):
    """
    Показывает что видит нейросеть.
    Для side+depth/side+wrist показывает апскейленную вторичную камеру,
    хотя нейросеть получает реальный 8x8.
    """
    if obs is None:
        return
    
    # Обработка Dict observation (side+depth или side+wrist)
    if isinstance(obs, dict):
        image = np.array(obs['image'])
        
        # Определяем какой ключ для вторичной камеры
        secondary_key = 'depth' if 'depth' in obs else 'wrist'
        secondary = np.array(obs[secondary_key])
        
        # Убираем batch dimension если есть
        while len(image.shape) > 4:
            image = image[0]
            secondary = secondary[0]
        
        # image: (frame_stack, H, W, 1), secondary: (frame_stack, 32, 32, 1)
        frame_stack, h, w, _ = image.shape
        sec_size = secondary.shape[1]  # Размер secondary камеры (32x32)
        
        # Последний кадр
        img_frame = image[-1, :, :, 0].astype(np.uint8)
        sec_frame = secondary[-1, :, :, 0].astype(np.uint8)
        
        # Масштабируем для визуализации
        img_big = cv2.resize(img_frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        # Апскейлим secondary 32x32 -> 64*scale
        sec_big = cv2.resize(sec_frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        combined = np.hstack([img_big, sec_big])
        cv2.putText(combined, "Front 64x64", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        label = "Depth" if secondary_key == 'depth' else "Wrist"
        cv2.putText(combined, f"{label} {sec_size}x{sec_size}", (w * scale + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        cv2.imshow(window_name, combined)
        return cv2.waitKey(1)
    
    # Box observation (side only)
    obs = np.array(obs)
    
    # Обработка разных форматов observation
    # VecEnv возвращает (1, frame_stack, H, W, n_cameras)
    # Обычный env возвращает (frame_stack, H, W, n_cameras)
    while len(obs.shape) > 4:
        obs = obs[0]
    
    # Теперь obs shape: (frame_stack, H, W, n_cameras)
    if len(obs.shape) != 4:
        return
    
    frame_stack, h, w, n_cameras = obs.shape
    
    # Берём последний кадр (самый свежий)
    frames = []
    for cam in range(n_cameras):
        frame = obs[-1, :, :, cam]  # Последний кадр, камера cam
        # Масштабируем для лучшей видимости
        frame_big = cv2.resize(frame.astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        frames.append(frame_big)
    
    # Подписи в зависимости от режима
    if n_cameras == 2:
        combined = np.hstack(frames)
        cv2.putText(combined, "Front", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(combined, "Wrist", (w * scale + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    else:
        combined = frames[0]
        cv2.putText(combined, "Front Camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    cv2.imshow(window_name, combined)
    key = cv2.waitKey(1)
    return key


def load_config(model_path):
    """Загружает config для модели. Ищет несколько вариантов."""
    import re
    
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # Вариант 1: точное совпадение (model_config.json)
    config_path = model_path.replace('.zip', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Вариант 2: общий конфиг эксперимента {exp_name}_config.json
    # pickplace_mobilenet_side+depth_ppo_1000k_10k.zip -> pickplace_mobilenet_side+depth_ppo_1000k_config.json
    base_path = re.sub(r'_\d+k\.zip$', '_config.json', model_path)
    if os.path.exists(base_path):
        with open(base_path, 'r') as f:
            return json.load(f)
    
    # Вариант 3: _final_config или _best_config
    for suffix in ['_final_config.json', '_best_config.json']:
        base_path = re.sub(r'_\d+k\.zip$', suffix, model_path)
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                return json.load(f)
    
    # Вариант 4: ищем любой config в той же папке с похожим именем
    match = re.match(r'(.+?)_\d+k', model_name)
    if match:
        exp_base = match.group(1)
        for fname in os.listdir(model_dir):
            if fname.startswith(exp_base) and fname.endswith('_config.json'):
                with open(os.path.join(model_dir, fname), 'r') as f:
                    return json.load(f)
    
    return None


def test_model(model_path, n_episodes=10, use_gui=True, slow=True):
    print("=" * 60)
    print("TESTING PICK-AND-PLACE MODEL")
    print("=" * 60)
    
    try:
        _run_test(model_path, n_episodes, use_gui, slow)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if use_gui:
            print("\n[Press Enter to close]")
            try:
                input()
            except:
                pass


def _run_test(model_path, n_episodes=10, use_gui=True, slow=True):
    """Внутренняя функция теста"""
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
        
        camera_mode = config.get('camera_mode', 'side')
        env_kwargs = {
            'image_size': config.get('image_size', 64),
            'frame_stack': 4,
            'camera_mode': camera_mode,
            'max_steps': 200
        }
        algo = config.get('algorithm', 'ppo')
        network = config.get('network', 'mobilenet')
    else:
        print("[!] Config not found, using defaults")
        camera_mode = 'side'
        env_kwargs = {
            'image_size': 64,
            'frame_stack': 4,
            'camera_mode': camera_mode,
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
        custom_objs = {
            "MobileNetExtractor": MobileNetExtractor,
            "EfficientNetExtractor": EfficientNetExtractor,
            "SimpleCNNExtractor": SimpleCNNExtractor,
            "MobileNetWithDepthExtractor": MobileNetWithDepthExtractor
        }
        if algo == 'ppo':
            model = PPO.load(model_path, env=env, custom_objects=custom_objs)
        else:
            model = SAC.load(model_path, env=env, custom_objects=custom_objs)
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
    
    # Определяем количество камер для размера окна
    n_cameras = env_kwargs.get('n_cameras', 1)
    if camera_mode == 'both':
        n_cameras = 2
    
    # Создаём окно для визуализации с правильными пропорциями
    if use_gui:
        window_width = 256 * n_cameras  # 256 для 1 камеры, 512 для 2
        window_height = 256
        cv2.namedWindow("Neural Network Input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Neural Network Input", window_width, window_height)
        cv2.moveWindow("Neural Network Input", 50, 50)
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        grasped = False
        
        print(f"Episode {ep + 1}/{n_episodes}")
        
        # Показываем начальное наблюдение
        if use_gui:
            visualize_observation(obs, camera_mode=camera_mode)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_length += 1
            
            if info.get('object_grasped', False):
                grasped = True
            
            # Визуализируем что видит нейросеть
            if use_gui:
                visualize_observation(obs, camera_mode=camera_mode)
            
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
    
    # Закрываем окна
    if use_gui:
        cv2.destroyAllWindows()
    
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
