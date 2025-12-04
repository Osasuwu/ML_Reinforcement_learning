"""Быстрый тест среды и обучения."""
import numpy as np
import sys
sys.path.insert(0, '.')

from robot_env import RobotEnv
from feature_extractor import MobileNetExtractor, SimpleCNNExtractor


def test_env():
    print("=" * 50)
    print("ТЕСТ СРЕДЫ")
    print("=" * 50)
    
    env = RobotEnv(camera_mode="side")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Тест: движение к объекту
    print("\nТест движения к объекту...")
    for step in range(100):
        # Простая эвристика: двигаться вперёд и вниз, закрыть схват
        action = np.array([0.3, 0.2, -0.3, 0, 0, 0, 0, 1.0])
        obs, reward, term, trunc, info = env.step(action)
        
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward:.2f}, grasped={info.get('object_grasped', False)}, dist={info.get('dist_ee_to_obj', 0):.3f}")
        
        if term:
            print(f"  Terminated: {info.get('reason', 'unknown')}")
            break
    
    env.close()
    print("[OK] Тест среды пройден")


def test_training():
    print("\n" + "=" * 50)
    print("ТЕСТ ОБУЧЕНИЯ (100 шагов)")
    print("=" * 50)
    
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    
    env = RobotEnv(camera_mode="side")
    env = Monitor(env)
    
    policy_kwargs = dict(
        features_extractor_class=SimpleCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=64,
        batch_size=32,
    )
    
    print("Обучение...")
    model.learn(total_timesteps=100)
    print("[OK] Тест обучения пройден")
    
    env.close()


if __name__ == "__main__":
    test_env()
    test_training()
    print("\n" + "=" * 50)
    print("[OK] ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 50)
    print("\nЗапуск обучения:")
    print("  python train.py --network mobilenet --steps 1000000")
