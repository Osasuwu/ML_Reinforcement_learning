from RL3.robot_env import RobotEnv
import numpy as np

# Тест с фиксированной позицией
print("=== Fixed position (curriculum) ===")
env = RobotEnv(camera_mode="side", use_gui=False, fixed_object_pos=(0.45, 0.2))
all_rewards = []

for ep in range(3):
    obs, _ = env.reset()
    for step in range(100):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        all_rewards.append(r)
        if term or trunc:
            break

print(f"Rewards: min={min(all_rewards):.3f}, max={max(all_rewards):.3f}")
print(f"Mean: {np.mean(all_rewards):.3f}")
positive = sum(1 for r in all_rewards if r > 0)
print(f"Positive: {positive} / {len(all_rewards)}")
env.close()

# Тест бездействия
print("\n=== Zero actions ===")
env = RobotEnv(camera_mode="side", use_gui=False, fixed_object_pos=(0.45, 0.2))
obs, _ = env.reset()
for i in range(5):
    obs, r, _, _, _ = env.step(np.zeros(8))
    print(f"Step {i+1}: reward = {r:.3f}")
env.close()
