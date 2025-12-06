import sys
sys.path.insert(0, 'RL3')
from robot_env import RobotEnv
import numpy as np

env = RobotEnv(use_gui=False, camera_mode='side+depth', image_size=64, frame_stack=4)

print('Testing new reward:')
obs, _ = env.reset()

total = 0
for i in range(50):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    total += r
    if i % 10 == 9:
        dist_xy = info.get('dist_xy', 0)
        dist_z = info.get('dist_z', 0)
        print(f'  Step {i+1}: r={r:.3f}, dist_xy={dist_xy:.3f}, dist_z={dist_z:.3f}')

print(f'Total reward: {total:.3f}')
env.close()
