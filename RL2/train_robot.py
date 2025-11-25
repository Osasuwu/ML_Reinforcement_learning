import numpy as np
import matplotlib.pyplot as plt
from robot_rl_env import RobotEnv
from collections import defaultdict
import pickle
import os


class QLearningAgent:
    """
    Q-learning агент для управления двухколёсным роботом.
    """
    
    def __init__(self, n_actions=27, learning_rate=0.25, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9965, epsilon_min=0.01):
        self.n_actions = n_actions  # 27 действий (3 скорости × 3x3 комбинации колёс)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-таблица как словарь
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def discretize_state(self, state):
        """
        Дискретизация непрерывного состояния.
        state: [x_robot, y_robot, angle_to_target, distance]
        
        Улучшенная версия для индивидуального управления колёсами:
        - 16 направлений (каждые 22.5°) - более точное различение углов
        - 5 уровней расстояния
        - Итого: 16 × 5 = 80 состояний
        """
        x, y, angle_to_target, distance = state
        
        # Дискретизация угла к цели (16 направлений - каждые 22.5 градусов)
        angle_bin = int(np.floor((angle_to_target + np.pi) / (2 * np.pi / 16)))
        angle_bin = np.clip(angle_bin, 0, 15)
        
        # Дискретизация расстояния (5 уровней для лучшей точности)
        if distance < 0.7:
            dist_bin = 0  # Очень близко
        elif distance < 1.5:
            dist_bin = 1  # Близко
        elif distance < 2.5:
            dist_bin = 2  # Средне
        elif distance < 3.5:
            dist_bin = 3  # Далеко
        else:
            dist_bin = 4  # Очень далеко
        
        # Состояние: 16 углов x 5 расстояний = 80 состояний
        return (angle_bin, dist_bin)
    
    def get_action(self, state):
        """
        Выбор действия с epsilon-greedy стратегией.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Обновление Q-таблицы.
        Q[s,a] = Q[s,a] + α [r + γ max_a' Q[s',a'] - Q[s,a]]
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            td_target = reward + self.gamma * max_next_q
        
        td_error = td_target - current_q
        self.q_table[discrete_state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """
        Уменьшение epsilon.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """
        Сохранение Q-таблицы.
        """
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-таблица сохранена в {filename}")
    
    def load(self, filename):
        """
        Загрузка Q-таблицы.
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.n_actions), loaded_table)
            print(f"Q-таблица загружена из {filename}")
        else:
            print(f"Файл {filename} не найден")


def train_agent(n_episodes=1500, render_interval=500, gui=False, save_path="q_table.pkl"):
    """
    Обучение агента.
    """
    env = RobotEnv(gui=gui)
    agent = QLearningAgent()
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    distances_at_end = []
    
    print("=" * 70)
    print("ОБУЧЕНИЕ РОБОТА-НАВИГАТОРА")
    print("=" * 70)
    print(f"Параметры: episodes={n_episodes}, α={agent.alpha}, γ={agent.gamma}")
    print(f"Действия (27): Индивидуальное управление колёсами + выбор скорости")
    print(f"  Скорость: Медленная(0.4x), Средняя(0.7x), Быстрая(1.0x)")
    print(f"  Левое/Правое: Назад, Стоп, Вперёд")
    print(f"  action = speed*3 + left*3 + right*3")
    print("-" * 70)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if gui and episode % render_interval == 0:
                env.render()
        
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Финальное расстояние
        final_distance = env._get_distance()
        distances_at_end.append(final_distance)
        
        if final_distance < env.goal_threshold:
            success_count += 1
        
        # Логирование
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_distance = np.mean(distances_at_end[-50:])
            success_rate = success_count / (episode + 1) * 100
            
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Steps: {avg_length:6.1f} | "
                  f"Dist: {avg_distance:5.2f}m | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Success: {success_rate:.1f}%")
    
    print("-" * 70)
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Итоговая успешность: {success_count / n_episodes * 100:.1f}%")
    print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
    print(f"Средняя награда (последние 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Средняя длина эпизода (последние 100): {np.mean(episode_lengths[-100:]):.1f}")
    print("-" * 70)
    
    # Сохранение Q-таблицы
    agent.save(save_path)
    
    env.close()
    
    return agent, episode_rewards, episode_lengths, distances_at_end


def plot_training_results(episode_rewards, episode_lengths, distances):
    """
    Визуализация результатов обучения.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    window = 50
    
    # 1. График наград
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                color='red', linewidth=2, label=f'MA({window})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Progress: Reward per Episode', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. График длины эпизодов
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, alpha=0.3, color='green', label='Episode Length')
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg,
                color='orange', linewidth=2, label=f'MA({window})')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Training Progress: Episode Length', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. График финального расстояния
    ax3 = axes[1, 0]
    ax3.plot(distances, alpha=0.3, color='purple', label='Final Distance')
    if len(distances) >= window:
        moving_avg = np.convolve(distances, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(distances)), moving_avg,
                color='red', linewidth=2, label=f'MA({window})')
    ax3.axhline(y=0.3, color='r', linestyle='--', label='Goal Threshold (0.3m)')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Distance to Goal (m)', fontsize=12)
    ax3.set_title('Final Distance to Goal', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Гистограмма длины эпизодов (последние 200)
    ax4 = axes[1, 1]
    recent_lengths = episode_lengths[-200:] if len(episode_lengths) >= 200 else episode_lengths
    ax4.hist(recent_lengths, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax4.axvline(x=100, color='r', linestyle='--', label='Target: <100 steps')
    ax4.set_xlabel('Episode Length (steps)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Episode Lengths (Last 200)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_results_robot.png', dpi=300, bbox_inches='tight')
    print("График сохранён в training_results_robot.png")
    plt.show()


def test_agent(agent, n_episodes=5, gui=True):
    """
    Тестирование обученного агента.
    """
    env = RobotEnv(gui=gui)
    
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОГО АГЕНТА")
    print("=" * 70)
    
    def get_action_name(action):
        """Получить название действия по его номеру"""
        speed_level = action // 9
        wheel_combo = action % 9
        left = wheel_combo // 3
        right = wheel_combo % 3
        
        speed_names = ['Медленно', 'Средне', 'Быстро']
        wheel_names = ['Назад', 'Стоп', 'Вперёд']
        
        return f"{speed_names[speed_level]}: L:{wheel_names[left]}, R:{wheel_names[right]}"
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        action_counts = [0] * 27  # 27 действий вместо 9
        
        initial_distance = env._get_distance()
        
        print(f"\nТест эпизод {episode + 1}:")
        print(f"Позиция робота: ({state[0]:.2f}, {state[1]:.2f})")
        print(f"Позиция цели: ({env.target_pos[0]:.2f}, {env.target_pos[1]:.2f})")
        print(f"Начальное расстояние: {initial_distance:.2f}м")
        
        while not done:
            discrete_state = agent.discretize_state(state)
            action = np.argmax(agent.q_table[discrete_state])
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
            action_counts[action] += 1
            
            if gui:
                env.render(sleep_time=0.03)
            
            state = next_state
        
        final_distance = env._get_distance()
        
        print(f"Результат:")
        print(f"  Шагов: {step_count}")
        print(f"  Награда: {total_reward:.2f}")
        print(f"  Финальное расстояние: {final_distance:.2f}м")
        print(f"  Использование действий (топ-5):")
        
        # Показываем только топ-5 наиболее используемых действий
        action_usage = [(i, count) for i, count in enumerate(action_counts) if count > 0]
        action_usage.sort(key=lambda x: x[1], reverse=True)
        
        for i, (action, count) in enumerate(action_usage[:5]):
            print(f"    {get_action_name(action)}: {count} раз ({count/step_count*100:.1f}%)")
        
        if final_distance < env.goal_threshold:
            print("  ✓ ЦЕЛЬ ДОСТИГНУТА!")
        else:
            print("  ✗ Цель не достигнута")
    
    env.close()


if __name__ == "__main__":
    # Обучение агента
    agent, rewards, lengths, distances = train_agent(
        n_episodes=1500,
        render_interval=500,
        gui=False,  # Установите True для визуализации процесса обучения
        save_path=r"q_table_robot.pkl"
    )
    
    # Визуализация результатов
    plot_training_results(rewards, lengths, distances)
    
    # Тестирование агента
    print("\nЗапуск тестирования с визуализацией...")
    print("(Закройте окно PyBullet после просмотра для продолжения)")
    test_agent(agent, n_episodes=5, gui=True)
    
    print("\n" + "=" * 70)
    print("ПРОГРАММА ЗАВЕРШЕНА")
    print("=" * 70)
