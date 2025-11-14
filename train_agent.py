import numpy as np
import matplotlib.pyplot as plt
from simple_rl_env import SimpleEnv
from collections import defaultdict


class QLearningAgent:
    """
    Q-learning агент с дискретизацией пространства состояний.
    """
    
    def __init__(self, n_actions=2, learning_rate=0.3, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-таблица как словарь для разреженного хранения
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def discretize_state(self, state):
        """
        Дискретизация непрерывного состояния.
        state: [x_agent, x_target]
        """
        x_agent, x_target = state
        
        # Дискретизация с шагом 0.5 для лучшей точности
        x_agent_disc = round(x_agent / 0.5) * 0.5
        x_target_disc = round(x_target / 0.5) * 0.5
        
        return (x_agent_disc, x_target_disc)
    
    def get_action(self, state):
        """
        Выбор действия с epsilon-greedy стратегией.
        """
        if np.random.random() < self.epsilon:
            # Exploration: случайное действие
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: лучшее действие по Q-таблице
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Обновление Q-таблицы по формуле:
        Q[s,a] = Q[s,a] + α [r + γ max_a' Q[s',a'] - Q[s,a]]
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Текущее Q-значение
        current_q = self.q_table[discrete_state][action]
        
        # TD-target
        if done:
            td_target = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            td_target = reward + self.gamma * max_next_q
        
        # TD-error и обновление
        td_error = td_target - current_q
        self.q_table[discrete_state][action] += self.alpha * td_error
    
    def decay_epsilon(self):
        """
        Уменьшение epsilon для снижения exploration.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(n_episodes=1000, render_interval=100, gui=False):
    """
    Обучение агента в среде.
    """
    env = SimpleEnv(gui=gui)
    agent = QLearningAgent()
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("Начало обучения...")
    print(f"Параметры: episodes={n_episodes}, α={agent.alpha}, γ={agent.gamma}")
    print("-" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Выбор и выполнение действия
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Обновление Q-таблицы
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Рендеринг (если включен GUI и эпизод для визуализации)
            if gui and episode % render_interval == 0:
                env.render()
        
        # Уменьшение epsilon
        agent.decay_epsilon()
        
        # Сохранение статистики
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        # Проверка успеха (достижение цели)
        if step_count < env.max_steps:
            success_count += 1
        
        # Логирование
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Success: {success_rate:.1f}%")
    
    print("-" * 60)
    print(f"Обучение завершено!")
    print(f"Финальная успешность: {success_count / n_episodes * 100:.1f}%")
    print(f"Размер Q-таблицы: {len(agent.q_table)} состояний")
    
    env.close()
    
    return agent, episode_rewards, episode_lengths


def plot_training_results(episode_rewards, episode_lengths):
    """
    Визуализация результатов обучения.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # График наград
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Скользящее среднее
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), 
                moving_avg, 
                color='red', 
                linewidth=2, 
                label=f'Moving Average ({window})')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Training Progress: Reward per Episode', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График длины эпизодов
    ax2.plot(episode_lengths, alpha=0.3, color='green', label='Episode Length')
    
    if len(episode_lengths) >= window:
        moving_avg_len = np.convolve(episode_lengths, 
                                     np.ones(window)/window, 
                                     mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), 
                moving_avg_len, 
                color='orange', 
                linewidth=2, 
                label=f'Moving Average ({window})')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Training Progress: Episode Length', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("График сохранён в training_results.png")
    plt.show()


def test_agent(agent, n_episodes=5, gui=True):
    """
    Тестирование обученного агента с визуализацией.
    """
    env = SimpleEnv(gui=gui)
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОГО АГЕНТА")
    print("=" * 60)
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        print(f"\nТест эпизод {episode + 1}:")
        print(f"Начальная позиция агента: {state[0]:.2f}")
        print(f"Позиция цели: {state[1]:.2f}")
        print(f"Начальное расстояние: {abs(state[0] - state[1]):.2f}")
        
        while not done:
            # Используем только exploitation (epsilon=0)
            discrete_state = agent.discretize_state(state)
            action = np.argmax(agent.q_table[discrete_state])
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            if gui:
                env.render(sleep_time=0.05)
            
            state = next_state
        
        final_distance = abs(state[0] - state[1])
        print(f"Итог: шагов={step_count}, награда={total_reward:.2f}, "
              f"финальное расстояние={final_distance:.2f}")
        
        if final_distance < env.goal_threshold:
            print("✓ ЦЕЛЬ ДОСТИГНУТА!")
        else:
            print("✗ Цель не достигнута")
    
    env.close()


if __name__ == "__main__":
    # Обучение агента
    agent, rewards, lengths = train_agent(
        n_episodes=1000,
        render_interval=100,
        gui=False  # Установите True для визуализации процесса обучения
    )
    
    # Визуализация результатов
    plot_training_results(rewards, lengths)
    
    # Тестирование агента
    print("\nЗапуск тестирования с визуализацией...")
    print("(Закройте окно PyBullet после просмотра для завершения)")
    test_agent(agent, n_episodes=3, gui=True)
