from dqn import DQNAgent
from env import ConnectFourEnv
from game import ConnectFour

env = ConnectFourEnv()
game = ConnectFour()


agent = DQNAgent(state_size=game.rows * game.cols, action_size=game.cols, seed=0)

num_episodes = 5
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions, epsilon)

        next_state, reward, done, info = env.step(action)

        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
