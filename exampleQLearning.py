import numpy as np
import random

class GridWorldEnv:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.agent_position = (0, 0)
        self.goal_position = (width-1, height-1)
        self.actions = ["up", "down", "left", "right"]
        
    def reset(self):
        self.agent_position = (0, 0)
        return self.agent_position
        
    def step(self, action):
        x, y = self.agent_position
        
        if action == "up" and y < self.height - 1:
            y += 1
        elif action == "down" and y > 0:
            y -= 1
        elif action == "right" and x < self.width - 1:
            x += 1
        elif action == "left" and x > 0:
            x -= 1
            
        self.agent_position = (x, y)
        
        # Check if goal is reached
        done = self.agent_position == self.goal_position
        
        # Simple reward: -1 for each step, +10 for reaching goal
        reward = 10 if done else -1
        
        return self.agent_position, reward, done

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # State-action values

    def get_q_value(self, state, action):
        """ Get Q-value for a state-action pair """
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        """ Choose action using epsilon-greedy strategy """
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)  # Exploration
        else:
            # Exploitation: Choose the best known action
            q_values = [self.get_q_value(state, a) for a in self.actions]
            return self.actions[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        """ Update Q-value using Bellman equation """
        best_next_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * best_next_q)
        self.q_table[(state, action)] = new_q
 
# Random agent
class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, state=None):
        """ Select a random action """
        return random.choice(self.actions)
       
# Create environment and agent
env = GridWorldEnv()
agent = RandomAgent(actions=env.actions)

# Run a single episode
state = env.reset()
done = False
total_reward = 0

print("Starting episode...")
while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    print(f"Action: {action}, New State: {next_state}, Reward: {reward}")

print(f"Episode finished with total reward: {total_reward}")