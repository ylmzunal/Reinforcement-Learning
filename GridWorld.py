import numpy as np
import random

# Gridworld environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size  # Grid size (5x5)
        self.start_state = (0, 0)  # Agent starts at top-left corner
        self.goal_state = (4, 4)  # Goal is at bottom-right corner
        self.state = self.start_state

    def reset(self):
        """ Reset environment and return initial state """
        self.state = self.start_state
        return self.state

    def step(self, action):
        """ Take an action and return new state, reward, and done flag """
        x, y = self.state
        if action == "up":
            x = max(0, x - 1)
        elif action == "down":
            x = min(self.size - 1, x + 1)
        elif action == "left":
            y = max(0, y - 1)
        elif action == "right":
            y = min(self.size - 1, y + 1)

        self.state = (x, y)

        # Define rewards
        if self.state == self.goal_state:
            return self.state, 10, True  # Goal reached, reward +10
        else:
            return self.state, -1, False  # Step cost -1

# Random agent
class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def select_action(self):
        """ Select a random action """
        return random.choice(self.actions)

# Create environment and agent
env = GridWorld()
agent = RandomAgent(actions=["up", "down", "left", "right"])

# Run a single episode
state = env.reset()
done = False
total_reward = 0

print("Starting episode...")
while not done:
    action = agent.select_action()
    next_state, reward, done = env.step(action)
    total_reward += reward
    print(f"Action: {action}, New State: {next_state}, Reward: {reward}")

print(f"Episode finished with total reward: {total_reward}")