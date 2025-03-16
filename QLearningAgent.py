import numpy as np
import random
import time  # For adding a delay between steps

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
    
    def render(self):
        """ Visualize the grid world """
        grid = [['[ ]' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark the goal
        goal_x, goal_y = self.goal_state
        grid[goal_x][goal_y] = '[G]'
        
        # Mark the agent's position
        agent_x, agent_y = self.state
        grid[agent_x][agent_y] = '[A]'
        
        # Print the grid
        print("+" + "---+" * self.size)
        for row in grid:
            print("|", end=" ")
            for cell in row:
                print(cell, end=" ")
            print("|")
            print("+" + "---+" * self.size)

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
    
    def print_q_values(self, state):
        """ Print Q-values for the current state """
        print(f"Q-values at state {state}:")
        for action in self.actions:
            q_value = self.get_q_value(state, action)
            print(f"  {action}: {q_value:.2f}")

# Create environment and agent
env = GridWorld()
agent = QLearningAgent(actions=["up", "down", "left", "right"])

# Train the agent
num_episodes = 1000
verbose = False  # Set to True if you want to see training visualization

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    if verbose and (episode + 1) % 100 == 0:
        print(f"\nEpisode {episode + 1} visualization:")
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"\nAction: {action}, Reward: {reward}")
            env.render()
            agent.print_q_values(next_state)
            time.sleep(0.5)  # Add delay to see the visualization
        
        state = next_state  # Update the current state
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed")

# Run a test episode (with no exploration)
original_epsilon = agent.epsilon
agent.epsilon = 0  # Pure exploitation
state = env.reset()
done = False
total_reward = 0

print("\nStarting test episode visualization:")
env.render()  # Show initial state

step_counter = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    state = next_state  # Update the current state
    total_reward += reward
    step_counter += 1
    
    # Clear screen (doesn't work in all environments)
    # print("\033c", end="")
    
    print(f"\nStep {step_counter}:")
    print(f"Action: {action}, New State: {next_state}, Reward: {reward}")
    env.render()  # Visualize the current state
    agent.print_q_values(state)  # Show Q-values
    time.sleep(0.8)  # Wait to see the visualization

print(f"\nEpisode finished with total reward: {total_reward}")

# Print policy
print("\nLearned policy:")
for x in range(env.size):
    for y in range(env.size):
        state = (x, y)
        q_values = [agent.get_q_value(state, a) for a in agent.actions]
        best_action = agent.actions[np.argmax(q_values)]
        print(f"At state {state}, best action: {best_action}")

# Restore original epsilon
agent.epsilon = original_epsilon