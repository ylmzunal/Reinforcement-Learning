import gym

# Create the CartPole environment
env = gym.make("CartPole-v1")

state = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action (0: Left, 1: Right)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # Visualize the environment

env.close()