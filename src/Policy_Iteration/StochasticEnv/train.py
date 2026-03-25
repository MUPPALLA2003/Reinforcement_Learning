import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from utils.seed import set_seed
from src.Policy_Iteration.StochasticEnv.model import PolicyIterationStoc

plt.ion()  

def render_board(env):
    plt.clf()
    plt.imshow(env.render())
    plt.axis("off")
    plt.pause(0.5)

def run(env, policy, max_steps=20):
    state, _ = env.reset()
    render_board(env)

    for _ in range(max_steps):
        action = int(policy[state].item())
        observation, reward, terminated, truncated, _ = env.step(action)
        render_board(env)
        state = observation

        if terminated or truncated:
            status = "reached goal" if reward > 0 else "fell in hole"
            print(f"Episode finished: {status} | reward: {reward}")
            break

    plt.ioff()   

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    model = PolicyIterationStoc(train_env)
    policy, values = model.policy_iteration()
    print(f"Optimal Policy: {policy}")
    print(f"State Values:   {values}")
    train_env.close()

    render_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
    run(render_env, policy)
    render_env.close()