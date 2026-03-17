import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from utils.seed import set_seed
from src.Policy_Iteration.DeterministicEnv.model import Policy_Iteration_Det

set_seed(42)

def render_board(env):
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()
 
def run(env, policy):
    initial_state, _ = env.reset()
    render_board(env)

    for _ in range(10):
        action = int(policy[initial_state].item())
        observation, reward, flag, _, _ = env.step(action)
        render_board(env)
 
        if flag:
            print(f"Episode finished with reward: {reward}")
            break
 
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    output = Policy_Iteration_Det(train_env, device)
    policy, values = output.policy_iteration()
    print(f"Optimal Policy: {policy}")
    print(f"State Values:   {values}")
    train_env.close()

    render_env = gym.make('FrozenLake-v1', map_name="4x4", render_mode="rgb_array", is_slippery=False)
    run(render_env, policy)
    render_env.close()
 