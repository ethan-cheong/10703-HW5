import os
import gym
import random
import argparse
import numpy as np
import tensorflow as tf
from networks import CartPoleNetwork
from self_play import self_play
from replay import ReplayBuffer
from config import get_cartpole_config

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


SEED = 0


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_output_directories():
    """Create directories for storing plots and other outputs"""
    dirs = [
        'outputs',
        'outputs/plots',
        'outputs/plots/training',
        'outputs/plots/testing'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs[0]  # return base output directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MuZero.")
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save outputs")
    args = parser.parse_args()
    print("num simulations:", args.num_simulations)
    
    # Setup output directories
    output_dir = setup_output_directories()
    print(f"Outputs will be saved to: {output_dir}")

    # Set seeds for reproducibility
    set_seeds()
    # Create the cartpole network
    network = CartPoleNetwork(action_size=2, state_shape=(None, 4), embedding_size=4, max_value=200)
    print("Network Made")
    # Set the configuration for muzero
    config = get_cartpole_config(args.num_simulations)  # Create Environment
    env = gym.make("CartPole-v0")

    # Create buffer to store games
    replay_buffer = ReplayBuffer(config)
    self_play(env, config, replay_buffer, network)
