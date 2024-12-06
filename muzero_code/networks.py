import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import L2
import numpy as np
import math
from typing import Callable
from networks_base import BaseNetwork


def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)


def action_to_one_hot(action, action_space_size):
    """
    Compute one hot of action to be combined with state representation
    """
    return np.array([1 if i == action else 0 for i in range(action_space_size)]).reshape(1, -1)


class CartPoleNetwork(BaseNetwork):
    def __init__(self, action_size, state_shape, embedding_size, max_value):
        """
        Defines the CartPoleNetwork
        action_size: the number of actions
        state_shape: the shape of the input state
        embedding_size: the size of the embedding layer for representation
        max_value: denotes the max reward of a game for value transform
        """
        self.action_size = action_size
        self.state_shape = state_shape
        self.embedding_size = embedding_size
        self.train_steps = 0
        self.hidden_neurons = 48
        # value support size should be <= math.ceil(math.sqrt(max_value)) + 1
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        # Only in negative values are allowed
        self.full_support_size = 2 * self.value_support_size + 1
        regularizer = L2(1e-4)  # Weight Decay Strength

        # Create Networks, we use tanh activation as a normalizer on the state
        # Standard muzero performs max-min normalization
        representation_network = Sequential(
            [
                Dense(
                    self.hidden_neurons,
                    activation="relu",
                    kernel_regularizer=regularizer,
                ),
                Dense(
                    self.embedding_size,
                    activation="tanh",
                    kernel_regularizer=regularizer,
                ),
            ]
        )
        # Change to full support if negative values allowed
        # map internal state into values
        value_network = Sequential(
            [
                Dense(
                    self.hidden_neurons,
                    activation="relu",
                    kernel_regularizer=regularizer,
                ),
                Dense(self.value_support_size, kernel_regularizer=regularizer),
            ]
        )

        # map internal state into policy vector. (use softmax on the output to get action distribution.)
        policy_network = Sequential(
            [
                Dense(
                    self.hidden_neurons,
                    activation="relu",
                    kernel_regularizer=regularizer,
                ),
                Dense(action_size, kernel_regularizer=regularizer),
            ]
        )

        # The following two networks were modelled jointly in the paper
        # map state and action to next state
        dynamic_network = Sequential(
            [
                Dense(
                    self.hidden_neurons,
                    activation="relu",
                    kernel_regularizer=regularizer,
                ),
                Dense(
                    self.embedding_size,
                    activation="tanh",
                    kernel_regularizer=regularizer,
                ),
            ]
        )
        # map state and action to reward
        reward_network = Sequential(
            [
                Dense(16, activation="relu", kernel_regularizer=regularizer),
                Dense(1, kernel_regularizer=regularizer),
            ]
        )

        super().__init__(
            representation_network,
            value_network,
            policy_network,
            dynamic_network,
            reward_network,
        )

    def training_steps(self):
        return self.train_steps

    def _value_transform(self, value_support) -> float:
        value = self._softmax(value_support)
        # Change to -value_support_size -> value_support_size for full support
        value = np.dot(value, range(self.value_support_size))
        value = tf.math.sign(value) * (((tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
        return value.numpy()[0]

    def _scalar_to_support(self, target_value):
        batch = len(target_value)
        targets = np.zeros((batch, self.value_support_size))
        target_value = tf.math.sign(target_value) * (tf.math.sqrt(tf.math.abs(target_value) + 1) - 1 + 0.001 * target_value)
        target_value = tf.clip_by_value(target_value, 0, self.value_support_size)
        floor = tf.math.floor(target_value)
        rest = target_value - floor
        targets[range(batch), tf.cast(floor, tf.int32)] = 1 - rest
        indexes = tf.cast(floor, tf.int32) + 1
        mask = indexes < self.value_support_size
        batch_mask = tf.boolean_mask(range(batch), mask)
        rest_mask = tf.boolean_mask(rest, mask)
        index_mask = tf.boolean_mask(indexes, mask)
        targets[batch_mask, index_mask] = rest_mask
        return targets

    def _reward_transform(self, reward) -> float:
        """
        No reward transform for cartpole
        """
        return float(reward.numpy()[0])

    def _conditioned_hidden_state(self, hidden_state: np.array, action: int) -> np.array:
        """
        concatenate the hidden state and action for input to recurrent model
        """
        conditioned_hidden = tf.concat((hidden_state, action_to_one_hot(action, self.action_size)), axis=1)
        return conditioned_hidden

    def _softmax(self, values):
        """
        Compute softmax
        """
        return tf.nn.softmax(values)

    def get_value_target(self, state):
        return self._value_transform(self.target_network.__call__(state))

    def cb_get_variables(self) -> Callable:
        """
        Return a callback that return the trainable variables of the network.
        """

        def get_variables():
            networks = (
                self.representation_network,
                self.value_network,
                self.policy_network,
                self.dynamic_network,
                self.reward_network,
            )
            return [variables for variables_list in map(lambda n: n.trainable_weights, networks) for variables in variables_list]

        return get_variables

    def save(self, path):
        """Save the networks."""
        if not os.path.isdir(path):
            os.mkdir(path)

        self.representation_network.save_weights(path + "/representation_net")  # , save_format='h5py')
        self.value_network.save_weights(path + "/value_net")  # , save_format='h5py')
        self.policy_network.save_weights(path + "/policy_net")  # , save_format='h5py')
        self.dynamic_network.save_weights(path + "/dynamic_net")  # , save_format='h5py')
        self.reward_network.save_weights(path + "/reward_net")  # , save_format='h5py')
        print("saved network at path:", path)

    def load(self, path):
        """Load previously stored network parameters."""
        self.built = True
        self.representation_network.built = True
        self.representation_network.load_weights(path + "/representation_net")
        self.value_network.load_weights(path + "/value_net")
        self.policy_network.load_weights(path + "/policy_net")
        self.dynamic_network.load_weights(path + "/dynamic_net")
        self.reward_network.load_weights(path + "/reward_net")
        print("loaded pre-trained weights at path:", path)


def scale_gradient(tensor, scale):
    """
    Function to scale gradient as described in MuZero Appendix
    """
    return tensor * scale + tf.stop_gradient(tensor) * (1.0 - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    """
    Train Network for N steps
    """
    for _ in range(config.train_per_epoch):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)


def update_weights(config, network, optimizer, batch, train_results):
    """
    TODO: Implement this function
    Train the network_model by sampling games from the replay_buffer.
    config: A dictionary specifying parameter configurations
    network: The network class to train
    optimizer: The optimizer used to update the network_model weights
    batch: The batch of experience
    train_results: The class to store the train results

    Hints:
    The network initial_model should be used to create the hidden state
    The recurrent_model should be used as the dynamics, which unroll in the latent space.

    You should accumulate loss in the value, the policy, and the reward (after the first state)
    Loss Note: The policy outputs are the logits, same with the value categorical representation
    You should use tf.nn.softmax_cross_entropy_with_logits to compute the loss in these cases
    """
    (state_batch, targets_init_batch, targets_recurrent_batch, actions_batch) = batch

    # Convert to tensors
    state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)

    # Extract initial targets: value, reward, policy
    init_values, init_rewards, init_policies = zip(*targets_init_batch)
    init_values = tf.convert_to_tensor(init_values, dtype=tf.float32)
    init_policies = tf.convert_to_tensor(init_policies, dtype=tf.float32)
    # initial rewards are always 0 at initial inference, so no need to compute reward loss initially
    # Convert scalar values to categorical
    init_values_support = network._scalar_to_support(init_values)

    # For recurrent steps, we have a list of targets for each unroll step.
    # targets_recurrent_batch: list of length num_unroll_steps
    # Each element is a tuple of lists: (value, reward, policy) for each sample in the batch

    num_unroll_steps = len(targets_recurrent_batch)

    with tf.GradientTape() as tape:
        # Initial inference
        # initial_model(state_batch) returns: hidden_representation, value, policy_logits
        hidden_rep, pred_value, pred_policy_logits = network.initial_model(state_batch)

        # Compute initial losses
        # Value loss (cross entropy between pred_value and init_values_support)
        value_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_value, labels=init_values_support)
        value_loss = tf.reduce_mean(value_loss)

        # Policy loss (cross entropy)
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_policy_logits, labels=init_policies)
        policy_loss = tf.reduce_mean(policy_loss)

        # No reward loss at initial step
        reward_loss = 0.0

        # Scale value loss by 0.25
        total_loss = 0.25 * value_loss + policy_loss

        # Now unroll for num_unroll_steps
        # hidden_rep is the representation R_t
        # For each step, we input: [R_t, one_hot(action_t)]
        # Then apply recurrent_model to get next hidden state, reward, value, policy
        current_hidden = hidden_rep

        for step_idx, (step_values) in enumerate(targets_recurrent_batch):
            # step_values = list of tuples for each sample in batch: (value, reward, policy)
            step_values, step_rewards, step_policies = zip(*step_values)

            # Convert to tensors
            step_values = tf.convert_to_tensor(step_values, dtype=tf.float32)
            step_rewards = tf.convert_to_tensor(step_rewards, dtype=tf.float32)
            step_policies = tf.convert_to_tensor(step_policies, dtype=tf.float32)

            # Convert scalar values to support
            step_values_support = network._scalar_to_support(step_values)

            # Actions for this unroll step
            step_actions = actions_batch[step_idx]
            step_actions = tf.convert_to_tensor(step_actions, dtype=tf.int32)

            # Create one-hot of actions
            action_one_hot = tf.one_hot(step_actions, depth=config.action_space_size, dtype=tf.float32)
            # Conditioned hidden state
            conditioned_hidden = tf.concat([current_hidden, action_one_hot], axis=1)

            # Recurrent inference
            # recurrent_model(conditioned_hidden) returns: (hidden_representation, reward, value, policy_logits)
            next_hidden, pred_reward, pred_value, pred_policy_logits = network.recurrent_model(conditioned_hidden)

            # Compute losses at this step
            # Value loss
            step_value_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_value, labels=step_values_support)
            step_value_loss = tf.reduce_mean(step_value_loss)

            # Reward loss (MSE)
            # pred_reward shape: (batch, 1)
            step_reward_loss = tf.reduce_mean((tf.squeeze(pred_reward, axis=1) - step_rewards) ** 2)

            # Policy loss
            step_policy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_policy_logits, labels=step_policies)
            step_policy_loss = tf.reduce_mean(step_policy_loss)

            # Combine step losses: scale value loss by 0.25
            step_loss = 0.25 * step_value_loss + step_policy_loss + step_reward_loss

            # Scale the gradient by 1/num_unroll_steps
            step_loss = scale_gradient(step_loss, 1.0 / num_unroll_steps)

            # Add to total loss
            total_loss += step_loss

            # Update total running losses for logging
            value_loss += step_value_loss
            reward_loss += step_reward_loss
            policy_loss += step_policy_loss

            # Half the gradient of the latent representation for next step
            current_hidden = scale_gradient(next_hidden, 0.5)

        # Log losses
        # total_value_loss here includes initial + recurrent steps
        total_value_loss = value_loss
        total_policy_loss = policy_loss
        total_reward_loss = reward_loss

        train_results.total_losses.append(total_loss.numpy())
        train_results.value_losses.append(total_value_loss.numpy())
        train_results.policy_losses.append(total_policy_loss.numpy())
        train_results.reward_losses.append(total_reward_loss.numpy())

        # Minimize
        variables = network.cb_get_variables()()
        grads = tape.gradient(total_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

    network.train_steps += 1
