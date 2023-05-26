from copy import deepcopy
import gymnasium as gym
import numpy as np
import os
import time
from typing import Callable, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor

from CategoricalDQN import CategoricalDQN
from log_data import EpochLogger
from ReplayBuffer import ReplayBuffer


def count_variable(module: nn.Module) -> int:
	return sum([np.prod(p.shape) for p in module.parameters()])


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	logger_kwargs = dict(
		output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name
	)
	return logger_kwargs


def categorical_dqn(
		make_env: Callable,
		seed: int,
		logger_kwargs: Dict,
		# parameters of categorical DQN
		atom_size: int = 51,
		value_max: float = 200.0,
		value_min: float = 0.0,
		# parameters for replay buffer
		replay_buffer_capacity: int = 1_000,
		batch_size: int = 32,
		# parameters for computing loss
		discount_factor: float = 0.99,
		# parameters for optimizers
		lr: float = 1e-3,
		# parameters for update
		target_update: int = 100,
		# parameters for main loop
		num_epochs: int = 10,
		steps_per_epoch: int = 4_000,
		# parameters for end of epoch handling
		save_freq: int = 1,
		# parameters for testing agent's performance
		test_episodes: int = 5,
):
	# Set device.
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Set up logger
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate training and testing environments.
	env, test_env = make_env(), make_env()
	
	# Instantiate online and target agents.
	agent = CategoricalDQN(env.observation_space, env.action_space).to(device)
	target_agent = deepcopy(agent).to(device)
	for parameter in target_agent.parameters():
		parameter.requires_grad = False
	num_variables = count_variable(agent)
	logger.log("\nNumber of variables: \t %d\n" % num_variables)
	
	# Instantiate replay buffer.
	replay_buffer = ReplayBuffer(
		env.observation_space,
		env.action_space,
		capacity=replay_buffer_capacity,
		batch_size=batch_size,
	)
	
	# Set up the loss function.
	def compute_loss(data) -> Tensor:
		observation, action, reward, next_observation, done = \
			data['observation'], data['action'], data['reward'], \
			data['next_observation'], data['done']
		
		with torch.no_grad():
			next_action = torch.argmax(target_agent(next_observation), dim=1)
			next_value_distribution = target_agent.get_value_distribution(observation)
			next_value_distribution = next_value_distribution[
				range(batch_size), next_action
			]
			
			# Compute the projection onto support.
			# Shape of t_z is (batch_size, atom_size).
			t_z = torch.clamp(
				input=(reward.reshape(-1, 1) +
					(1-done).reshape(-1, 1) * discount_factor *
					target_agent.support),
				min=value_min,
				max=value_max,
			)
			delta_z = (value_max - value_min) / (atom_size - 1)
			b = (t_z - value_min) / delta_z
			l = b.floor().long()
			u = b.ceil().long()
			
			# offset is a tensor with shape (batch_size, atom_size)
			# with element of datatype long.
			offset = torch.linspace(
				start=0, end=(batch_size-1)*atom_size, steps=batch_size
			).long().unsqueeze(1).expand(batch_size, atom_size).to(device)
			
			# probability distribution of t_z
			project_value_distribution = torch.zeros_like(
				next_value_distribution
			).to(device)
			project_value_distribution.reshape(-1).index_add_(
				0,
				(l + offset).view(-1),
				(next_value_distribution * (u.float() - b)).reshape(-1),
			)
			project_value_distribution.reshape(-1).index_add_(
				0,
				(u + offset).view(-1),
				(next_value_distribution * (b - l.float())).reshape(-1),
			)
		
		value_distribution = agent.get_value_distribution(observation)
		value_distribution = value_distribution[range(batch_size), action]
		log_probability = torch.log(value_distribution)
		
		loss = - torch.mean(
			torch.sum(project_value_distribution*log_probability, dim=1)
		)
		
		return loss
		
	# Set up optimizers.
	optimizer = Adam(params=agent.parameters(), lr=lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up update function.
	def update(step):
		data = replay_buffer.sample_batch(device)
		
		# Update online agent.
		optimizer.zero_grad()
		loss = compute_loss(data)
		loss.backward()
		optimizer.step()
		
		agent.decrease_epsilon()
		
		# Update target agent.
		if (step + 1) % target_update == 0:
			target_agent.load_state_dict(agent.state_dict())
	
	# There seems no need to execute target_agent.reset_noise().
	# Because target_agent is only used in compute_loss(), and
	# target_agent.reset_noise() is always called.
	
	def test_agent_performance():
		for _ in range(test_episodes):
			episode_len, episode_reward = 0, 0
			observation, _ = test_env.reset(seed=seed)
			done = False
			while not done:
				action = agent.select_action(
					torch.tensor(observation, dtype=torch.float32).to(device)
				)
				next_observation, reward, terminated, truncated, _ = \
					test_env.step(action)
				
				episode_len += 1
				episode_reward += reward
				done = terminated or truncated
				
				# Critical!!!
				observation = next_observation
			
			logger.store(
				Test_Episode_Len=episode_len,
				Test_Episode_Reward=episode_reward,
			)
	
	# Set up the main loop.
	def main_loop():
		start_time = time.time()
		episode_len, episode_reward = 0, 0
		observation, _ = env.reset(seed=seed)
		for step in range(num_epochs * steps_per_epoch):
			action = agent.select_action(
				torch.tensor(observation, dtype=torch.float32).to(device)
			)
			next_observation, reward, terminated, truncated, _ = env.step(action)
			
			episode_len += 1
			episode_reward += reward
			replay_buffer.store(
				observation, action, reward, next_observation, done=terminated
			)
			
			# Critical!!!
			observation = next_observation
			
			# End of episode handling.
			if terminated or truncated:
				logger.store(
					Episode_Len=episode_len,
					Episode_Reward=episode_reward,
				)
				episode_len, episode_reward = 0, 0
				observation, _ = env.reset(seed=seed)
			
			# Update handling.
			if len(replay_buffer) >= batch_size:
				update(step)
			
			# End of epoch handling.
			if (step + 1) % steps_per_epoch == 0:
				epoch = (step + 1) // steps_per_epoch
				
				if (epoch % save_freq == 0) or (epoch + 1 == num_epochs):
					logger.save_state({'Env': env}, None)
				
				# Test agent's performance.
				test_agent_performance()
				
				# Log information.
				logger.log_tabular('Epoch', epoch)
				logger.log_tabular('Test_Episode_Len', with_min_and_max=True)
				logger.log_tabular('Test_Episode_Reward', with_min_and_max=True)
				logger.log_tabular('Time', time.time() - start_time)
				logger.dump_tabular()
	
	main_loop()


def main():
	exp_name = 'NoisyNetwork_CartPole'
	env_name = 'CartPole-v1'
	num_runs = 3
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join(
		['./data/', time.strftime("%Y-%m-%d_%H-%M-%S_"), exp_name]
	)
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir)
		categorical_dqn(lambda: gym.make(env_name), seed, logger_kwargs)


if __name__ == '__main__':
	main()
