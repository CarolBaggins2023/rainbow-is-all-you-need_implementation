import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from typing import Dict, Tuple
from copy import deepcopy
import time
import os
import gymnasium as gym

from DQNAgent import DQNAgent
from ReplayBuffer import ReplayBuffer
from log_data import EpochLogger


def count_variable(module: nn.Module):
	return sum([np.prod(p.shape) for p in module.parameters()])


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
	# Make a seed-specific subfolder in the experiment directory.
	hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
	subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
	logger_kwargs = dict(
		output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name
	)
	return logger_kwargs


def dqn(
		make_env,
		make_test_env,
		seed: int,
		logger_kwargs: Dict,
		replay_buffer_capacity: int = 1_000,
		discount_factor: float = 0.99,
		lr: float = 1e-3,
		batch_size: int = 32,
		steps_per_epoch: int = 4_000,
		num_epochs: int = 10,
		save_freq: int = 1,
		target_update: int = 100,
):
	# Set up device.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Set up logger.
	logger = EpochLogger(**logger_kwargs)
	logger.save_config(locals())
	
	# Set random seed.
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	# Instantiate train and test environments.
	env, test_env = make_env(), make_test_env()
	
	# Instantiate main and target agent.
	agent = DQNAgent(env.observation_space, env.action_space).to(device)
	target_agent = deepcopy(agent).to(device)
	for parameter in target_agent.parameters():
		parameter.requires_grad = False
	target_agent.eval()
	
	num_variables = count_variable(agent.value_function)
	logger.log("\nNumber of variables:\t value function: %d\n" % num_variables)
	
	# Instantiate replay buffer.
	replay_buffer = ReplayBuffer(
		env.observation_space,
		env.action_space,
		capacity=replay_buffer_capacity,
	)
	
	# Set up loss function.
	def compute_loss(data: Dict) -> Tuple[Tensor, Dict]:
		observation, action, reward, next_observation, done = \
			data['observation'], data['action'], data['reward'],\
			data['next_observation'], data['done']
		
		# Critical:
		# Make sure the action value is in the right shape.
		action_value = torch.squeeze(
			agent(observation).gather(1, action.reshape(-1, 1))
		)
		# Note:
		# torch.max get a sequence type data, we need to take [0] of it.
		next_action_value = torch.squeeze(
			torch.max(target_agent(next_observation), dim=1, keepdim=True)[0]
		)
		td_target = reward + discount_factor * (1 - done) * next_action_value
		loss = torch.mean((action_value - td_target) ** 2)
		# loss = torch.nn.functional.smooth_l1_loss(action_value, td_target)
		
		information = dict(
			Action_Value=action_value.detach().cpu().numpy(),
			TD_Target=td_target.cpu().numpy(),
			Loss=loss.detach().cpu().numpy(),
		)
		
		return loss, information
	
	# Set up the optimizer.
	optimizer = Adam(params=agent.parameters(), lr=lr)
	
	# Set up model saving.
	logger.setup_pytorch_saver(agent)
	
	# Set up the update function.
	def update(step: int):
		data = replay_buffer.sample_batch(batch_size, device)
		
		optimizer.zero_grad()
		loss, information = compute_loss(data)
		loss.backward()
		optimizer.step()
		
		logger.store(**information)
		
		agent.epsilon_decrease()
		logger.store(Epsilon=agent.epsilon)
		
		if (step + 1) % target_update == 0:
			target_agent.load_state_dict(agent.state_dict())
		
		return
	
	def test_agent_performance():
		episode_len, episode_reward = 0, 0
		observation, _ = test_env.reset(seed=seed)
		done = False
		while not done:
			action = agent.select_action(
				torch.tensor(observation, dtype=torch.float32).to(device),
				is_test=True,
			)
			next_observation, reward, terminated, truncated, _ = test_env.step(action)
			
			episode_len += 1
			episode_reward += reward
			done = terminated or truncated
			
			# Critical!!!
			observation = next_observation
			
		logger.store(
			Test_Episode_Len=episode_len,
			Test_Episode_Reward=episode_reward,
		)
	
	# Run the main loop.
	def main_loop():
		start_time = time.time()
		episode_len, episode_reward = 0, 0
		observation, _ = env.reset(seed=seed)
		for step in range(steps_per_epoch * num_epochs):
			action = agent.select_action(
				torch.tensor(observation, dtype=torch.float32).to(device)
			)
			next_observation, reward, terminated, truncated, _ = env.step(action)
			
			episode_len += 1
			episode_reward += reward
			replay_buffer.store(
				observation, action, reward, next_observation, done=terminated or truncated
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
			if len(replay_buffer) > batch_size:
				update(step)
			
			# End of epoch handling.
			if (step + 1) % steps_per_epoch == 0:
				epoch = (step + 1) // steps_per_epoch
				
				# Save environment.
				if (epoch % save_freq == 0) or (epoch + 1 == num_epochs):
					logger.save_state({'Env': env}, None)
					
				test_agent_performance()
				
				# Log information.
				logger.log_tabular('Epoch', epoch)
				logger.log_tabular('Episode_Reward', average_only=True)
				logger.log_tabular('Episode_Len', average_only=True)
				logger.log_tabular('Test_Episode_Reward', average_only=True)
				logger.log_tabular('Test_Episode_Len', average_only=True)
				logger.log_tabular('Total_Interactions', step)
				logger.log_tabular('Action_Value', average_only=True)
				logger.log_tabular('TD_Target', average_only=True)
				logger.log_tabular('Loss', average_only=True)
				logger.log_tabular('Epsilon', average_only=True)
				logger.log_tabular('Time', time.time() - start_time)
				logger.dump_tabular()
	
	main_loop()
	env.close()
	test_env.close()


def main():
	env_name = 'CartPole-v1'
	exp_name = 'DQN_CartPole'
	num_runs = 3
	seeds = [10 * i for i in range(num_runs)]
	data_dir = ''.join(
		['./data/', time.strftime("%Y-%m-%d_%H-%M-%S_"), exp_name]
	)
	for seed in seeds:
		logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir)
		dqn(lambda: gym.make(env_name), lambda: gym.make(env_name, render_mode='human'), seed, logger_kwargs)


if __name__ == '__main__':
	main()
