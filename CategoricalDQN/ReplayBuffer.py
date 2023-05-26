import numpy as np
import torch
from torch import Tensor
from typing import Dict, SupportsFloat


def combine_shape(capacity: int, shape):
	if shape is None:
		return capacity
	elif np.isscalar(shape):
		return capacity, shape
	else:
		return capacity, *shape


class ReplayBuffer:
	def __init__(
			self,
			observation_space,
			action_space,
			capacity: int,
			batch_size: int,
	):
		self.observation_buf = np.zeros(
			shape=combine_shape(capacity, observation_space.shape[0]),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			shape=combine_shape(capacity, action_space.shape),
			dtype=np.int64,
		)
		self.reward_buf = np.zeros(capacity, dtype=np.float32)
		self.next_observation_buf = np.zeros_like(
			a=self.observation_buf, dtype=np.float32
		)
		self.done_buf = np.zeros_like(a=self.reward_buf, dtype=np.float32)
		
		self.ptr, self.size, self.capacity = 0, 0, capacity
		self.batch_size = batch_size
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: SupportsFloat,
			next_observation: np.ndarray,
			done: bool,
	):
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.next_observation_buf[self.ptr] = next_observation
		self.done_buf[self.ptr] = done
		
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample_batch(self, device: torch.device) -> Dict[str, Tensor]:
		idxs = np.random.randint(low=0, high=self.size, size=self.batch_size)
		data = dict(
			observation=self.observation_buf[idxs],
			action=self.action_buf[idxs],
			reward=self.reward_buf[idxs],
			next_observation=self.next_observation_buf[idxs],
			done=self.done_buf[idxs],
		)
		data_tensor = {
			key: torch.tensor(value, dtype=torch.float32).to(device)
			for key, value in data.items()
		}
		data_tensor['action'] = torch.tensor(
			data['action'], dtype=torch.int64
		).to(device)
		return data_tensor
	
	def __len__(self) -> int:
		return self.size
