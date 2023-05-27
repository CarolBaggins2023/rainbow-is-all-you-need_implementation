from collections import deque
import numpy as np
import torch
from torch import Tensor
from typing import Dict, SupportsFloat, Tuple


def combine_shape(capacity: int, shape):
	if shape is None:
		return capacity
	elif np.isscalar(shape):
		return capacity, shape
	else:
		return capacity, *shape


class NStepReplayBuffer:
	def __init__(
			self,
			observation_space,
			action_space,
			capacity: int,
			batch_size: int,
			# parameters of n-step learning
			n_step: int,
			discount_factor: float,
	):
		self.observation_buf = np.zeros(
			shape=combine_shape(capacity, observation_space.shape[0]),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			shape=combine_shape(capacity, action_space.shape),
			dtype=np.int64,
		)
		self.reward_buf = np.zeros(shape=capacity, dtype=np.float32)
		self.next_observation_buf = np.zeros_like(self.observation_buf)
		self.done_buf = np.zeros_like(self.reward_buf)
		
		self.ptr, self.size, self.capacity = 0, 0, capacity
		self.batch_size = batch_size
		
		# n_step_deque is used to store last n step transitions.
		self.n_step = n_step
		self.n_step_deque = deque(maxlen=self.n_step)
		self.discount_factor = discount_factor
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: SupportsFloat,
			next_observation: np.ndarray,
			done: bool,
	):
		transition = dict(
			observation=observation,
			action=action,
			reward=reward,
			next_observation=next_observation,
			done=done
		)
		self.n_step_deque.append(transition)
		
		# Single step transition is not ready.
		if len(self.n_step_deque) < self.n_step:
			return
		
		# In n-step learning, we don't store every single transition.
		# We only store transitions with n-step.
		# So, we need to get the transition 'before' n_steps.
		observation, action, reward, next_observation, done = \
			self._get_n_step_transition()
		
		# Then just store the transition as we did before.
		# But notice that, now, (observation, action, reward,
		# next_observation, done) is completely different with the input,
		# unless the input transition reaches the terminal state.
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.next_observation_buf[self.ptr] = next_observation
		self.done_buf[self.ptr] = done
		
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample_batch(self, device) -> Dict[str, Tensor]:
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
	
	def _get_n_step_transition(
			self
	) -> Tuple[np.ndarray, np.ndarray, SupportsFloat, np.ndarray, bool]:
		# We can see that, actually, when we call store(), we are storing
		# transition beginning from the oldest (observation, action)
		# in the deque.
		begin_transition = self.n_step_deque[0]
		observation, action, reward, next_observation, done = \
			begin_transition['observation'], begin_transition['action'], \
			begin_transition['reward'], begin_transition['next_observation'], \
			begin_transition['done']
		# If the beginning transition reaches terminal state, its consequent
		# transitions are meaningless, so just return it.
		if done is True:
			return observation, action, reward, next_observation, done
		
		# If the beginning transition doesn't reach terminal state,
		# we can consider consequent n-step transitions, as long as they are
		# in the same episode.
		discount = self.discount_factor
		for transition in list(self.n_step_deque)[1:]:
			reward += discount * transition['reward']
			next_observation, done = transition['next_observation'], transition['done']
			discount *= self.discount_factor
			
			# If this transition has already reached terminal state,
			# transitions after it are not in the same episode with the
			# begging transition.
			if transition['done'] is True:
				break
		
		return observation, action, reward, next_observation, done
	
	def __len__(self) -> int:
		return self.size
