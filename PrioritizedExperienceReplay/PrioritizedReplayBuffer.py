import numpy as np
from typing import SupportsFloat, Dict, List
from torch import Tensor
import torch

from SegmentTree import SumSegmentTree, MinSegmentTree


def combine_shape(capacity: int, shape):
	if shape is None:
		return capacity
	elif np.isscalar(shape):
		return capacity, shape
	else:
		return capacity, *shape


class PrioritizedReplayBuffer:
	def __init__(
			self,
			observation_space,
			action_space,
			capacity: int,
			batch_size: int,
			alpha: float,
	):
		"""If there is any doubts, refer to the picture inserted in the note."""
		self.observation_buf = np.zeros(
			shape=combine_shape(capacity, observation_space.shape[0]),
			dtype=np.float32,
		)
		self.action_buf = np.zeros(
			shape=combine_shape(capacity, action_space.shape),
			dtype=np.int64,
		)
		self.reward_buf = np.zeros(shape=capacity, dtype=np.float32)
		self.next_observation_buf = np.zeros_like(
			a=self.observation_buf, dtype=np.float32
		)
		self.done_buf = np.zeros_like(a=self.reward_buf, dtype=np.float32)
		
		self.ptr, self.size, self.capacity = 0, 0, capacity
		self.batch_size = batch_size
		
		# alpha determines how much prioritization is used.
		self.alpha = alpha
		self.max_priority = 1.0
		# tree_ptr plays the same role of self.ptr in storing experience.
		# And we can see that changes of tree_ptr and ptr
		# are synchronous (in self.store()).
		self.tree_ptr = 0
		
		# SegmentTree is a full binary tree, so its capacity is the power of 2.
		tree_capacity = 1
		while tree_capacity < self.capacity:
			tree_capacity *= 2
		
		self.sum_tree = SumSegmentTree(capacity=tree_capacity)
		self.min_tree = MinSegmentTree(capacity=tree_capacity)
	
	def store(
			self,
			observation: np.ndarray,
			action: np.ndarray,
			reward: SupportsFloat,
			next_observation: np.ndarray,
			done: float,
	):
		# Buffer handling.
		self.observation_buf[self.ptr] = observation
		self.action_buf[self.ptr] = action
		self.reward_buf[self.ptr] = reward
		self.next_observation_buf[self.ptr] = next_observation
		self.done_buf[self.ptr] = done
		
		self.ptr = (self.ptr+1) % self.capacity
		self.size = min(self.size+1, self.capacity)
		
		# Priority tree handling.
		# New transition has the maximum priority.
		self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
		self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
		self.tree_ptr = (self.tree_ptr+1) % self.capacity

	def sample_batch(self, device, beta: float) -> Dict:
		# Improvement of priority replay buffer.
		# In priority replay buffer, we no longer randomly select transitions.
		# idxs = np.random.randint(low=0, high=self.size, size=self.batch_size)
		# Now, we select transitions according to their priority.
		idxs = self._sample_proportional()
		weights = np.array([self._calculate_weight(idx, beta) for idx in idxs])
		
		data = dict(
			indices=idxs,
			observation=self.observation_buf[idxs],
			action=self.action_buf[idxs],
			reward=self.reward_buf[idxs],
			next_observation=self.next_observation_buf[idxs],
			done=self.done_buf[idxs],
			weights=weights,
		)
		
		data_tensor = {
			key: torch.tensor(value, dtype=torch.float32).to(device)
			for key, value in data.items()
		}
		data_tensor['indices'] = idxs
		data_tensor['action'] = torch.tensor(
			data['action'], dtype=torch.int64
		).to(device)
		return data_tensor
	
	def update_priorities(self, idxs: List[int], priorities: np.ndarray):
		for idx, priority in zip(idxs, priorities):
			# If a transition has been sampled, its priority should decrease.
			self.sum_tree[idx] = priority**self.alpha
			self.min_tree[idx] = priority**self.alpha
			
			self.max_priority = max(self.max_priority, priority)
	
	def _sample_proportional(self) -> List[int]:
		idxs = list()
		# We segment the stored transitions into several parts,
		# thus it is less likely to sample close transitions.
		# (Help to remove correlation between transitions.)
		priority_total = self.sum_tree.sum(start=0, end=self.size-1)
		segment = priority_total/self.batch_size
		
		for i in range(self.batch_size):
			start = segment*i
			end = segment*(i+1)
			upper_bound = np.random.uniform(low=start, high=end)
			idx = self.sum_tree.retrieve(upper_bound)
			idxs.append(idx)
		
		return idxs
	
	def _calculate_weight(self, idx: int, beta: float) -> float:
		# Calculate weight.
		priority_total = self.sum_tree.sum(start=0, end=self.capacity)
		priority_proportion = self.sum_tree[idx] / priority_total
		weight = (self.size*priority_proportion)**(-beta)
		
		# Calculate max weight.
		priority_min = self.min_tree.min(start=0, end=self.capacity)
		priority_min_proportion = priority_min / priority_total
		max_weight = (self.size*priority_min_proportion)**(-beta)
		
		normalized_weight = weight/max_weight
		return normalized_weight

	def __len__(self) -> int:
		return self.size
