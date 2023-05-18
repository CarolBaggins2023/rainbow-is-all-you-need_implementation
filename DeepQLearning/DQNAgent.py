import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class MLP(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, out_activation):
		super().__init__()
		self.fc1 = nn.Linear(in_features=in_dim, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=128)
		self.fc3 = nn.Linear(in_features=128, out_features=out_dim)
		self.activation = nn.ReLU()
		self.out_activation = out_activation
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.fc3(x)
		x = self.out_activation(x)
		return x


class DQNAgent(nn.Module):
	def __init__(
			self,
			observation_space,
			action_space,
			epsilon_decay: float = 1.0 / 2000,
			max_epsilon: float = 1.0,
			min_epsilon: float = 0.1,
	):
		super().__init__()
		self.observation_space = observation_space
		self.action_space = action_space
		self.value_function = MLP(
			in_dim=self.observation_space.shape[0],
			out_dim=self.action_space.n,
			out_activation=nn.Identity(),
		)
		self.epsilon, self.max_epsilon, self.min_epsilon, self.epsilon_decay =\
			1.0, max_epsilon, min_epsilon, epsilon_decay
	
	def select_action(
			self,
			observation: Tensor,
			is_test: bool = False,
	) -> np.ndarray:
		with torch.no_grad():
			if self.epsilon > np.random.random() and not is_test:
				action = self.action_space.sample()
			else:
				action = self.value_function(observation).argmax()
				action = action.detach().cpu().numpy()
		
		return action
	
	def epsilon_decrease(self):
		self.epsilon = max(
			self.min_epsilon,
			self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
		)
	
	def forward(self, observation: Tensor) -> Tensor:
		action_value = self.value_function(observation)
		return action_value
