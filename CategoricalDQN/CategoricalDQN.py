import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class CategoricalMLP(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			atom_size: int,
			support: Tensor,
	):
		super().__init__()
		self.out_features = out_features
		self.atom_size = atom_size
		self.support = support
		
		self.fc1 = nn.Linear(in_features=in_features, out_features=128)
		self.fc2 = nn.Linear(in_features=128, out_features=128)
		self.fc3 = nn.Linear(
			in_features=128,
			out_features=out_features * atom_size
		)
		self.activation = nn.ReLU()
	
	def forward(self, x: Tensor) -> Tensor:
		value_distribution = self.get_value_distribution(x)
		# The output is still action value, not its distribution.
		# So we need to sum up the dimension of atom size.
		action_value = torch.sum(self.support * value_distribution, dim=-1)
		return action_value
	
	def get_value_distribution(self, x: Tensor) -> Tensor:
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		# Dim1 and dim2 of the output of self.fc3 consist a matrix,
		# each row of which denotes the distribution of an action.
		x = self.fc3(x).reshape(-1, self.out_features, self.atom_size)
		# But above row is not a real probability distribution, since it
		# does not sum to 1. So, we execute softmax.
		value_distribution = nn.functional.softmax(input=x, dim=-1)
		# for avoid nan
		value_distribution = torch.clamp(input=value_distribution, min=1e-3)
		return value_distribution
		

class CategoricalDQN(nn.Module):
	def __init__(
			self,
			observation_space,
			action_space,
			max_epsilon: float = 1.0,
			min_epsilon: float = 0.1,
			epsilon_decay: float = 1.0 / 2000,
			# parameters of categorical DQN
			atom_size: int = 51,
			value_max: float = 200.0,
			value_min: float = 0.0,
	):
		super().__init__()
		self.max_epsilon, self.min_epsilon, self.epsilon_decay =\
			max_epsilon, min_epsilon, epsilon_decay
		self.epsilon = self.max_epsilon
		self.action_space = action_space
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.support = torch.linspace(
			start=value_min, end=value_max, steps=atom_size
		).to(device)
		self.value_function = CategoricalMLP(
			observation_space.shape[0], action_space.n, atom_size, self.support
		)
	
	def forward(self, observation: Tensor, action: Tensor = None) -> Tensor:
		action_value = self.value_function(observation)
		if action is None:
			return action_value
		else:
			return action_value.gather(1, action.reshape(-1, 1))
	
	def get_value_distribution(self, observation: Tensor) -> Tensor:
		return self.value_function.get_value_distribution(observation)
	
	def select_action(self, observation: Tensor) -> np.ndarray:
		with torch.no_grad():
			if np.random.random() < self.epsilon:
				action = self.action_space.sample()
			else:
				action_value = self.value_function(observation)
				action = torch.argmax(input=action_value).detach().cpu().numpy()
		return action
	
	def decrease_epsilon(self):
		self.epsilon = max(
			self.min_epsilon,
			self.epsilon - (self.max_epsilon-self.min_epsilon)*self.epsilon_decay,
		)
