import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

from NoisyLayer import NoisyLayer


class NoisyNetwork(nn.Module):
	def __init__(
			self,
			in_features: int,
			out_features: int,
			out_activation: Callable,
	):
		super().__init__()
		self.fc = nn.Linear(in_features=in_features, out_features=128)
		self.noisy_layer1 = NoisyLayer(in_features=128, out_features=128)
		self.noisy_layer2 = NoisyLayer(in_features=128, out_features=out_features)
		self.activation = nn.ReLU()
		self.out_activation = out_activation()
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.activation(self.fc(x))
		x = self.activation(self.noisy_layer1(x))
		x = self.noisy_layer2(x)
		x = self.out_activation(x)
		return x
	
	def reset_noise(self):
		self.noisy_layer1.reset_noise()
		self.noisy_layer2.reset_noise()


class NoisyAgent(nn.Module):
	def __init__(self, observation_space, action_space):
		super().__init__()
		self.observation_space = observation_space
		self.action_space = action_space
		self.value_function = NoisyNetwork(
			in_features=self.observation_space.shape[0],
			out_features=self.action_space.n,
			out_activation=nn.Identity,
		)

	def select_action(self, observation: Tensor) -> np.ndarray:
		with torch.no_grad():
			action_value = self.value_function(observation)
			action = torch.argmax(input=action_value, dim=-1)
			return action.detach().cpu().numpy()
	
	def forward(self, observation: Tensor, action: Tensor = None):
		action_value = self.value_function(observation)
		if action is None:
			return action_value
		else:
			return action_value.gather(1, action.reshape(-1, 1))
	
	def reset_noise(self):
		self.value_function.reset_noise()
