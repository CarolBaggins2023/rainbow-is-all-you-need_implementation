import math
import torch
import torch.nn as nn
from torch import Tensor


class NoisyLayer(nn.Module):
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		
		# parameters of weight
		self.weight_mu = nn.Parameter(
			data=torch.zeros(
				size=[self.out_features, self.in_features],
				dtype=torch.float32,
			)
		)
		self.weight_sigma = nn.Parameter(
			data=torch.zeros_like(input=self.weight_mu, dtype=torch.float32)
		)
		# noise of weight
		self.weight_epsilon = nn.Parameter(
			data=torch.zeros_like(input=self.weight_mu, dtype=torch.float32),
			requires_grad=False,
		)
		
		# parameters of bias
		self.bias_mu = nn.Parameter(
			data=torch.zeros(size=[self.out_features], dtype=torch.float32)
		)
		self.bias_sigma = nn.Parameter(
			data=torch.zeros_like(input=self.bias_mu, dtype=torch.float32)
		)
		# noise of bias
		self.bias_epsilon = nn.Parameter(
			data=torch.zeros_like(input=self.bias_mu, dtype=torch.float32),
			requires_grad=False,
		)
		
		self.reset_parameters()
		self.reset_noise()
	
	# initialization
	def reset_parameters(self):
		# Refer to "3.2 initialization" of noisy networks in the paper.
		mu_range = 1 / math.sqrt(self.in_features)
		nn.init.uniform_(tensor=self.weight_mu, a=-mu_range, b=mu_range)
		nn.init.constant_(
			tensor=self.weight_sigma,
			val=0.5/math.sqrt(self.in_features),
		)
		nn.init.uniform_(tensor=self.bias_mu, a=-mu_range, b=mu_range)
		nn.init.constant_(
			tensor=self.bias_sigma,
			val=0.5/math.sqrt(self.in_features),
		)
	
	# Make new noise.
	def reset_noise(self):
		# epsilon_in is f(epsilon_i), epsilon_out is f(epsilon_j) in (10) and (11)
		x_in = torch.randn(size=[self.in_features])
		epsilon_in = torch.sign(x_in) * torch.sqrt(torch.abs(x_in))
		x_out = torch.randn(size=[self.out_features])
		epsilon_out = torch.sign(x_out) * torch.sqrt(torch.abs(x_out))
		
		self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
		self.bias_epsilon.copy_(epsilon_out)
	
	def forward(self, x: Tensor) -> Tensor:
		return nn.functional.linear(
			input=x,
			weight=self.weight_mu + self.weight_sigma * self.weight_epsilon,
			bias=self.bias_mu + self.bias_sigma * self.bias_epsilon,
		)
