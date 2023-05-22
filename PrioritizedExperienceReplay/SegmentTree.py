from typing import Callable
import operator


class SegmentTree:
	def __init__(self, capacity: int, operation: Callable, init_value: float):
		self.capacity = capacity
		self.operation = operation
		self.tree = list(init_value for _ in range(2 * self.capacity))
	
	def operate(self, query_start: int, query_end: int) -> float:
		# For example, l = [0]*10. Then, l[0:-1] is equal to l[0:9].
		if query_end < 0:
			query_end += self.capacity
		query_end -= 1
		
		# In this implementation of segment tree, node index starts from 1.
		# So, if a node's idx is ptr, its left child's idx is 2*ptr,
		# and its right child's idx is 2*ptr+1.
		return self._operate_helper(
			query_start, query_end, node_ptr=1, node_start=0, node_end=self.capacity-1
		)
	
	def _operate_helper(
			self,
			query_start: int,
			query_end: int,
			node_ptr: int,
			node_start: int,
			node_end: int
	) -> float:
		# Note:
		# Range of the right child is (mid, node_end] or [mid+1, node_end],
		# not [mid, node_end].
		if (query_start == node_start) and (query_end == node_end):
			return self.tree[node_ptr]
		node_mid = (node_start+node_end)//2
		if query_end <= node_mid:
			return self._operate_helper(
				query_start, query_end, 2*node_ptr, node_start, node_mid
			)
		else:
			if query_start >= node_mid+1:
				return self._operate_helper(
					query_start, query_end, 2*node_ptr+1, node_mid+1, node_end
				)
			else:
				return self.operation(
					self._operate_helper(
						query_start, node_mid, 2*node_ptr, node_start, node_mid
					),
					self._operate_helper(
						node_mid+1, query_end, 2*node_ptr+1, node_mid+1, node_end
					)
				)
	
	def __setitem__(self, idx: int, value: float):
		# Only leaves are accessible to users.
		# Only nodes from self.tree[capacity] to self.tree[2*capacity-1]
		# are leaf nodes.
		idx += self.capacity
		self.tree[idx] = value
		
		# We not only need to update the value of the target leaf node,
		# but also need to update its parents.
		idx //= 2
		while idx >= 1:
			self.tree[idx] = self.operation(self.tree[2*idx], self.tree[2*idx+1])
			idx //= 2
	
	def __getitem__(self, idx: int) -> float:
		# For the same reason in self.__setitem__().
		idx += self.capacity
		return self.tree[idx]


class SumSegmentTree(SegmentTree):
	def __init__(self, capacity: int):
		super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)
	
	def sum(self, start: int, end: int) -> float:
		# Returns arr[start] + ... + arr[end].
		return super().operate(query_start=start, query_end=end)
	
	def retrieve(self, upper_bound: float) -> int:
		# Find the highest index about upper bound in the tree.
		idx = 1
		# While not leaf node.
		while idx < self.capacity:
			left_child = 2*idx
			right_child = 2*idx+1
			if self.tree[left_child] > upper_bound:
				idx = left_child
			else:
				upper_bound -= self.tree[left_child]
				idx = right_child
				
		# Inverse process of idx += self.capacity in self.__setitem__()
		# and self.__getitem__().
		idx -= self.capacity
		return idx


class MinSegmentTree(SegmentTree):
	def __init__(self, capacity: int):
		super().__init__(capacity=capacity, operation=min, init_value=float("inf"))
	
	def min(self, start: int, end: int) -> float:
		# Returns min(arr[start], ..., arr[end]).
		return super().operate(query_start=start, query_end=end)
