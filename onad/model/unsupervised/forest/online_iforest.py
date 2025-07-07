from numpy import ndarray
import numpy as np
from typing import Dict
from onad.base.model import BaseModel

class OnlineINode:
    @staticmethod
    def create(inode_type: str, **kwargs) -> 'OnlineINode':
        if inode_type == 'boundedrandomprojectiononlineinode':
            # Only Alias, no seperate Class needed
            return OnlineINode(**kwargs)
        raise ValueError(f'Bad inode type {inode_type}')

    def __init__(self, data_size: int, children: ndarray, depth: int, node_index: int,
                 split_vector=None, split_value=None):
        self.data_size = data_size
        self.children = children
        self.depth = depth
        self.node_index = node_index
        self.split_vector = split_vector  # numpy array or None
        self.split_value = split_value    # float or None
        self.is_leaf = children is None or len(children) == 0


class OnlineITree:
    @staticmethod
    def create(itree_type: str, **kwargs) -> 'OnlineITree':
        if itree_type == 'boundedrandomprojectiononlineitree':
            return BoundedRandomProjectionOnlineITree(**kwargs)
        raise ValueError(f'Bad itree type {itree_type}')

    @staticmethod
    def c_factor(n: int) -> float:
        # Expected pathlenght for n Samples (Standard Isolation Forest c(n))
        if n <= 1:
            return 0
        else:
            # Euler-Mascheroni Constant ~0.5772
            return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def __init__(self, max_leaf_samples: int, type: str, subsample: float,
                 branching_factor: int, data_size: int):
        self.max_leaf_samples = max_leaf_samples
        self.type = type
        self.subsample = subsample
        self.branching_factor = branching_factor
        self.data_size = data_size
        self.root = None
        self.next_node_index = 0

    def learn(self, data: ndarray):
        self.next_node_index = 0
        _, self.root = self.recursive_build(data, 0)

    def recursive_build(self, data: ndarray, depth: int):
        n_samples = data.shape[0]
        if n_samples <= self.max_leaf_samples or n_samples < self.branching_factor:
            node = OnlineINode(data_size=n_samples, children=np.array([]), depth=depth,
                              node_index=self.next_node_index)
            self.next_node_index += 1
            return self.next_node_index, node

        # Random Split-Vector (random projection)
        split_vector = np.random.normal(size=data.shape[1])
        split_vector /= np.linalg.norm(split_vector)  # Normieren

        projections = data @ split_vector

        # Median as Split-value
        split_value = np.median(projections)

        left_data = data[projections <= split_value]
        right_data = data[projections > split_value]

        # if a site is empty, split
        if len(left_data) == 0 or len(right_data) == 0:
            size = n_samples // 2
            left_data = data[:size]
            right_data = data[size:]

        children = []
        self.next_node_index += 1  # for actual node
        _, left_child = self.recursive_build(left_data, depth + 1)
        _, right_child = self.recursive_build(right_data, depth + 1)
        children.append(left_child)
        children.append(right_child)

        node = OnlineINode(data_size=n_samples, children=np.array(children, dtype=object),
                          depth=depth, node_index=self.next_node_index,
                          split_vector=split_vector, split_value=split_value)
        return self.next_node_index, node

    def path_length(self, x: ndarray, node: OnlineINode, depth: int = 0) -> float:
        if node.is_leaf:
            return depth + OnlineITree.c_factor(node.data_size)
        proj = np.dot(x, node.split_vector)
        if proj <= node.split_value:
            return self.path_length(x, node.children[0], depth + 1)
        else:
            return self.path_length(x, node.children[1], depth + 1)

    def predict(self, data: ndarray) -> ndarray:
        lengths = np.array([self.path_length(x, self.root) for x in data])
        c = OnlineITree.c_factor(self.max_leaf_samples)
        scores = 2 ** (-lengths / c)
        return scores


class OnlineIForest(BaseModel):
    def __init__(self, num_trees: int, window_size: int, branching_factor: int, max_leaf_samples: int,
                 type: str, subsample: float, n_jobs: int):
        self.num_trees = num_trees
        self.window_size = window_size
        self.branching_factor = branching_factor
        self.max_leaf_samples = max_leaf_samples
        self.type = type
        self.subsample = subsample
        self.trees = []
        self.data_window = []
        self.data_size = 0
        self.n_jobs = n_jobs

    def learn_batch(self, data: ndarray):
        raise NotImplementedError

    def score_batch(self, data: ndarray):
        raise NotImplementedError


class BoundedRandomProjectionOnlineITree(OnlineITree):
    def __init__(self, max_leaf_samples, type, subsample, branching_factor, data_size):
        super().__init__(max_leaf_samples, type, subsample, branching_factor, data_size)


class BoundedRandomProjectionOnlineIForest(OnlineIForest):
    def __init__(self, num_trees=10, window_size=256, branching_factor=2, max_leaf_samples=64,
                 type='fixed', subsample=1.0, n_jobs=1):
        super().__init__(num_trees, window_size, branching_factor, max_leaf_samples, type, subsample, n_jobs)
        self.trees = [BoundedRandomProjectionOnlineITree(
            max_leaf_samples=max_leaf_samples,
            type=type,
            subsample=subsample,
            branching_factor=branching_factor,
            data_size=window_size
        ) for _ in range(num_trees)]
        self.data_window = []

    def learn_one(self, x: Dict[str, float]) -> None:
        arr = np.array(list(x.values()))
        if len(self.data_window) >= self.window_size:
            self.data_window.pop(0)
        self.data_window.append(arr)
        self.data_size = len(self.data_window)
        data_array = np.array(self.data_window)
        for tree in self.trees:
            tree.learn(data_array)

    def score_one(self, x: Dict[str, float]) -> float:
        arr = np.array([list(x.values())])
        scores = np.mean([tree.predict(arr)[0] for tree in self.trees])
        return scores

    def learn_batch(self, data: ndarray):
        for row in data:
            if len(self.data_window) >= self.window_size:
                self.data_window.pop(0)
            self.data_window.append(row)
        self.data_size = len(self.data_window)
        for tree in self.trees:
            tree.learn(np.array(self.data_window))

    def score_batch(self, data: ndarray):
        return np.mean([tree.predict(data) for tree in self.trees], axis=0)
