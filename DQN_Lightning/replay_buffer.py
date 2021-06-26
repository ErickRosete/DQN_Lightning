"""
Deep Reinforcement Learning: Deep Q-network (DQN)
"""

from collections import deque, namedtuple
from typing import Tuple
import numpy as np


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them
    >>> ReplayBuffer(5)  # doctest: +ELLIPSIS
    <...reinforce_learn_Qnet.ReplayBuffer object at ...>
    """

    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity: size of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )
