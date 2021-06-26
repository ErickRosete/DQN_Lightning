import torch.nn as nn

class DQN(nn.Module):
    """
    Simple MLP network
    >>> DQN(10, 5)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQN(
      (net): Sequential(...)
    )
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
