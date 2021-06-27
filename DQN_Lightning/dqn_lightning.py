"""
Deep Reinforcement Learning: Deep Q-network (DQN)
"""
from DQN_Lightning.agent import Agent
from DQN_Lightning.network import DQN
from DQN_Lightning.replay_buffer import ReplayBuffer
from DQN_Lightning.rl_dataset import RLDataset
import argparse
from collections import OrderedDict
from typing import List, Tuple
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model
    >>> DQNLightning(env="CartPole-v1")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    DQNLightning(
      (net): DQN(
        (net): Sequential(...)
      )
      (target_net): DQN(
        (net): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        env: str,
        replay_size: int = 200,
        warm_start_steps: int = 200,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_last_frame: int = 200,
        sync_rate: int = 10,
        lr: float = 1e-2,
        batch_size: int = 4,
        tau: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.sync_rate = sync_rate
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau

        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.populate(self.warm_start_steps)

        self.episode_idx = 0
        self.episode_return = 0
        self.episode_length = 0

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state `x` through the network and gets the `q_values` of each action as an output
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output

    def get_deterministic_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device="cuda", requires_grad=False)
        q_values = self.net(state)
        _, action = torch.max(q_values, dim=0)
        action = int(action.item())
        return action

    def evaluate(self, num_episodes = 5, render=False):
        succesful_episodes, episodes_returns, episodes_lengths = 0, [], []
        for episode in range(1, num_episodes + 1):
            observation = self.env.reset()
            episode_return = 0
            for step in range(self.env._max_episode_steps):
                action = self.get_deterministic_action(observation)
                observation, reward, done, info = self.env.step(action)
                episode_return += reward
                if render:
                    self.env.render()
                if done:
                    break
            if ("success" in info) and info['success']:
                succesful_episodes += 1
            episodes_returns.append(episode_return) 
            episodes_lengths.append(step)
        accuracy = succesful_episodes/num_episodes
        return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_epoch_end(self, _):
        val_return = float('-inf') 
        if self.episode_done:
            # Log every episode end
            wandb = self.logger.experiment
            wandb_logs = {"train/episode_return": self.episode_return,
                          "train/episode_length": self.episode_length,
                          "train/episode_number": self.episode_idx}
            if self.episode_idx % 10 == 0:
                val_accuracy, val_return, val_length = self.evaluate()
                wandb_logs['validation/accuracy'] = val_accuracy
                wandb_logs['validation/avg_episode_return'] = val_return
                wandb_logs['validation/avg_episode_length'] = val_length

            wandb.log(wandb_logs)
            self.episode_return, self.episode_length = 0, 0
            self.episode_idx += 1
        
        # Monitored metric to save model
        self.log('val_episode_return', val_return, on_epoch=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch received
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start - self.global_step + 1 / self.eps_last_frame)

        # step through environment with agent
        reward, self.episode_done = self.agent.play_step(self.net, epsilon, device)
        self.episode_return += reward
        self.episode_length += 1

        # calculates training loss
        loss = self.dqn_mse_loss(batch)
        self.log('loss', loss, on_step=True)

        #Update target networks
        self.soft_update(self.target_net, self.net, self.tau)
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    @staticmethod
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("DQNLightning")
        parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment tag")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
        parser.add_argument("--replay_size", type=int, default=int(1e5), help="capacity of the replay buffer")
        parser.add_argument(
            "--warm_start_steps",
            type=int,
            default=1000,
            help="how many samples do we use to fill our buffer at the start of training"
        )
        parser.add_argument("--eps_last_frame", type=int, default=10000, help="what frame should epsilon stop decaying")
        parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
        parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
        return parent_parser


def main(args) -> None:
    model = DQNLightning(**vars(args))
    checkpoint_val_callback = ModelCheckpoint(
        monitor='val_episode_return',
        dirpath="trained_models",
        filename='model_{step:06d}-{val_episode_return:.3f}',
        save_top_k=10,
        verbose=True,
        mode='max',
        save_last=True,
        )
    wandb_logger = WandbLogger(name="DQN_model")
    trainer = pl.Trainer(gpus=1, max_steps=10000,
                         callbacks=[checkpoint_val_callback],
                         logger=wandb_logger)
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = DQNLightning.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)