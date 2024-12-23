import math
import random
from itertools import count

import gymnasium as gym
import matplotlib.pyplot as plot
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from ReplayMemory import ReplayMemory, Transition

logger = structlog.get_logger()

env = gym.make('CartPole-v1', render_mode='human')
plot.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
logger.info(f'Using device: {device}')


class IDK:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    steps_done = 0
    episode_durations = []

    def __init__(self) -> None:
        self.state, self.info = env.reset()
        self.actions = env.action_space.n
        self.n_observations = len(self.state)

        self.policy_net = DQN(self.n_observations, self.actions).to(device)
        self.target_net = DQN(self.n_observations, self.actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self, state_) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state_).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    def optimize_model(self) -> None:
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda state_: state_ is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([state_ for state_ in batch.next_state
                                           if state_ is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def reinforcement_learning(self) -> None:
        match torch.cuda.is_available(), torch.backends.mps.is_available():
            case True, True:
                max_episodes = 600
            case _:
                max_episodes = 500

        max_score = 0
        for episode in range(max_episodes):
            score = 0
            # Initialize the environment and get its state
            state_, info_ = env.reset()
            state_ = torch.tensor(state_, dtype=torch.float32, device=device).unsqueeze(0)
            for tick_ in count():
                action = self.select_action(state_)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                score += reward.item()

                match terminated:
                    case True:
                        next_state = None
                    case _:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state_, action, next_state, reward)

                # Move to the next state
                state_ = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    max_score = max(max_score, score)
                    logger.info(f'Episode: {episode} Score:{score}, MaxScore: {max_score}')
                    self.episode_durations.append(tick_ + 1)
                    break

        logger.info('Complete')
        logger.info(f'Max Score: {max_score}')
        plot.ioff()
        plot.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    IDK().reinforcement_learning()
