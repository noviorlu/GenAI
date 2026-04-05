"""
TO EDIT: A simple, generic replay buffer

Functions to edit:
    sample_random_data: line 103
"""
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from cs224r.infrastructure.utils import *


class ReplayBuffer():
    """
    Defines a replay buffer to store past trajectories

    Attributes
    ----------
    paths: list
        A list of rollouts
    obs: np.array
        An array of observations
    acs: np.array
        An array of actions
    rews: np.array
        An array of rewards
    next_obs:
        An array of next observations
    terminals:
        An array of terminals

    Methods
    -------
    add_rollouts:
        Add rollouts and processes them into their separate components
    sample_random_data:
        Selects a random batch of data
    sample_recent_data:
        Selects the most recent batch of data
    """
    def __init__(self, max_size: int = 1000000):

        self.max_size: int = max_size

        # Store each rollout
        self.paths: List[Dict[str, Any]] = []

        # Store (concatenated) component arrays from each rollout
        self.obs: Optional[np.ndarray] = None
        self.acs: Optional[np.ndarray] = None
        self.rews: Optional[Union[np.ndarray, List[np.ndarray]]] = None
        self.next_obs: Optional[np.ndarray] = None
        self.terminals: Optional[np.ndarray] = None

    def __len__(self) -> int:
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths: List[Dict[str, Any]], concat_rew: bool = True):
        """
        Adds paths into the buffer and processes them into separate components

        :param paths: a list of paths to add
        :param concat_rew: whether rewards should be concatenated or appended
        """
        # Add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # Convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards = (
            convert_listofrollouts(paths))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = concatenated_rewards[-self.max_size:] if concat_rew else unconcatenated_rewards[-self.max_size:] # type: ignore
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            # Tell Pylance explicitly that these are no longer None
            assert self.obs is not None
            assert self.acs is not None
            assert self.rews is not None
            assert self.next_obs is not None
            assert self.terminals is not None

            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, concatenated_rewards]
                )[-self.max_size:]
            else:
                if isinstance(unconcatenated_rewards, list) and isinstance(self.rews, list):
                    self.rews += unconcatenated_rewards
                elif isinstance(self.rews, list):
                    self.rews.append(unconcatenated_rewards) # type: ignore
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        """
        Samples a batch of random transitions

        :param batch_size: the number of transitions to sample
        :return:
            obs: a batch of observations
            acs: a batch of actions
            rews: a batch of rewards
            next_obs: a batch of next observations
            terminals: a batch of terminals
        """
        assert self.obs is not None
        assert self.acs is not None
        assert self.rews is not None
        assert isinstance(self.rews, np.ndarray)
        assert self.next_obs is not None
        assert self.terminals is not None

        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        ## return batch_size number of random entries
        ## from each of the 5 component arrays above
        indices = np.random.permutation(self.obs.shape[0])[:batch_size]

        return (
            self.obs[indices],
            self.acs[indices],
            self.rews[indices],
            self.next_obs[indices],
            self.terminals[indices],
        )
    
    def sample_recent_data(self, batch_size=1):
        """
        Samples a batch of the most recent transitions transitions

        :param batch_size: the number of transitions to sample
        :return:
            obs: a batch of observations
            acs: a batch of actions
            rews: a batch of rewards
            next_obs: a batch of next observations
            terminals: a batch of terminals
        """
        assert self.obs is not None
        assert self.acs is not None
        assert self.rews is not None
        assert isinstance(self.rews, np.ndarray)
        assert self.next_obs is not None
        assert self.terminals is not None

        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )
