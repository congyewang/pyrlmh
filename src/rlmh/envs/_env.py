from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import scipy
import scipy.linalg
from numpy.typing import NDArray


class RLMHEnvBase(gym.Env, ABC):
    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, NDArray[np.float64]]], Union[float, np.float64]
        ],
        initial_sample: Union[np.float64, NDArray[np.float64]],
        initial_covariance: Union[np.float64, NDArray[np.float64], None] = None,
    ) -> None:
        super().__init__()

        self.sample_dim: int = int(np.prod(initial_sample.shape))  # sample dimension
        self.steps: int = 1  # iteration time
        # log target probability density function without numerical stabilization
        self.log_target_pdf_unsafe = log_target_pdf_unsafe

        if initial_covariance is None:
            initial_covariance = (2.38 / np.sqrt(self.sample_dim)) * np.eye(
                self.sample_dim
            )

        # Observation specification
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.left_shift(self.sample_dim, 1),),
            dtype=np.float64,
        )
        # Action specification
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.left_shift(self.sample_dim, 1),),
            dtype=np.float64,
        )

        # Initialize State
        initial_next_proposed_sample = self.np_random.multivariate_normal(
            mean=initial_sample, cov=initial_covariance, size=1
        ).flatten()
        self.state = np.concatenate(
            (initial_sample, initial_next_proposed_sample)
        )  # state at this time, s_{t}

        # Store
        self.store_observation: List[NDArray[np.float64]] = []
        self.store_action: List[NDArray[np.float64]] = []
        self.store_log_acceptance_rate: List[Union[np.float64, float]] = []
        self.store_accepted_status: List[bool] = []
        self.store_reward: List[Union[np.float64, float]] = []

        self.store_current_sample: List[NDArray[np.float64]] = []
        self.store_proposed_sample: List[NDArray[np.float64]] = []

        self.store_current_mean: List[NDArray[np.float64]] = []
        self.store_proposed_mean: List[NDArray[np.float64]] = []

        self.store_current_covariance: List[NDArray[np.float64]] = []
        self.store_proposed_covariance: List[NDArray[np.float64]] = []

        self.store_log_target_proposed: List[Union[np.float64, float]] = []
        self.store_log_target_current: List[Union[np.float64, float]] = []
        self.store_log_proposal_proposed: List[Union[np.float64, float]] = []
        self.store_log_proposal_current: List[Union[np.float64, float]] = []

        self.store_accepted_mean: List[NDArray[np.float64]] = []
        self.store_accepted_sample: List[NDArray[np.float64]] = []
        self.store_accepted_covariance: List[NDArray[np.float64]] = []

    def log_target_pdf(self, x: Union[float, np.float64, NDArray[np.float64]]):
        res = self.log_target_pdf_unsafe(x)

        # Numerical stability
        if np.isinf(res):
            res = -np.finfo(np.float64).max

        return res

    def log_proposal_pdf(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> np.float64:
        return self._log_laplace_pdf(x, mean, cov)

    def mcmc_noise(
        self, mean: NDArray[np.float64], cov: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._laprnd(mean, cov)

    def reward_function(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        log_alpha: NDArray[np.float64],
        log_mode: bool = True,
    ) -> np.float64:
        if log_mode:
            return (
                2 * np.log(np.linalg.norm(current_sample - proposed_sample, 2))
                + log_alpha
            )
        else:
            return np.power(
                np.linalg.norm(current_sample - proposed_sample, 2), 2
            ) * np.exp(log_alpha)

    def accepted_process(
        self,
        current_sample: NDArray[np.float64],
        proposed_sample: NDArray[np.float64],
        current_mean: NDArray[np.float64],
        proposed_mean: NDArray[np.float64],
        current_covariance: NDArray[np.float64],
        proposed_covariance: NDArray[np.float64],
    ) -> Tuple[
        bool, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], np.float64
    ]:
        # Calculate Log Target Density
        log_target_current = self.log_target_pdf(current_sample)
        log_target_proposed = self.log_target_pdf(proposed_sample)

        # Calculate Log Proposal Densitys
        log_proposal_current = self.log_proposal_pdf(
            current_sample, proposed_mean, proposed_covariance
        )
        log_proposal_proposed = self.log_proposal_pdf(
            proposed_sample, current_mean, current_covariance
        )

        # Calculate Log Acceptance Rate
        log_alpha = min(
            0.0,
            log_target_proposed
            - log_target_current
            + log_proposal_current
            - log_proposal_proposed,
        )

        # Accept or Reject
        if np.log(self.np_random.random()) < log_alpha:
            accepted_status = True
            accepted_sample = proposed_sample
            accepted_mean = proposed_mean
            accepted_covariance = proposed_covariance
        else:
            accepted_status = False
            accepted_sample = current_sample
            accepted_mean = current_mean
            accepted_covariance = current_covariance

        # Store
        # Store Sample
        self.store_current_sample.append(current_sample)
        self.store_proposed_sample.append(proposed_sample)

        # Store Mean
        self.store_current_mean.append(current_mean)
        self.store_proposed_mean.append(proposed_mean)

        # Store Covariance
        self.store_current_covariance.append(current_covariance)
        self.store_proposed_covariance.append(proposed_covariance)

        # Store Log Densities
        self.store_log_target_current.append(log_target_current)
        self.store_log_target_proposed.append(log_target_proposed)

        self.store_log_proposal_current.append(log_proposal_current)
        self.store_log_proposal_proposed.append(log_proposal_proposed)

        # Store Acceptance
        self.store_accepted_status.append(accepted_status)
        self.store_log_acceptance_rate.append(log_alpha)

        self.store_accepted_sample.append(accepted_sample)
        self.store_accepted_mean.append(accepted_mean)
        self.store_accepted_covariance.append(accepted_covariance)

        return (
            accepted_status,
            accepted_sample,
            accepted_mean,
            accepted_covariance,
            log_alpha,
        )

    def _laprnd(
        self, mean: NDArray[np.float64], cov: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        dim = cov.shape[0]

        # Cholesky factorisation of Sigma = R * R'
        R = scipy.linalg.cholesky(cov, lower=True)

        # sample d times from standard univariate Laplace
        # standard Laplace == difference of two standard exponential RVs
        z = scipy.stats.expon.rvs(scale=1, size=dim) - scipy.stats.expon.rvs(
            scale=1, size=dim
        )

        # transform to correlated multivariate Laplace
        res = mean + np.dot(R, z)

        return res

    def _log_laplace_pdf(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> np.float64:
        # Cholesky factorisation of Sigma = R * R'
        R = scipy.linalg.cholesky(cov, lower=True)

        # return log(pdf(x)) up to an x-independent constant
        res = -np.linalg.norm(np.linalg.solve(R, x - mean), ord=1)

        return res

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Reset environment to initial state and return initial observation.

        Args:
            seed (int | None, optional): Random seed. Defaults to None.
            options (dict[str, Any] | None, optional): Defaults to None.

        Returns:
            tuple[Any, dict[str, Any]]
        """
        # Gym Recommandation
        super().reset(seed=seed, options=options)

        # Set Random Seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self.state, {}

    @abstractmethod
    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError("step is not implemented.")


class RLMHEnv(RLMHEnvBase):
    def __init__(
        self,
        log_target_pdf_unsafe: Callable[
            [Union[float, np.float64, NDArray[np.float64]]], Union[float, np.float64]
        ],
        initial_sample: Union[np.float64, NDArray[np.float64]],
        initial_covariance: Union[np.float64, NDArray[np.float64], None] = None,
        target_mean: Union[np.float64, NDArray[np.float64], None] = None,
        target_covariance: Union[np.float64, NDArray[np.float64], None] = None,
    ) -> None:
        super().__init__(log_target_pdf_unsafe, initial_sample, initial_covariance)

        if target_mean is None:
            target_mean = initial_sample
        if target_covariance is None:
            target_covariance = initial_covariance

        self.target_mean = target_mean
        self.target_covariance = target_covariance

    def _gamma_c(
        self,
        x: Union[float, np.float64, NDArray[np.float64]],
        mu: Union[float, np.float64, NDArray[np.float64]],
        sigma: Union[float, np.float64, NDArray[np.float64]],
        c: float = 10.0,
    ):
        sigma_half = scipy.linalg.sqrtm(sigma)
        eta = np.linalg.norm(np.linalg.solve(sigma_half, (x - mu))) ** 2 / c**2

        if eta >= 0 and eta < 0.5:
            res = 0
        elif eta >= 0.5 and eta < 1:
            res = (1 + np.exp(-((4 * eta - 3) / (4 * eta**2 - 6 * eta + 2)))) ** (-1)
        else:
            res = 1

        return res

    def step(
        self, action: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[Any, Any]]:
        # Unpack state
        current_sample = self.state[0 : self.sample_dim]
        proposed_sample = self.state[self.sample_dim :]

        # Unpack action
        current_psi = action[0 : self.sample_dim]
        proposed_psi = action[self.sample_dim :]

        current_phi = current_psi + self._gamma_c(
            current_sample, self.target_mean, self.target_covariance
        ) * (current_sample - current_psi)
        proposed_phi = proposed_psi + self._gamma_c(
            proposed_sample, self.target_mean, self.target_covariance
        ) * (proposed_sample - proposed_psi)

        # Proposal means
        proposal_covariance_root = scipy.linalg.sqrtm(self.target_covariance)
        current_mean = self.target_mean + proposal_covariance_root @ current_phi
        proposed_mean = self.target_mean + proposal_covariance_root @ proposed_phi

        # Accept or Reject
        _, accepted_sample, accepted_mean, accepted_covariance, log_alpha = (
            self.accepted_process(
                current_sample,
                proposed_sample,
                current_mean,
                proposed_mean,
                self.target_covariance,
                self.target_covariance,
            )
        )

        # Proposal means
        proposal_covariance_root = scipy.linalg.sqrtm(self.target_covariance)
        current_mean = self.target_mean + proposal_covariance_root @ current_phi
        proposed_mean = self.target_mean + proposal_covariance_root @ proposed_phi

        # Update Observation
        next_proposed_sample = self.mcmc_noise(accepted_mean, accepted_covariance)
        observation = np.concatenate((accepted_sample, next_proposed_sample))
        self.state = observation

        # Store
        self.store_observation.append(observation)
        self.store_action.append(action)

        # Calculate Reward
        reward = self.reward_function(current_sample, proposed_sample, log_alpha)
        self.store_reward.append(reward)

        # Update Steps
        self.steps += 1
        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info
