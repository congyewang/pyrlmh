from gymnasium.envs.registration import register

from ._env import RLMHEnv

register(
    id="RLMHEnv-v0",
    entry_point="src.rlmcmc.envs._env:RLMHEnv",
)


__all__ = ["RLMHEnv"]
