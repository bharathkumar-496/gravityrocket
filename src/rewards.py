"""Reward rules. Each one sits between the game and PPO and changes only the reward."""
import gymnasium as gym


def is_steady(obs):
    """Upright and slow, using the raw observation: vx, vy, tilt."""
    return abs(obs[4]) < 0.2 and abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2


class TimeFee(gym.Wrapper):
    """Built-in reward, minus a tiny charge per frame. Should still land."""

    def __init__(self, env, fee=0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverHack(gym.Wrapper):
    """The bad rule: +1 for every frame spent upright and slow.
    Pays for sitting in a nice spot, so staying in the air out-earns landing."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward + (1.0 if is_steady(obs) else 0.0), terminated, truncated, info


class HoverFix(gym.Wrapper):
    """The fix: same idea, but pay for the *change* into steadiness, plus a fee per frame.
    Sitting still earns nothing, and waiting costs something."""

    def __init__(self, env, bonus=1.0, fee=0.05):
        super().__init__(env)
        self.bonus, self.fee, self.was_steady = bonus, fee, False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.was_steady = is_steady(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = is_steady(obs)
        reward += self.bonus * (float(now) - float(self.was_steady)) - self.fee
        self.was_steady = now
        return obs, reward, terminated, truncated, info


RULES = {
    "baseline": lambda env: env,
    "time_fee": TimeFee,
    "hover_hack": HoverHack,
    "hover_fix": HoverFix,
}
