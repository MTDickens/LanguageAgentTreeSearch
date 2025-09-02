import gym
import numpy as np


class FrozenLakeTextEnv:
    def __init__(self, map_name: str = "4x4", is_slippery: bool = True):
        self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
        self.traj = {"observations": [], "actions": []}
        self.steps = 0

    def reset(self, seed=None, return_info=False, options=None):
        obs = self.env.reset(seed=seed, options=options)
        if isinstance(obs, tuple):
            obs = obs[0]
        self.traj = {"observations": [], "actions": []}
        self.steps = 0
        desc = self._obs_to_text(obs)
        question = "You are in the FrozenLake gridworld. Reach the goal 'G' without falling in holes 'H'. Use actions: left, right, up, down."
        observation = question + "\n" + desc
        info = {"steps": self.steps, "answer": None}
        self.traj["observations"].append(observation)
        return (observation,  info)

    def step(self, action: str):
        action = action.strip()
        if action.startswith('think['):
            obs_text = 'OK.'
            reward = 0.0
            done = False
            info = {"steps": self.steps, "answer": None}
        else:
            a = self._action_to_int(action)
            obs, reward, done, info = self.env.step(a)
            desc = self._obs_to_text(obs)
            obs_text = desc
            info = {"steps": self.steps, "answer": None}
            if done:
                if reward == 1.0:
                    obs_text = f"Episode finished, reward = 1.0\n"
                else:
                    obs_text = f"Episode finished, reward = 0.0\n"
        self.traj["observations"].append(obs_text)
        self.traj["actions"].append(action)
        self.steps += 1
        return obs_text, float(reward), bool(done), info

    def clone_state(self):
        # Gym FrozenLake doesn't support clone; we store trajectory only.
        return {"traj": self.traj.copy(), "steps": self.steps}

    def _action_to_int(self, action: str) -> int:
        action = action.lower()
        if action.startswith('left'):
            return 0
        if action.startswith('down'):
            return 1
        if action.startswith('right'):
            return 2
        if action.startswith('up'):
            return 3
        raise AssertionError("Invalid action for FrozenLake. Use: left/right/up/down, or think[...]")

    def _obs_to_text(self, obs):
        # obs is index into the grid (row-major)
        try:
            desc = self.env.unwrapped.desc
        except Exception:
            desc = None
        size = desc.shape[0] if desc is not None else int(np.sqrt(self.env.observation_space.n))
        row = obs // size
        col = obs % size
        grid_lines = []
        if desc is not None:
            for r in range(size):
                line = ''.join([chr(c) for c in desc[r]])
                grid_lines.append(line)
        grid_text = "\n".join(grid_lines) if grid_lines else f"Grid size: {size}x{size}"
        return f"Position: ({row}, {col})\n{grid_text}"
