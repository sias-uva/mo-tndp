import numpy as np

import gymnasium as gym
from gymnasium import spaces
from motndp.city import City


class MOTNDP(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {}

    def __init__(self, city: City, nr_stations: int, render_mode=None):
        self.city = city
        self.nr_stations = nr_stations
        self.stations_placed = 0
        # size of the grid
        # self.grid_size = self.city.grid_size
        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.city.grid_size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 8 actions, corresponding to
        # 0: walk up
        # 1: walk up-right
        # 2: walk right
        # 3: walk down-right
        # 4: walk down
        # 5: walk down-left
        # 6: walk left
        # 7: walk up-left
        self.action_space = spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([-1, 1]),
            2: np.array([0, 1]),
            3: np.array([1, 1]),
            4: np.array([1, 0]),
            5: np.array([1, -1]),
            6: np.array([0, -1]),
            7: np.array([-1, -1]),
        }

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        # self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location}
    
    def _get_info(self):
        return {}
    
    def _calculate_reward(self, segment, use_pct=True):
        assert self.city.group_od_mx, 'Cannot use multi-objective reward without group definitions. Provide --groups_file argument'

        sat_od_mask = self.city.satisfied_od_mask(segment)
        sat_group_ods = np.zeros(len(self.city.group_od_mx))
        sat_group_ods_pct = np.zeros(len(self.city.group_od_mx))
        for i, g_od in enumerate(self.city.group_od_mx):
            sat_group_ods[i] = (g_od * sat_od_mask).sum().item()
            sat_group_ods_pct[i] = sat_group_ods[i] / g_od.sum()

        if use_pct:
            group_rw = sat_group_ods_pct
        else:
            group_rw = sat_group_ods
        
        return group_rw
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random, by sampling its x and y coordinates
        agent_x = self.np_random.integers(0, self.city.grid_x_size)
        agent_y = self.np_random.integers(0, self.city.grid_y_size)
        self._agent_location = np.array([agent_x, agent_y])
        self.stations_placed = 0

        observation = self._get_obs()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation
    
    def step(self, action):
        self.stations_placed += 1
        # Map the action to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_location = np.array([
            self._agent_location[0] + direction[0],
            self._agent_location[1] + direction[1]
        ])
        # If we leave the grid, we stay in the same place
        if np.any((new_location < 0) | (new_location[0] > self.city.grid_x_size - 1) | ((new_location[1] > self.city.grid_y_size - 1))):
            new_location = self._agent_location

        # We add a new dimension to the agent's location to match grid_to_vector's generalization
        segment = np.array([self.city.grid_to_vector(self._agent_location[None, :]).item(), self.city.grid_to_vector(new_location[None, :]).item()])
        reward = self._calculate_reward(segment)
        # An episode is done iff the agent has reached the target
        terminated = self.stations_placed >= self.nr_stations

        self._agent_location = new_location
        observation = self._get_obs()
        info  = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info

