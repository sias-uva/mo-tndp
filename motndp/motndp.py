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
        # visited cells in the grid (by index)
        self.covered_cells_vid = []
        # visited cells in the grid (by coordinate)
        self.covered_cells_gid = [] 
        # covered segments in the grid (pairs of cells)
        self.covered_segments = []
        # size of the grid
        # self.grid_size = self.city.grid_size
        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, self.city.grid_size - 1, shape=(2,), dtype=int),
        #     }
        # )
        # self.observation_space = spaces.Box(0, self.city.grid_size - 1, shape=(2,), dtype=int)
        self.observation_space = spaces.Discrete(self.city.grid_size)

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
        self.action_mask = np.ones(self.action_space.n, dtype=np.int8)

        """
        The following array maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = np.array([
            [-1, 0],
            [-1, 1],
            [0 , 1],
            [1 , 1],
            [1 , 0],
            [1, -1],
            [0, -1],
            [-1, -1]
        ])

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
        return {'location': self._agent_location}
    
    def _get_info(self):
        return {'segments': self.covered_segments, 'action_mask': self.action_mask}
    
    def _calculate_reward(self, segment, use_pct=True):
        assert self.city.group_od_mx, 'Cannot use multi-objective reward without group definitions. Provide --groups_file argument'

        if segment in self.covered_segments:
            return np.zeros(len(self.city.group_od_mx))

        segment = np.array(segment)
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
    
    def _update_action_mask(self, location, prev_action=None):
        # Apply action mask based on the location of the agent (it should stay inside the grid) & previously visited cells (it should not visit the same cell twice)
        possible_locations = location + self._action_to_direction 
        self.action_mask = np.all(possible_locations >= 0, axis=1) &  \
                            (possible_locations[:, 0] < self.city.grid_x_size) & \
                            (possible_locations[:, 1] < self.city.grid_y_size) & \
                            (~np.isin(self.city.grid_to_vector(possible_locations), self.covered_cells_vid)) # mask out visited cells
        self.action_mask = self.action_mask.astype(np.int8)
                
        # Dissallow the agent to go back to the previous cell
        # TODO This is a hacky way to do this. Should be a better way to do this.
        # if prev_action:
        #     if prev_action <= 3:
        #         self.action_mask[prev_action + 4] = 0
        #     elif prev_action >= 4:
        #         self.action_mask[prev_action - 4] = 0


    def reset(self, seed=None, loc=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if loc:
            if type(loc) == tuple:
                loc = np.array([loc[0], loc[1]])
            self._agent_location = loc
        # Choose the agent's location uniformly at random, by sampling its x and y coordinates
        else:
            agent_x = self.np_random.integers(0, self.city.grid_x_size)
            agent_y = self.np_random.integers(0, self.city.grid_y_size)
            self._agent_location = np.array([agent_x, agent_y])
            
        self.stations_placed =  1
        self.covered_cells_vid = [self.city.grid_to_vector(self._agent_location[None, :]).item()]
        self.covered_cells_gid = [self._agent_location]
        self.covered_segments = []

        self._update_action_mask(self._agent_location)
        observation = self._get_obs()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, self._get_info()
    
    def step(self, action):
        # Map the action to the direction we walk in
        direction = self._action_to_direction[action]
        new_location = np.array([
            self._agent_location[0] + direction[0],
            self._agent_location[1] + direction[1]
        ])
        self.stations_placed += 1

        # We add a new dimension to the agent's location to match grid_to_vector's generalization
        from_idx = self.city.grid_to_vector(self._agent_location[None, :]).item()
        to_idx = self.city.grid_to_vector(new_location[None, :]).item()
        reward = self._calculate_reward([from_idx, to_idx])

        self.covered_segments.append([from_idx, to_idx])
        self.covered_segments.append([to_idx, from_idx])
        self.covered_cells_vid.append(to_idx)
        self.covered_cells_gid.append(new_location)
        # An episode is done iff the agent has reached the target
        terminated = self.stations_placed >= self.nr_stations

        self._agent_location = new_location
        
        # Update the action mask
        self._update_action_mask(self._agent_location, action)

        observation = self._get_obs()
        info  = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info

