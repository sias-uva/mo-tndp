import numpy as np

import gymnasium as gym
from gymnasium import spaces
from motndp.city import City
from motndp.constraints import Constraints


class MOTNDP(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, city: City, constraints: Constraints, nr_stations: int, starting_loc=None, obs_type='full_dict', od_type='pct', chained_reward=False, render_mode=None):
        """
        Args:
            city (City): City object that contains the grid and the groups.
            constraints (Constraints): Transport constraints object with the constraints on movement in the grid.
            nr_stations (int): Episode length. Total number of stations to place (each station is an episode step).
            starting_loc (tuple): Set the default starting location of the agent in the grid. If None, the starting location is chosen randomly, or chosen in _reset().
            obs_type (str): Type of observation to return. Can be 'full_dict' (returns a dictionary with all information) or 'location_vector' (returns a one-hot vector of the agent's location in the grid), 'location' (returns the agent's location in grid coordinates), 'location_vid' (returns the agent's location as a discrete index).
            od_type (str): Type of Origin Destination metric. Can be 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).
            chained_reward (bool): If True, each new station will receive an additional reward based not only on the ODs covered between the immediate previous station, but also those before.
            render_mode (str): RENDERING IS NOT IMPLEMENTED YET.
        """
        
        assert obs_type in ['full_dict', 'location_vector', 'location', 'location_vid'], 'Invalid observation type. Choose one of: full_dict, location_vector, location, location_vid'
        assert od_type in ['pct', 'abs'], 'Invalid Origin-Destination Type. Choose one of: pct, abs'
        
        # city environment
        self.city = city
        # action-masking constraints
        self.mask_actions = constraints.mask_actions
        # total number of stations (steps) to place
        self.nr_stations = nr_stations
        # default starting location of the agent
        self.starting_loc = starting_loc
        # observation type
        self.observation_type = obs_type
        # origin-destination calculation type
        self.od_type = od_type
        # chained reward
        # An example of the chained reward: Episode with two actions leads to transport line 1 -> 2 -> 3. 
        # If chained_reward is false, the reward received after the second action will be the OD between 2-3.
        # If chained_reward is true, the reward received after the second action will be the OD between 2-3 AND 1-3, because station 1 and 3 are now connected (and therefore their demand is deemed satisfied).
        self.chained_reward = chained_reward
        self.stations_placed = 0
        # visited cells in the grid (by index)
        self.covered_cells_vid = []
        # visited cells in the grid (by coordinate)
        self.covered_cells_gid = [] 
        # covered segments in the grid (pairs of cells)
        self.covered_segments = []
        # Stations from the existing lines that are connected to the line that the agent is currently building.
        self.connections_with_existing_lines = set()
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
        # Allowed actions are updated at each step, based on the current location
        self.action_mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # Nr of groups in the city (used for multi-objective reward)
        self.nr_groups = self.city.groups.shape[0]
        # Reward space is a vector of length `self.nr_groups` with values between 0 and 1 -- used in morl-baselines
        self.reward_space = spaces.Box(low=np.float32([0] * self.nr_groups), high=np.float32([1] * self.nr_groups), dtype=np.float32)

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


    def _get_obs(self):
        ## TODO: fix this mess
        if self.observation_type == 'full_dict':
            return {'location': self._agent_location, 
                    'location_vid': self._agent_location_vid, 
                    'location_vector': self._agent_location_vector}
        elif self.observation_type == 'location_vector':
            return self._agent_location_vector
        elif self.observation_type == 'location':
            return self._agent_location
        elif self.observation_type == 'location_vid':
            return [self._agent_location_vid]
    
    def _get_info(self):
        return {'segments': self.covered_segments, 'action_mask': self.action_mask, 'covered_cells_vid': self.covered_cells_vid, 'covered_cells_gid': self.covered_cells_gid}
    
    def _calculate_reward(self, segment):
        assert self.city.group_od_mx is not None, 'Cannot use multi-objective reward without group definitions. Provide --groups_file argument'

        # Return zero rewards if segment is already covered
        if segment in self.covered_segments:
            return np.zeros(self.city.group_od_mx.shape[0]), np.empty((0, 2))

        segment = np.asarray(segment)
        if self.chained_reward:
            # Convert covered cells to numpy array (if not already)
            cells_to_chain = np.asarray(self.covered_cells_vid)

            # Get stations connected to the new segment
            connected_stations = self.city.connections_with_existing_lines(segment)
            
            # Extend only if connected stations exist
            if connected_stations:
                # self.connections_with_existing_lines = list(set(self.connections_with_existing_lines).union(connected_stations))
                self.connections_with_existing_lines.update(connected_stations)

            # Get the satisfied OD mask with connections and cells to chain
            sat_od_mask, sat_od_pairs = self.city.satisfied_od_mask(
                segment, 
                cells_to_chain=cells_to_chain, 
                connected_cells=self.connections_with_existing_lines, 
                segments_to_ignore=self.covered_segments, 
                return_od_pairs=True
            )
        else:
            # Get the satisfied OD mask without chained reward
            sat_od_mask, sat_od_pairs = self.city.satisfied_od_mask(
                segment, 
                segments_to_ignore=self.covered_segments, 
                return_od_pairs=True
            )

        # Compute satisfied group ODs and their percentages        
        sat_group_ods = (self.city.group_od_mx * sat_od_mask).sum(axis=(1, 2))
        sat_group_ods_pct = np.divide(sat_group_ods, self.city.group_od_sum, out=np.zeros_like(sat_group_ods), where=self.city.group_od_sum != 0)

        # Determine reward type (percentage or absolute)
        group_rw = sat_group_ods_pct if self.od_type == 'pct' else sat_group_ods
        
        ##### TODO delete this
        # non_zero_od = (g_od * sat_od_mask).nonzero()
        # non_zero_pairs = np.array([non_zero_od[0].tolist(), non_zero_od[1].tolist()]).T
        # self.all_sat_od_pairs.extend(non_zero_pairs.tolist())
        #####

        return group_rw, sat_od_pairs
        
    def _update_agent_location(self, new_location):
        """Given the new location of the agent in grid coordinates (x, y), update a number of internal variables.
        _agent_location: the new location of the agent in grid coordinates (x, y)
        _agent_location_vid: the new location of the agent as a discrete discrete index (0,0) -> 0, (0,1) -> 1, ....
        _agent_location_vector: the new location of the agent as a one-hot vector.

        Args:
            new_location (_type_): _description_
        """
        self._agent_location = new_location
        self._agent_location_vid = self.city.grid_to_vector(self._agent_location[None, :]).item()
        self._agent_location_vector = np.zeros(self.observation_space.n)
        self._agent_location_vector[self._agent_location_vid] = 1
    
    def _update_action_mask(self, location, prev_action=None):
        # Apply action mask based on the location of the agent (it should stay inside the grid) & previously visited cells (it should not visit the same cell twice)
        possible_locations = location + self._action_to_direction
        self.action_mask = self.mask_actions(location, possible_locations, self.covered_cells_gid)

    def is_action_allowed(self, location, action):
        possible_locations = location + self._action_to_direction
        action_mask = self.mask_actions(location, possible_locations, self.covered_cells_gid)
        return action_mask[action] == 1

    def reset(self, seed=None, loc=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # loc argument supersedes self.starting_loc
        starting_loc = loc if loc else self.starting_loc
        if starting_loc:
            if type(starting_loc) == tuple:
                starting_loc = np.array([starting_loc[0], starting_loc[1]])
            self._update_agent_location(starting_loc)
            
        # Choose the agent's location uniformly at random, by sampling its x and y coordinates
        else:
            agent_x = self.np_random.integers(0, self.city.grid_x_size)
            agent_y = self.np_random.integers(0, self.city.grid_y_size)
            self._update_agent_location(np.array([agent_x, agent_y]))

        self.stations_placed =  1
        self.covered_cells_vid = [self.city.grid_to_vector(self._agent_location[None, :]).item()]
        self.covered_cells_gid = [self._agent_location]
        self.covered_segments = []
        self.connections_with_existing_lines = set()
        
        # todo delete this
        self.all_sat_od_pairs = []

        self._update_action_mask(self._agent_location)
        observation = self._get_obs()

        return observation, self._get_info()
    
    def step(self, action):
        # Map the action to the direction we walk in
        direction = self._action_to_direction[action]
        
        if self.is_action_allowed(self._agent_location, action):
            new_location = np.array([
                self._agent_location[0] + direction[0],
                self._agent_location[1] + direction[1]
            ])
            self.stations_placed += 1

            # We add a new dimension to the agent's location to match grid_to_vector's generalization
            from_idx = self.city.grid_to_vector(self._agent_location[None, :]).item()
            to_idx = self.city.grid_to_vector(new_location[None, :]).item()
            reward, sat_od_pairs = self._calculate_reward([from_idx, to_idx])
            
            # Update the covered segments and cells, based on sat_od_pairs
            for pair in sat_od_pairs:
                self.covered_segments.append(pair.tolist())
                self.covered_segments.append(pair[::-1].tolist()) # add the reverse OD pair 

            self.covered_cells_vid.append(to_idx)
            self.covered_cells_gid.append(new_location)

            # Update the agent's location
            self._update_agent_location(new_location)
            # Update the action mask
            self._update_action_mask(self._agent_location)
        else:
            raise Exception("Not allowed action was taken. Make sure you apply the constraints to the action selection.")
            # reward = np.zeros(self.nr_groups)
            # TODO reconsider if this counter should be here (because the agent is not moving, thus there is no station placed). 
            # if I remove it I need to consider that the episode needs to terminate somehow.
            # self.stations_placed += 1

        # An episode is done if the agent has placed all stations under the budget or if there's no more actions to take
        terminated = self.stations_placed >= self.nr_stations or np.sum(self.action_mask) == 0

        observation = self._get_obs()
        info  = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info
