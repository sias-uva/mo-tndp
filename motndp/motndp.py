import numpy as np
import gymnasium as gym
from gymnasium import spaces
from motndp.city import City
from motndp.constraints import Constraints

# Maps actions from `self.action_space` to the direction the agent will walk in if that action is taken.
ACTION_TO_DIRECTION = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
)

class MOTNDP(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        city: City,
        constraints: Constraints,
        nr_stations: int,
        starting_loc=None,
        od_type="pct",
        chained_reward=False,
        render_mode=None,
    ):
        """
        Args:
            city (City): City object that contains the grid and the groups.
            constraints (Constraints): Transport constraints object with the constraints on movement in the grid.
            nr_stations (int): Episode length. Total number of stations to place (each station is an episode step).
            starting_loc (tuple): Set the default starting location of the agent in the grid. If None, the starting location is chosen randomly, or chosen in _reset().
            od_type (str): Type of Origin Destination metric. Can be 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).
            chained_reward (bool): If True, each new station will receive an additional reward based not only on the ODs covered between the immediate previous station, but also those before.
            render_mode (str): RENDERING IS NOT IMPLEMENTED YET.
        """

        assert od_type in ["pct", "abs"], "Invalid Origin-Destination Type. Choose one of: pct, abs"

        self.city = city
        self.mask_actions = constraints.mask_actions
        self.nr_stations = nr_stations
        self.starting_loc = starting_loc
        self.od_type = od_type
        # An example of the chained reward: Episode with two actions leads to transport line 1 -> 2 -> 3.
        # If chained_reward is false, the reward received after the second action will be the OD between 2-3.
        # If chained_reward is true, the reward received after the second action will be the OD between 2-3 AND 1-3, because station 1 and 3 are now connected (and therefore their demand is deemed satisfied).
        self.chained_reward = chained_reward

        self.observation_space = spaces.MultiDiscrete([city.grid_x_size, city.grid_y_size])

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

        # Reward space is a vector of length `self.nr_groups`
        if self.od_type == "pct":
            low_reward = np.zeros(self.nr_groups)
            high_reward = np.ones(self.nr_groups)
        else:
            low_reward = np.zeros(self.nr_groups)
            high_reward = self.city.group_od_mx.sum(axis=(1, 2))

        self.reward_space = spaces.Box(
            low=np.float32(low_reward), high=np.float32(high_reward), dtype=np.float32
        )

    def _get_obs(self):
        return self._loc_grid_coordinates

    def _get_info(self):
        return {
            "segments": self.covered_segments,
            "action_mask": self.action_mask,
            "covered_cells_gid": self.covered_cells_gid,
            "location_grid_coordinates": self._loc_grid_coordinates,
            "location_grid_index": self._loc_grid_index,
            "location_one_hot": self._loc_one_hot,
        }

    def _calculate_reward(self, segment):
        assert (
            self.city.group_od_mx is not None
        ), "Cannot use multi-objective reward without group definitions. Provide --groups_file argument"

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
                return_od_pairs=True,
            )
        else:
            # Get the satisfied OD mask without chained reward
            sat_od_mask, sat_od_pairs = self.city.satisfied_od_mask(
                segment, segments_to_ignore=self.covered_segments, return_od_pairs=True
            )

        # Compute satisfied group ODs and their percentages
        sat_group_ods = (self.city.group_od_mx * sat_od_mask).sum(axis=(1, 2))
        sat_group_ods_pct = np.divide(
            sat_group_ods,
            self.city.group_od_sum,
            out=np.zeros_like(sat_group_ods),
            where=self.city.group_od_sum != 0,
        )

        # Determine reward type (percentage or absolute)
        group_rw = sat_group_ods_pct if self.od_type == "pct" else sat_group_ods

        return group_rw, sat_od_pairs
    
    def _update_agent_location(self, new_location):
        """Given the new location of the agent in grid coordinates (x, y), update a number of internal variables.
        _loc_grid_coordinates: the new location of the agent in grid coordinates (x, y)
        _loc_grid_index: the new location of the agent as a discrete discrete index (0,0) -> 0, (0,1) -> 1, ....
        _loc_one_hot: the new location of the agent as a one-hot vector.

        Args:
            new_location (_type_): _description_
        """
        self._loc_grid_coordinates = new_location
        self._loc_grid_index = self.city.grid_to_vector(self._loc_grid_coordinates[None, :]).item()
        self._loc_one_hot = self.city.one_hot_encode(self._loc_grid_index)

        
    def _update_action_mask(self, location):
        # Apply action mask based on the location of the agent (it should stay inside the grid) & previously visited cells (it should not visit the same cell twice)
        possible_locations = location + ACTION_TO_DIRECTION
        self.action_mask = self.mask_actions(
            location, possible_locations, self.covered_cells_gid
        )

    def is_action_allowed(self, location, action):
        possible_locations = location + ACTION_TO_DIRECTION
        action_mask = self.mask_actions(
            location, possible_locations, self.covered_cells_gid
        )
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
            
        self.stations_placed = 1
        self.covered_cells_gid = [self._loc_grid_coordinates]
        # covered segments in the grid (pairs of cells)
        self.covered_segments = []
        # Stations from the existing lines that are connected to the line that the agent is currently building.
        self.connections_with_existing_lines = set()

        self._update_action_mask(self._loc_grid_coordinates)
        observation = self._get_obs()

        return observation, self._get_info()

    def step(self, action):
        # Map the action to the direction we walk in
        direction = ACTION_TO_DIRECTION[action]

        if self.is_action_allowed(self._loc_grid_coordinates, action):
            new_location = np.array(
                [
                    self._loc_grid_coordinates[0] + direction[0],
                    self._loc_grid_coordinates[1] + direction[1],
                ]
            )
            self.stations_placed += 1

            # We add a new dimension to the agent's location to match grid_to_vector's generalization
            from_idx = self.city.grid_to_vector(self._loc_grid_coordinates[None, :]).item()
            to_idx = self.city.grid_to_vector(new_location[None, :]).item()
            reward, sat_od_pairs = self._calculate_reward([from_idx, to_idx])

            # Update the covered segments and cells, based on sat_od_pairs
            for pair in sat_od_pairs:
                self.covered_segments.append(pair.tolist())
                self.covered_segments.append(
                    pair[::-1].tolist()
                )  # add the reverse OD pair

            self.covered_cells_gid.append(new_location)

            # Update the agent's location
            self._update_agent_location(new_location)
            # Update the action mask
            self._update_action_mask(self._loc_grid_coordinates)
        else:
            raise Exception(
                "Not allowed action was taken. Make sure you apply the constraints to the action selection."
            )

        # An episode is done if the agent has placed all stations under the budget or if there's no more actions to take
        terminated = (
            self.stations_placed >= self.nr_stations or np.sum(self.action_mask) == 0
        )

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info
