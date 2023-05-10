from abc import ABC, abstractmethod

import numpy as np

from motndp.city import City

class Constraints(ABC):
    """Limits the space of the agent's next action, based on different spatial/directional constraints.."""

    def __init__(self):
        super(Constraints, self).__init__()

    @abstractmethod
    def mask_actions(self, current_location, possible_next_locations, visited_locations):
        """Applies different routing constraints and returns a binary mask of allowed actions, of size (n_actions,).
        The selection is based on the previous & current locations (direction).

        Args:
            current_location (numpy.ndarray): current location of the agent, of size (2,).
            possible_next_locations (numpy.ndarray): possible next locations of the agent, of size (n_actions, 2).
            visited_locations (numpy.ndarray): previously visited locations of the agent, of size (n_visited, 2).

        Returns:
        numpy.ndarray: mask of size (n_actions,) with 1s for allowed actions and 0s for disallowed actions.
        """
        pass


class BasicConstraints(Constraints):
    """Do not move outside of the grid, do not go to previously visited locations."""

    def __init__(self, city: City):
        super(BasicConstraints, self).__init__()

        self.city = city


    def mask_actions(self, current_location, possible_next_locations, visited_locations):
        action_mask = np.all(possible_next_locations >= 0, axis=1) &  \
                            (possible_next_locations[:, 0] < self.city.grid_x_size) & \
                            (possible_next_locations[:, 1] < self.city.grid_y_size) & \
                            ~(possible_next_locations[:, None] == visited_locations).all(2).any(1) # mask out visited cells -> https://stackoverflow.com/questions/54828039/how-to-match-pairs-of-values-contained-in-two-numpy-arrays
                            # np.all(~np.isin(possible_next_locations, visited_locations), axis=1) 
        
        return action_mask.astype(np.int8)

class MetroConstraints(Constraints):
    """Do not move outside of the grid, do not go to previously visited locations.
    Also, move in a way that resembles a metro network (mostly straight line), without meandering and circular routes.
    More information on these constraints on Wei et al. 2020 paper: City Metro Network Expansion with Reinforcement Learning: https://www.kdd.org/kdd2020/accepted-papers/view/city-metro-network-expansion-with-reinforcement-learning
    """

    def __init__(self, city: City):
        super(MetroConstraints, self).__init__()

        self.city = city


    def mask_actions(self, current_location, possible_next_locations, visited_locations):
        # Dont allow actions that go outside of the grid or to previously visited locations
        action_mask = np.all(possible_next_locations >= 0, axis=1) &  \
                            (possible_next_locations[:, 0] < self.city.grid_x_size) & \
                            (possible_next_locations[:, 1] < self.city.grid_y_size) & \
                            ~(possible_next_locations[:, None] == visited_locations).all(2).any(1) # mask out visited cells -> https://stackoverflow.com/questions/54828039/how-to-match-pairs-of-values-contained-in-two-numpy-arrays
                            # np.all(~np.isin(possible_next_locations, visited_locations), axis=1) 
        action_mask = action_mask.astype(np.int8)

        if len(visited_locations) < 2:
            return action_mask
        
        # get the direction of the agent (based on the last two visited locations)
        direction = current_location - visited_locations[-2]
        direction[direction > 0] = 1
        direction[direction < 0] = -1

        # mask out actions that are not in the direction of the agent, to prevent meandering and circular routes
        # upwards movement
        if np.array_equal(direction, (-1, 0)):
            action_mask[[3, 4, 5]] = 0
        # downwards movement
        elif np.array_equal(direction, (1, 0)):
            action_mask[[0, 1, 7]] = 0
        # left movement
        elif np.array_equal(direction, (0, -1)):
            action_mask[[1, 2, 3]] = 0
        # right movement
        elif np.array_equal(direction, (0, 1)):
            action_mask[[5, 6, 7]] = 0
        # up-left movement
        elif np.array_equal(direction, (-1, -1)):
            action_mask[[1, 2, 3, 4, 5]] = 0
        # up-right movement
        elif np.array_equal(direction, (-1, 1)):
            action_mask[[3, 4, 5, 6, 7]] = 0
        # down-left movement
        elif np.array_equal(direction, (1, -1)):
            action_mask[[0, 1, 2, 3, 7]] = 0
        # down-right movement
        elif np.array_equal(direction, (1, 1)):
            action_mask[[0, 1, 5, 6, 7]] = 0

        return action_mask
