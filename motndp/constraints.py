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
