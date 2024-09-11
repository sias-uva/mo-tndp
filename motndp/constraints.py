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

    dir_up = (-1, 0)
    dir_down = (1, 0)
    dir_left = (0, -1)
    dir_right = (0, 1)
    dir_upleft = (-1, -1)
    dir_upright = (-1, 1)
    dir_downleft = (1, -1)
    dir_downright = (1, 1)

    def __init__(self, city: City):
        super(MetroConstraints, self).__init__()

        self.city = city

    def get_direction(self, current_location, visited_locations):
        direction = current_location - visited_locations[-2]
        direction[direction > 0] = 1
        direction[direction < 0] = -1
        
        for loc in visited_locations[:-3]:
            dir = current_location - loc
            dir[dir > 0] = 1
            dir[dir < 0] = -1

            if np.array_equal(dir, self.dir_upleft):
                direction = self.dir_upleft
            elif np.array_equal(dir, self.dir_upright):
                direction = self.dir_upright
            elif np.array_equal(dir, self.dir_downleft):
                direction = self.dir_downleft
            elif np.array_equal(dir, self.dir_downright):
                direction = self.dir_downright
                
        return direction

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
        
        # get the direction of the agent (based on the last visited locations)
        direction = self.get_direction(current_location, visited_locations)

        if len(visited_locations) > 2:
            # We want to detect indirect diagonal movement, e.g. up then left, or left then up and so on.
            # To do this, we check the direction of the last two steps.
            two_step_dir = current_location - visited_locations[-3]
            two_step_dir[two_step_dir > 0] = 1
            two_step_dir[two_step_dir < 0] = -1

            if np.array_equal(two_step_dir, self.dir_upleft):
                direction = self.dir_upleft
            elif np.array_equal(two_step_dir, self.dir_upright):
                direction = self.dir_upright
            elif np.array_equal(two_step_dir, self.dir_downleft):
                direction = self.dir_downleft
            elif np.array_equal(two_step_dir, self.dir_downright):
                direction = self.dir_downright

        # mask out actions that are not in the direction of the agent, to prevent meandering and circular routes
        # up-left movement
        if (np.array_equal(direction, self.dir_upleft)):
            action_mask[[1, 2, 3, 4, 5]] = 0
        # up-right movement
        elif (np.array_equal(direction, self.dir_upright)):
            action_mask[[3, 4, 5, 6, 7]] = 0
        # down-left movement
        elif (np.array_equal(direction, self.dir_downleft)):
            action_mask[[0, 1, 2, 3, 7]] = 0
        # down-right movement
        elif (np.array_equal(direction, self.dir_downright)):
            action_mask[[0, 1, 5, 6, 7]] = 0
        # upwards movement
        elif np.array_equal(direction, self.dir_up):
            action_mask[[3, 4, 5]] = 0
        # downwards movement
        elif np.array_equal(direction, self.dir_down):
            action_mask[[0, 1, 7]] = 0
        # left movement
        elif np.array_equal(direction, self.dir_left):
            action_mask[[1, 2, 3]] = 0
        # right movement
        elif np.array_equal(direction, self.dir_right):
            action_mask[[5, 6, 7]] = 0

        return action_mask
