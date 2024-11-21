# Multi-Objective Transport Network Design Problem
A Gymnasium environment to design public transport networks, by satisfying the OD-flows of multiple socio-economic groups (objectives). 

![animation of transport network designer agent](/resources/motndp.gif "MOTNDP")


## Description
The Multi-Objective Transport Network Design Problem (MOTNDP) is a combinatorial optimization problem that involves designing the optimal transport line in a city to meet travel demand between various locations.

The MOTNDP environment is a highly modular environment that allows for the creation of different types of transport network design problems.
In this environemnt, an agent is tasked with placing stations within a city represented as a grid. Episodes start in a cell, and the agent can move between adjacent cells to place new stations.
The environment is based on the City object, which contains the city grid and group definitions for each cell, and the Constraints object, which contains the constraints on agent's movement in the grid and can be used to mask actions, creating different types of transport lines, such as straight lines, curves, or loops.
Each grid cell represents a location associated with a specific group. The reward reflects the total OD demand satisfied for each group, either in absolute or percentage terms.

## Observation Space
The observation space is flexible and can be set with the `state_representation` argument. It can be:
- 'grid_coordinates': the agent's location in grid coordinates (x, y). For 'grid_coordinates', the observation space is a MultiDiscrete space with two dimensions: [grid_x_size, grid_y_size].
- 'grid_index': the agent's location as a scalar index (0,0) -> 0, (0,1) -> 1, ..... For 'grid_index', the observation space is a Discrete space with the size of the grid.
- 'one_hot': a one-hot vector of the agent's location in the grid. For 'one_hot', the observation space is a Box space with the shape of the grid.

## Action Space
The actions is a discrete space where:
- 0: walk up
- 1: walk up-right
- 2: walk right
- 3: walk down-right
- 4: walk down
- 5: walk down-left
- 6: walk left
- 7: walk up-left

At every step, the agent can move to one of the 8 adjacent cells, as long as the movement is allowed by the action mask.
When an agent moves to a new cell, it places a station in the grid, connecting the previous cell with the new one.

## Reward Space
The reward is a vector of length `nr_groups` (number of groups in the city). The reward reflects the total OD demand satisfied for each group, either in absolute or percentage terms.
The type of reward can be set with the `od_type` argument: 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).

## Starting State
The starting state is the initial location of the agent in the grid. The initial location can be set with the `starting_loc` argument. If not set, the starting location is chosen randomly.

## Episode Termination
An episode terminates when the agent has placed all stations under the budget or when there are no more actions to take.

## Arguments
- city (City): City object that contains the grid and the groups.
- constraints (Constraints): Transport constraints object with the constraints on movement in the grid.
- nr_stations (int): Episode length. Total number of stations to place (each station is an episode step).
- state_representation (str): State representation. Can be 'grid_coordinates' (returns the agent's location in grid coordinates), 'grid_index' (scalar index of grid coordinates) or 'one_hot' (one-hot vector).
- od_type (str): Type of Origin Destination metric. Can be 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).
- chained_reward (bool): If True, each new station will receive an additional reward based not only on the ODs covered between the immediate previous station, but also those before.
- starting_loc (tuple): Set the default starting location of the agent in the grid. If None, the starting location is chosen randomly, or chosen in _reset().
- render_mode (str): if 'human', the environment will render a pygame window with the agent's movement and the covered cells.
