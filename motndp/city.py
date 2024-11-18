import configparser
import json
import numpy as np
import itertools

def matrix_from_file(path, size_x, size_y):
    """Reads a grid file that is structured as idx1,idx2,weight and creates a matrix representation of it.
    It is used to read ses variables per grid, od matrices, etc.
    Note: non-existing values are assigned 0.

    Args:
        path (string): the path of the file.
        size_x (int): nr of rows in the matrix.
        size_y (int): nr of columns in the matrix.

    Returns:
        np.array: the matrix representation of the file.
    """
    mx = np.zeros((size_x, size_y))
    with open(path, 'r') as f:
        for line in f:
            idx1, idx2, weight = line.rstrip().split(',')
            idx1 = int(idx1)
            idx2 = int(idx2)
            weight = float(weight)

            mx[idx1][idx2] = weight
    return mx

class City(object):
    """The City Grid environment the agent learns from."""

    def grid_to_index(self, grid_coordinates):
        """Converts grid coordinates (x, y) to a single index (x^).

        Args:
            grid_coordinates (np.array): the grid coordinates to be converted to indices: [[x1, y1], [x2, y2], ...]

        Returns:
            np.array: converted vector indices.
        """
        grid_idx = grid_coordinates[:, 0] * self.grid_y_size + grid_coordinates[:, 1]
        return grid_idx

    def index_to_grid(self, grid_idx):
        """Converts a grid index to grid coordinates (x, y).

        Args:
            grid_idx (np.array or int): the grid index to be converted to grid coordinates.

        Returns:
            np.array: converted grid coordinates.
        """

        grid_x = (grid_idx // self.grid_y_size)
        grid_y = (grid_idx % self.grid_y_size)

        if isinstance(grid_idx, np.int64):
            return np.array([grid_x, grid_y])
        
        return np.column_stack((grid_x, grid_y))
    
    def one_hot_encode(self, grid_idx):
        """One hot encodes a grid index.

        Args:
            grid_idx (np.array): the grid index to be one hot encoded.

        Returns:
            np.array: one hot encoded vector.
        """
        one_hot = np.zeros(self.grid_size)
        one_hot[grid_idx] = 1
        return one_hot
        
    def process_lines(self, lines):
        """Creates a list of tensors for each line, from given grid indices. Used to create line/segment representations of metro lines.

        Args:
            lines (list): list of list of stations (grid indices).

        Returns:
            list: list of tensors, one for each line. Each tensor is a series of vector indices.
        """
        processed_lines = []
        for l in lines:
            l = np.array(l).astype(np.int64)
            # Convert grid indices (x,y) to vector indices (x^)
            l = self.grid_to_index(l)
            processed_lines.append(l)
        return processed_lines
    
    def agg_od_mx(self):
        """Calculates the aggregate origin-destination flow matrix for each grid cell in the city (inflows + outflows). Can be used for visualization purposes.

        Returns:
            np.array: aggregate od by grid
        """
        agg_od_g = np.zeros((self.grid_x_size, self.grid_y_size))
        for i in range(self.grid_size):
            g = self.index_to_grid(np.array([i]))[0]
            agg_od_g[g[0], g[1]] = np.sum(self.od_mx[i])

        return agg_od_g
    def connections_with_existing_lines(self, segment):
        # Satisfied OD pairs from the new segment, by considering connections to existing lines.
        # For each segment, we look for intersections to the existing lines (full, not only grids with stations).
        # If intersection is found, we add the extra satisfied ODs
        connected_stations = []
        for i, line_full in enumerate(self.existing_lines_full):
            line = self.existing_lines[i]
            intersection_full_line = np.transpose(((segment - line_full) == 0).nonzero())
            if intersection_full_line.shape[0] != 0:
                intersection_station_line = np.transpose(((segment - line) == 0).nonzero())

                # We filter the line grids based on the intersection between the new line and the sations of old lines.
                line_mask = np.ones(line.size, dtype=bool)
                line_mask[intersection_station_line[:, 0]] = False
                line_connections = line[line_mask]
                
                connected_stations.extend(line_connections.flatten().tolist())
                
        return connected_stations

    
    def satisfied_od_mask(self, segment, cells_to_chain=None, connected_cells=None, segments_to_ignore=None, return_od_pairs=False):
        """Computes a boolean mask of the satisfied OD flows of a given segment.

        Args:
            segment (np.array): vector indices resembling a segment.
            cells_to_chain (np.array): vector indices of cells that are connected to the segment. If not None, the OD flows between these cells and the new added cell will be summed to the reward.
            connected_cells (set): vector indices of cells that are connected to the segment, from the existing lines. If not None, the OD flows between these cells and the line cells will be summed to the reward.
            segments_to_ignore (list): list of segments to ignore when calculating the OD mask. Used to ignore the OD flows of the segments that are already placed.
            return_od_pairs (boolean): if set to true, the function will return the satisfied OD pairs of the given segment.

        Returns:
            np.array: a boolean mask of the satisfied OD flows of the given segment.
            np.array: the satisfied OD pairs of the given segment.
        """
        # Satisfied OD pairs from the new segment. Note that the segment can be multiple consecutive cells (more than 2).
        sat_od_pairs = np.array(list(itertools.combinations(segment.flatten(), 2)))

        # If there are previous cells to chain, add the OD pairs of the new segment to these cells.
        if cells_to_chain is not None and len(cells_to_chain) > 0:
            # Only chain to cells that are not in the segment, but previously placed stations
            cells_to_chain = cells_to_chain[cells_to_chain != segment[0]]
            sat_od_pairs = np.concatenate((sat_od_pairs, np.column_stack((cells_to_chain, np.full(len(cells_to_chain), segment[1])))))

        if connected_cells is not None and len(connected_cells) > 0:
            connected_cells = np.asarray(list(connected_cells))  # or np.array() if the list is not iterable
            filtered_segment = np.setdiff1d(segment, np.concatenate(self.existing_lines), assume_unique=True)
            # Get the cells of the new line to be connected to the existing line.
            new_line_cells = np.hstack((filtered_segment, cells_to_chain))
            new_sat_od_pairs = np.array(np.stack(np.meshgrid(new_line_cells, connected_cells), axis=-1).reshape(-1, 2))
            sat_od_pairs = np.vstack((sat_od_pairs, new_sat_od_pairs))
        
        if segments_to_ignore is not None and len(segments_to_ignore) > 0:
            ignore_set = set(map(tuple, segments_to_ignore))  # Convert to set for fast lookup
            sat_od_pairs = np.array([pair for pair in sat_od_pairs if tuple(pair) not in ignore_set])

        od_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        od_mask[sat_od_pairs[:, 0], sat_od_pairs[:, 1]] = 1
        
        if return_od_pairs:
            return od_mask, sat_od_pairs
        
        return od_mask

    def __init__(self, env_path, groups_file=None, ignore_existing_lines=False):
        """Initialise city environment.

        Args:
            env_path (Path): path to the folder that contains the needed initialisation files of the environment.
            groups_file (str): file within envirnoment folder that contains group membership for each grid square.
            ignore_existing_lines (boolean): if set to true, the environment will not load the current existing lines of the environment (check config.txt).
        """
        super(City, self).__init__()

        # read configuration file that contains basic parameters for the City.
        config = configparser.ConfigParser()
        config.read(env_path / 'config.txt')
        assert 'config' in config, "Config file not found or not in the correct format."

        # size of the grid
        self.grid_x_size = config.getint('config', 'grid_x_size')
        self.grid_y_size = config.getint('config', 'grid_y_size')
        self.grid_size = self.grid_x_size * self.grid_y_size

        # Create a (1, grid_size) grid where each cell is represented by its [x,y] indices.
        # Used to calculate distances from each grid cell, etc.
        mesh = np.meshgrid(np.arange(0, self.grid_x_size), np.arange(0, self.grid_y_size))

        # Build the normalized OD and SES matrices.
        self.od_mx = matrix_from_file(env_path / 'od.txt', self.grid_size, self.grid_size)
        self.od_mx = self.od_mx / np.max(self.od_mx)
        try:
            self.price_mx = matrix_from_file(env_path / 'average_house_price_gid.txt', self.grid_x_size, self.grid_y_size)
        except FileNotFoundError:
            print('Price matrix not available.')
            
        self.ignore_existing_lines = ignore_existing_lines
        # Read existing metro lines of the environment.
        if not ignore_existing_lines and config.has_option('config', 'existing_lines'):
            # json is used to load lists from ConfigParser as there is no built in way to do it.
            existing_lines = self.process_lines(json.loads(config.get('config', 'existing_lines')))
            # Full lines contains the lines + the squares between consecutive stations e.g. if line is (0,0)-(0,2)-(2,2) then full line also includes (0,1), (1,2).
            # These are important for when calculating connections between generated & lines and existing lines.
            existing_lines_full = self.process_lines(json.loads(config.get('config', 'existing_lines_full')))

            # Create line tensors
            self.existing_lines = [l.reshape(len(l), 1) for l in existing_lines]
            self.existing_lines_full = [l.reshape(len(l), 1) for l in existing_lines_full]
            
            # Exclude satisfied OD pairs from the existing lines.
            exclude_line_pairs = np.empty((0, 2), dtype=np.int64)
            for l in existing_lines:
                pair1 = np.array(list(itertools.combinations(l, 2)))
                pair2 = np.array(list(itertools.combinations(l[::-1], 2)))
                
                exclude_line_pairs = np.concatenate((exclude_line_pairs, pair1, pair2))
            self.od_mx[exclude_line_pairs[:, 0], exclude_line_pairs[:, 1]] = 0
        else:
            self.existing_lines = []
            self.existing_lines_full = []
            
        # Apply excluded OD segments to the od_mx. E.g. segments very close to the current lines that we want to set OD to 0.
        if config.has_option('config', 'excluded_od_segments'):
            exclude_segments = self.process_lines(json.loads(config.get('config', 'excluded_od_segments')))
            if len(exclude_segments) > 0:
                exclude_pairs = np.empty((0, 2), dtype=np.int64)
                for s in exclude_segments:
                    # Create two-way combinations of each segment.
                    # e.g. segment: 1-2-3-4, pairs: 1-2, 2-1, 1-3, 3-1, 1-4, 4-1, ... etc
                    
                    pair1 = np.array(list(itertools.combinations(s, 2)))
                    pair2 = np.array(list(itertools.combinations(s[::-1], 2)))

                    exclude_pairs = np.concatenate((exclude_pairs, pair1, pair2))
            
                self.od_mx[exclude_pairs[:, 0], exclude_pairs[:, 1]] = 0
        
        # If there are group memberships of each grid square, then create an OD matrix for each group.
        self.group_od_mx = None # initialize it so we can check later on if it has any value
        if groups_file:
            self.grid_groups = matrix_from_file(env_path / groups_file, self.grid_x_size, self.grid_y_size)
            # matrix_from_file initializes a tensor with np.zeros - we convert them to nans
            self.grid_groups[self.grid_groups == 0] = float('nan')
            # Get all unique groups
            self.groups = np.unique(self.grid_groups[~np.isnan(self.grid_groups)])
            # Create a group-specific od matrix for each group.
            
            self.group_od_mx = np.zeros((len(self.groups), self.grid_size, self.grid_size))
            for i, g in enumerate(self.groups):
                group_mask = np.zeros(self.od_mx.shape)
                group_squares = self.grid_to_index(np.transpose(np.nonzero(self.grid_groups == g)))
                # Original OD matrix is symmetrical, so group OD matrices should also be symmetrical.
                group_mask[group_squares, :] = 1
                group_mask[:, group_squares] = 1
                self.group_od_mx[i] = group_mask * self.od_mx
                
        # Total sum of OD flows by group, to be used for the reward calculation, but want to calculate only once.
        self.group_od_sum = np.array([g_od.sum() for g_od in self.group_od_mx])

