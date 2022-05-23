"""
    Script to deal with point clouds using spatial data structures.
"""

import numpy as np
from utils.measurers import TimeMeasurer
from utils.viewers import visualize_cloud_splitted
from utils.viewers import visualize_cloud
from utils.viewers import PointCloudViewer
from scipy.sparse import coo_matrix


def calculate_mbb(pointcloud):
    """
    Given a point cloud as a numpy array in format: x y z ... calculate its Minimal Bounding Box

    :param pointcloud: numpy array containing a 3D point cloud
    :return: (xmin, ymin, zmin), (xmax, ymax, zmax) describing the MBB
    """

    # Calculate the minimum coordinate and the maximum one of the given point cloud
    min_coord = np.amin(pointcloud[:, 0:3], axis=0)
    max_coord = np.amax(pointcloud[:, 0:3], axis=0)

    # Return min_coord & max_coord as numpy arrays of (3,) dimensions
    return min_coord, max_coord


def slice_cloud(pointcloud, split_size=1.0, stride=1.0, min_points=512, visualization=False):
    """
    Slice a point cloud regularly in blocks of split_size*split_size*split_size

    :param visualization: to visualize selected blocks during generation or not
    :param min_points: minimum of points to consider a block.
    :param stride: offset of the sliding window.
    :param split_size: size of the cell edge. Point cloud will be regularly divided in sections whose area is
    split_size^2
    :param pointcloud: numpy array containing a point cloud
    :return: list of numpy arrays, each containing all points in a block of size split_size*split_size
    """

    # TODO implement this using random selection of regions.

    # Compute MBB and dimensions of it
    min_coord, max_coord = calculate_mbb(pointcloud)
    mbb_dim = max_coord - min_coord  # np array with [d_x, d_y, d_z]

    # Number of blocks on each direction
    mbb_dim = mbb_dim / split_size

    # Iterate over point cloud & select blocks. Insert them in a list.
    blocks = []  # List of numpy arrays
    x, y, z = min_coord
    xmax, ymax, zmax = max_coord

    # Version 1: Pythonic
    # while x < max_coord[0]:
    #     y = min_coord[1]
    #     while y < max_coord[1]:
    #
    #         block = pointcloud[
    #             (x < pointcloud[:, 0]) & (pointcloud[:, 0] < x + split_size) &
    #             (y < pointcloud[:, 1]) & (pointcloud[:, 1] < y + split_size)
    #         ]
    #
    #         # Append block to the resulting list
    #         if block.shape[0] >= min_points:
    #             block = normalize_block(block, (min_coord, max_coord))
    #             blocks += [block]
    #             print("spatial.slice_cloud(): {} points in this block".format(block.shape[0]))
    #             print("spatial.slice_cloud(): {} blocks generated".format(len(blocks)))
    #
    #         y += stride
    #     x += stride

    # # Version 2: no Pythonic
    # # TODO stride is not being considered to assign blocks (stride = block size by default)
    # MARGIN = 10 # Blocks. Take care of not to access forbidden positions in the array.
    # num_blocks_x = int(np.ceil((xmax - x) / split_size)) + MARGIN  # x is considered here as xmin (of the MBB)
    # num_blocks_y = int(np.ceil((ymax - y) / split_size)) + MARGIN  # y is considered here as xmin (of the MBB)
    #
    # # Create lists depending on the number of blocks in each dimension (as a sparse matrix in C++)
    # blocks_x = []
    # for i in range(num_blocks_x):
    #     blocks_x.append([])
    #     for j in range(num_blocks_y):
    #         blocks_x[i].append([])
    #
    # # Assign the identifier (index in original array) to the corresponding block. Iterate over pointcloud
    # for id_point in range(pointcloud.shape[0]):  # For each point
    #     x_orig = pointcloud[id_point, 0]
    #     y_orig = pointcloud[id_point, 1]
    #
    #     id_block_x = int(np.floor((x_orig - x) / split_size))  # x is considered here as xmin (of the MBB)
    #     id_block_y = int(np.floor((y_orig - y) / split_size))  # y is considered here as ymin (of the MBB)
    #
    #     blocks_x[id_block_x][id_block_y].append(id_point)
    #
    #     if (id_point + 1) % 100000 == 0:
    #         print("Assigning points to blocks. Processed {} points out of {}".format(id_point+1, pointcloud.shape[0]))

    ##############################################
    # Version 3: use stride when generating blocks
    ##############################################

    assert stride <= split_size, 'Stride must be lower or equal than split_size'
    assert int(str(split_size / stride).split('.')[-1]) == 0, 'Stride must divide split_size'

    blocks_x = []
    miniblocks = []
    xmin, ymin, zmin = min_coord  # Z is not going to be used

    # Initialize x as xmin, so each miniblock will be assigned with a identifier across X and Y axis
    # The next loop generates a regular mesh for all points in the cloud, taking as minimum block size the stride
    x = xmin
    i = 0
    while x < xmax + split_size:
        y = ymin
        miniblocks.append([])
        blocks_x.append([])
        while y < ymax + split_size:
            miniblocks[i].append([])
            blocks_x[i].append([])
            y += stride

        x += stride
        i += 1

    # Assign points to miniblocks
    for id_point in range(pointcloud.shape[0]):  # For each point
        x_orig = pointcloud[id_point, 0]
        y_orig = pointcloud[id_point, 1]

        id_block_x = int(np.floor((x_orig - xmin) / stride))  # x is considered here as xmin (of the MBB)
        id_block_y = int(np.floor((y_orig - ymin) / stride))  # y is considered here as ymin (of the MBB)

        miniblocks[id_block_x][id_block_y].append(id_point)

        if (id_point + 1) % 100000 == 0:
            print("Assigning points to miniblocks. Processed {} points out of {}".format(id_point+1, pointcloud.shape[0]))

    # Once all points have been assigned to miniblocks, assign miniblocks to real blocks. Each block will contain
    # More than one miniblock.
    # Calculate miniblocks_per_block
    num_miniblocks_dim = int(split_size / stride)  # So, in a block will be contained num_miniblocks_dim^2 miniblocks.
    # Loop over each axis and group several miniblocks in a general purpose block
    print('Assigning miniblocks to blocks')
    x = xmin
    count = 0
    while x < xmax:
        y = ymin
        while y < ymax:
            base_miniblock_x, base_miniblock_y = int((x - xmin) / stride), int((y - ymin) / stride)  # (x, y)

            for i in range(num_miniblocks_dim):
                for j in range(num_miniblocks_dim):
                    blocks_x[base_miniblock_x][base_miniblock_y] += miniblocks[base_miniblock_x + i][base_miniblock_y + j]

            count += 1

            if (count + 1) % 100 == 0:
                print('Assigning miniblocks to blocks. Processed {} blocks'.format(count))

            y += stride
        x += stride


    print("All points have been assigned. Starting to extract blocks...")

    # Extract points contained in each block and append each one to the end of the list
    average_block_size = []
    for i in range(len(blocks_x)):
        for j in range(len(blocks_x[i])):
            # Generate a block if contains a number of points equal or greater to min_points
            block_size = len(blocks_x[i][j])
            if block_size > min_points:
                # Get identifiers (index) of the points contained in the block
                idx_points = np.array(blocks_x[i][j])

                # Generate block from the obtained indexes
                block = pointcloud[idx_points, :]

                # Normalize block
                block = normalize_block(block, (min_coord, max_coord))

                # Append block to the end of the list
                blocks += [block]
                print("Generating blocks. {} block(s) have been already generated".format(len(blocks)))

            if block_size > 0:
                average_block_size += [block_size]

    # Calculate statistics of block size
    avg_size = np.mean(average_block_size)
    std_dev_size = np.std(average_block_size)
    max_size = np.max(average_block_size)
    min_size = np.min(average_block_size)

    print("Block size: average {} standard deviation {} maximum {} minimum {}".format(avg_size, std_dev_size, max_size, min_size))

    return blocks


def bounding_volume(pointcloud):
    """
    Given a 3D point cloud as input, calculate the bounding volume in cubic meters.

    :param pointcloud: numpy array containing a point cloud
    :return: bounding volume of the point cloud in cubic meters
    """

    # Calculate MBB
    min_coord, max_coord = calculate_mbb(pointcloud)
    mbb_dims = max_coord - min_coord

    # Multiply all components in array to obtain volume
    volume = np.prod(mbb_dims)

    return volume

def normalize_block(block, bounding_box):
    """
    Given a block containing a region of the point cloud and its bounding box, compute the normalized coordinates in
    the scene and append then at the end of the array.

    :param block: numpy array containing a portion of the original point cloud.
    :param bounding_box: MBB of the original point cloud.
    :return: block with normalized x y z coordinates.
    """

    min_coord, max_coord = bounding_box

    # Normalize coordinates
    xyz_norm = (block[:, 0:3] - min_coord) / (max_coord - min_coord)

    # Adjust input block to be concatenated.
    xyz_int_rgb = block[:, :-1]
    labels = block[:, -1]
    labels = labels[:, np.newaxis]

    # Generate a new block containing: x y z intensity r g b x' y' z' label
    new_block = np.concatenate((xyz_int_rgb, xyz_norm, labels), axis=1)

    return new_block


class PointExtractor:
    def __init__(self):
        # Sparse matrix to hold point-block relationships
        self.point_block = None

    def extract_points(self, block, block_index, potentials, num_points, potential_limit):
        """
        Receive as input a block with N points. Order all points in the block by their potentials, from lowest to greatest,
        and select the num_points with the lowest potential. If potential is already equal or greater than potential_limit,
        return an empty array of points and process another block

        :param potential_limit: maximum potential to reach out within the iterations. When this value is reached, do not
        return any points, but None
        :param num_points: number of points to extract from the block
        :param block_index: index of the block in the general blocks array
        :param block: numpy array of N points with d dimensions plus IdX of each one. Indexes of the points are in last dim.
        :param potentials: potentials of the points in the block, to select the points with the lowest potential.
        :return: numpy array of [num_points, dimensions] if block is procesable, empty array otherwise.
        """

        # Get idx of all points in the block
        idx = block[:, -1]

        # Get potentials of the points
        block_potentials = potentials[idx]

        # Sort array by potentials
        block_potentials = block_potentials[np.argsort(block_potentials[:, 1])]

        # Check if all points have already been processed
        if block_potentials[0, 0] >= potential_limit:
            return None

        # Check number of points. If less than requested, repeat idx until have num_points
        if block_potentials.shape[0] < num_points:
            selected_idx = block_potentials[:, 0]

            # Update potentials array
            potentials[selected_idx, 1] += 0.1  # On each interaction

            # Repeat until obtain num_points samples
            sample = np.random.choice(selected_idx.shape[0], num_points, replace=True)
            selected_idx = selected_idx[sample]
        else:
            # Select first num_points points of the block
            selected_idx = block_potentials[:num_points, 0]

            # Update potentials array
            potentials[selected_idx, 1] += 0.1  # On each interaction

        # TODO extract points of the blockkkk
        """
        The indexes of the points obtained until now represent the index of each point according to its position
        in the main array. This is not suitable for this case, as points have different indexes in the blocks.
        """
        # Retrieve all points in a block
        idx_points_block = self.point_block.getcol(block_index)

        # Retrieve indexes of the selected points
        values = idx_points_block.getrow(selected_idx)

        # Generate array of the indexes of the points in the block
        idx_points_block = values.data

        # Extract points from the block
        points = block[idx_points_block]

        return points


    def associate_points_blocks(self, points, blocks):
        """
        Given a list of blocks, each containing an arbitrary set of points, associate real indexes with "imaginary" ones.
        That is, points will have different indexes in their blocks. Create a sparse matrix with points in rows and blocks
        in columns. So, when querying this matrix, we get the index of the point i in the block j.

        :param points: numpy array of points
        :param blocks: list of numpy arrays, each containing one or more points.
        :return: sparse matrix with points as rows and blocks as columns. So, (i, j) gives index of the point i in the
        """

        rows = []  # i
        cols = []  # j
        values = []  # value in (i, j)

        # Starting from block j, use ijv format https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        for j, block in enumerate(blocks):
            idx_block = block[:, -1]
            for value, i in idx_block:
                rows.append(i)
                cols.append(j)
                values.append(value)

        self.point_block = coo_matrix((values, (rows, cols)), shape=(points.shape[0], len(blocks)))