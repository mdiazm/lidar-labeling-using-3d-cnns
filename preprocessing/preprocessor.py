import numpy as np
import os
from sklearn.model_selection import KFold
import h5py
from laspy.file import File as LASFile
import time

# Custom imports
from preprocessing.spatial import *
from utils.viewers import *
from utils.download_data import *


class Processor:
    """
        Base class to prepare a dataset for training/testing.
    """

    def __init__(self, dset_name, base_path, training_items='train_items'):
        """
        Constructor of a basic data processor

        :param dset_name: The name (type) of the dataset: Semantic3D, LiDAR, etc
        :param base_path: 'data' folder path.
        """
        self.dset_name = dset_name
        self.base_path = os.path.join(base_path,
                                      dset_name)  # Base path of each dataset. Where point clouds & label files are stored
        self.train_dir = os.path.join(self.base_path, 'train')
        self.test_dir = os.path.join(self.base_path, 'test')
        self.inference_dir = os.path.join(self.base_path, 'inference')

        # Once point clouds has been processed for training
        self.sampled_clouds = os.path.join(self.train_dir, 'sampled')
        self.training_items = os.path.join(self.train_dir,
                                           training_items)  # Where the batches are going to be stored (each mini-batch has 4096 labeled points from a semantized point cloud)

        # Store data to classify
        self.classificable_cloud = None

        # Create previous dirs if they not exist
        if not os.path.isdir(self.sampled_clouds):
            os.mkdir(self.sampled_clouds)

        if not os.path.isdir(self.training_items):
            os.mkdir(self.training_items)

    def sample(self, pointcloud, method='random', retain=0.5, num_points=None):
        """
        Subsample a given point cloud using method specified as parameter. This methods works as a wrapper for
        the more specific methods.

        :param num_points: if defined, select exactly the number of points specified on that parameter.
        :param pointcloud: numpy array with the point cloud
        :param method: method of sampling: random, fps, regular, # todo later
        :param retain: in case of method==random, amount of points to retain in the subsampled point cloud.
        :return: subsampled pointcloud
        """

        print("{}: sampling point cloud with method... {}".format(self.dset_name, method))

        if method == 'random':
            subsampled_pointcloud = self.random_sample(pointcloud, retain, num_points)
        elif method == None:
            subsampled_pointcloud = pointcloud
        else:
            # TODO add more algorithms later
            pass

        return subsampled_pointcloud

    def random_sample(self, pointcloud, percentage=0.2, num_points=None):
        """
        Perform random sampling of points given a point cloud and the label of each point.

        :param num_points: if defined, select exactly the number of points specified on that parameter.
        :param pointcloud: numpy array where each row of the matrix is a N-dimensional point and its label
        :param percentage: [0, 1] amount of points to be randomly selected. The higher percentage is,
        the more points are going to be taken. Ideally, 1 means that all points are selected. Default: 20% of points

        :return: subsampled point cloud in the format of: points (np array), labels (np array).
        """

        pointcloud_size = pointcloud.shape[0]
        points_to_be_selected = int(percentage * pointcloud_size) if num_points is None else num_points

        if pointcloud_size == points_to_be_selected:
            return pointcloud
        elif points_to_be_selected > pointcloud_size:
            # Select with replacement
            sample_indices = np.random.choice(pointcloud_size, points_to_be_selected, replace=True)
        else:
            sample_indices = np.random.choice(pointcloud_size, points_to_be_selected, replace=False)

        # Extract points and their respective labels from the original arrays
        sample_pointcloud = pointcloud[sample_indices]

        return sample_pointcloud

    def load_point_cloud(self, name, train=True):
        """
        Abstract method. Depending on the format, each dataset loads point clouds in a way or another.

        :param name: name of the point cloud file.

        :return: original of points (np array), labels (np array).
        """

        print("{}: loading point cloud... {}".format(self.dset_name, name))
        pass

    def subsample_dataset(self, method='random'):
        """
        Abstract method. Redefine it in subclasses

        :param method:
        :param train: load train partition or test
        :return:
        """

        print("Subsampling dataset: {}".format(self.dset_name))
        pass

    def load_sampled_point_cloud(self, name, format="txt"):
        """
        Load a point cloud using np.loadtxt(). Those point clouds are stored in sampled directory

        :param name: name of the point cloud
        :return: numpy array containing the point cloud
        """

        # Set format of the point cloud
        name = name[:-3] + format

        print("{}: loading sampled point loud... {}".format(self.dset_name, name))

        # Path where the point cloud is stored
        load_path = os.path.join(self.sampled_clouds, name)

        # Load point cloud
        if format == "txt":
            pointcloud = np.loadtxt(load_path, delimiter=" ")
        elif format == "npy":
            pointcloud = np.load(load_path)

        return pointcloud

    def write_point_cloud(self, pointcloud, output_name, sampled=True, format="txt"):
        """
        Write a point cloud in the specified format. If sampled=False, output pointcloud will be written in the basepath
        of the corresponding dataset. Otherwise (sampled=True), received point cloud will be considered as a sampled
        point cloud and will be written in 'sampled_clouds' directory.

        :param pointcloud: NumPy array containing points and labels: x y z ... label
        :param output_name: the output name of the point cloud
        :param sampled: to write point cloud in sampled_clouds or not.
        :param format: format to store the point cloud. Default: plain text file
        """

        if sampled:
            print("{}: saving point cloud in sampled directory "
                  "with name... {}".format(self.dset_name, output_name))
        else:
            print("{}: saving point cloud in root directory "
                  "with name... {}".format(self.dset_name, output_name))

        if format == "txt":
            save_path = os.path.join(
                self.sampled_clouds if sampled else self.base_path,
                output_name + ".txt"
            )

            # Load point cloud using numpy.savetxt method.
            np.savetxt(save_path, pointcloud, delimiter=" ")
        elif format == "npy":
            save_path = os.path.join(
                self.sampled_clouds if sampled else self.base_path,
                output_name + ".npy"
            )

            # Load point cloud using numpy.savetxt method.
            np.save(save_path, pointcloud)
        else:
            # TODO Consider different formats later.
            pass

    def prepare_training_dataset(self, split_size=1.0, stride=1.0, num_points=4096, min_points=512, k=1,
                                 visualization=False):
        """
        Prepare training/validation sets using k-fold cross validation.

        :param k: number of partitions to generate using k-fold cross-validation
            :param stride: offset of the blocks. If stride == split_size, then resulting blocks are not overlapping.
        :param split_size: size of the partitions in the point cloud (regularly divided)
        :param num_points: number of points to extract from each block
        :param visualization: generated blocks are visualized during generation.
        :param min_points: minimum amount of points to consider a block
        """

        # Generate batches
        batches = self.generate_batches(split_size, stride, num_points, min_points, visualization=visualization)

        # Check if there is enough data to create the training and test datasets.
        if len(batches) == 0:
            print(
                "{}: not enough blocks to split into training/validation datasets using k-fold cross validation".format(
                    self.dset_name))
            exit(0)

        print("{}: partitioning batches into training/validation samples using k-fold with K={}".format(self.dset_name,
                                                                                                        k))
        # Partition batches according to parameter
        kfold = KFold(n_splits=k)
        for i, (train_index, validation_index) in enumerate(kfold.split(batches)):
            # Extract training samples and validation ones as following
            training_data = batches[train_index, :]
            validation_data = batches[validation_index, :]

            # Store both partitions in disk
            self.save_training_partition(training_data, partition=i)
            self.save_validation_partition(validation_data, partition=i)
            print("{}: generated and saved partition {}".format(self.dset_name, i))

    def generate_blocks(self, pointcloud, split_size=1.0, stride=1.0, num_points=4096, min_points=512,
                        visualization=False):
        """
        Generate blocks of split_size*split_size*split_size of the given point cloud and extract exactly 4096
        points of each block. The sampling algorithm can be changed: random, FPS, ...

        :param visualization: to visualize blocks during generation or not
        :param stride: offset of the blocks. If stride == split_size, then resulting blocks are not overlapping.
        :param pointcloud: numpy array containing a 3D point cloud: x y z ...
        :param split_size: size of the slice (block)
        :param num_points: number of points to be selected from each block
        :param min_points: minimum amount of points to consider a block
        :return: numpy array of dimensions [Nblocks, num_points, point_dim]
        """

        print("{}: slicing point cloud into blocks of {} points...".format(self.dset_name, num_points))

        # Call to function defined in preprocessing/spatial.py
        blocks = slice_cloud(pointcloud, split_size=split_size, stride=stride, min_points=min_points,
                             visualization=visualization)

        # Transform each block by selecting num_points of it
        blocks = map(lambda cloud: self.sample(cloud, 'random', 1.0, num_points), blocks)
        blocks = list(blocks)

        return blocks

    def generate_batches(self, split_size=1.0, stride=1.0, num_points=4096, min_points=512, visualization=False):
        """
        Generate a bunch of files each contanining num_batches of num_points.

        :param visualization: to visualize blocks during generation or not
        :param stride: offset of the blocks. If stride == split_size, then resulting blocks are not overlapping.
        :param split_size: size of the partitions in the point cloud (regularly divided)
        :param num_points: number of points to extract from each block
        :param min_points: minimum amount of points to consider a block
        :return: anything
        """

        blocks = []

        # Load each sampled cloud and generate blocks of it
        clouds = os.listdir(self.sampled_clouds)
        for i, cloud in enumerate(clouds):
            pointcloud = self.load_sampled_point_cloud(cloud, format="npy")
            # pointcloud = pointcloud[:4000000]
            # visualize_cloud(pointcloud)
            print("{}: generating blocks of {} points for {} point cloud".format(self.dset_name, num_points, cloud))
            blocks += self.generate_blocks(pointcloud, split_size, stride, num_points, min_points, visualization)

            print("{}: {} out of {} point clouds processed".format(self.dset_name, i + 1, len(clouds)))

        print("{}: block generation finished. Total number of blocks generated: {}".format(self.dset_name, len(blocks)))

        # Once that blocks of all points clouds have been generated, store them as compressed HDF5 file.
        # Concatenate all blocks to obtain N x num_points x M numpy array
        batches = []
        if len(blocks) > 0:
            blocks = map(lambda x: x[np.newaxis, :], blocks)
            blocks = list(blocks)
            batches = np.concatenate(blocks, axis=0)

            # Shuffle batches to avoid bias in partition generation
            idx = np.arange(0, batches.shape[0])
            np.random.shuffle(idx)
            batches = batches[idx, :, :]

        return batches

    def save_training_partition(self, data=None, partition=None):
        """
        Given a numpy array containing blocks from all point clouds, store in HDF5 file.

        :param partition: partition (of k-fold cross validation) to store in disk.
        :param data: numpy array of dimensions (N, 4096, M), where N is the total number of blocks, and M is the
        dimension of each point.
        :return: anything
        """

        # Path where generated file is going to be stored
        data_store_path = os.path.join(self.training_items, "train_{}.hdf5".format(partition))

        # Separate data and their labels
        labels = data[:, :, -1]
        data = data[:, :, :-1]

        # Store each item on separate files.
        if os.path.isfile(data_store_path):
            # Do nothing
            print("{}: training data file yet generated in {}".format(self.dset_name, data_store_path))
            pass
        else:
            # Store file
            file = h5py.File(data_store_path, 'w')
            print("{}: generating training data file in {}".format(self.dset_name, data_store_path))
            dset_object = file.create_dataset("train", data.shape, data=data, compression="gzip", compression_opts=9)
            labels_object = file.create_dataset("labels", labels.shape, data=labels, compression="gzip",
                                                compression_opts=9)
            file.close()

    def save_validation_partition(self, data=None, partition=None):
        """
        Given a numpy array containing blocks from all point clouds, store in HDF5 file.

        :param partition: partition (of k-fold cross validation) to store in disk.
        :param data: numpy array of dimensions (N, 4096, M), where N is the total number of blocks, and M is the
        dimension of each point.
        :return: anything
        """

        # Path where generated file is going to be stored
        data_store_path = os.path.join(self.training_items, "validation_{}.hdf5".format(partition))

        # Separate data and their labels
        labels = data[:, :, -1]
        data = data[:, :, :-1]

        # Store each item on separate files.
        if os.path.isfile(data_store_path):
            # Do nothing
            print("{}: validation data file yet generated in {}".format(self.dset_name, data_store_path))
            pass
        else:
            # Store file
            file = h5py.File(data_store_path, 'w')
            print("{}: generating training data file in {}".format(self.dset_name, data_store_path))
            dset_object = file.create_dataset("validation", data.shape, data=data, compression="gzip",
                                              compression_opts=9)
            labels_object = file.create_dataset("labels", labels.shape, data=labels, compression="gzip",
                                                compression_opts=9)
            file.close()

    def load_data(self):
        """
        Iterates over created training/validation splits and loads each.

        :return: loaded partitions in form of numpy arrays
        """

        # Files in train_items dir
        files = [file for file in os.listdir(self.training_items) if "train" in file]

        # Order files according to partition index
        files = sorted(files, key=lambda x: int(x[6]))

        # Load training and validation partitions iteratively
        for train_file in files:
            # Choose validation file
            validation_file = train_file.replace("train", "validation")

            # Create paths
            train_file_path = os.path.join(self.training_items, train_file)
            validation_file_path = os.path.join(self.training_items, validation_file)

            # Load both files
            train_h5py = h5py.File(train_file_path, mode="r")
            validation_h5py = h5py.File(validation_file_path, mode="r")

            # Load content of the files
            print("{}: loading train/validation partition {}".format(self.dset_name, train_file[6]))
            train_data, train_labels = train_h5py["train"][:], train_h5py["labels"][:]
            validation_data, validation_labels = validation_h5py["validation"][:], validation_h5py["labels"][:]

            # Close both files
            train_h5py.close()
            validation_h5py.close()

            # Yield content
            yield (train_data, train_labels), (validation_data, validation_labels)

    def download(self):
        """
        Decide to download full dataset from website or not.
        """


    def load_partition(self, k):
        """
        Load specific partition and return (train_data, train_labels), (validation_data, validation_labels)

        :param k: the training/validation partition to use
        :return: (train_data, train_labels), (validation_data, validation_labels)
        """

        # Training and validation files
        train_file = "train_{}.hdf5".format(k)
        validation_file = "validation_{}.hdf5".format(k)

        # Create paths
        train_file_path = os.path.join(self.training_items, train_file)
        validation_file_path = os.path.join(self.training_items, validation_file)

        # Load both files
        train_h5py = h5py.File(train_file_path, mode="r")
        validation_h5py = h5py.File(validation_file_path, mode="r")

        # Load content of the files
        print("{}: loading train/validation partition {}".format(self.dset_name, train_file[6]))
        train_data, train_labels = train_h5py["train"][:], train_h5py["labels"][:]
        validation_data, validation_labels = validation_h5py["validation"][:], validation_h5py["labels"][:]

        # Close both files
        train_h5py.close()
        validation_h5py.close()

        # Return loaded data
        return (train_data, train_labels), (validation_data, validation_labels)

    def classify_cloud(self, name, split_size=1.0, stride=1.0, num_points=4096, potential_limit=1.0):
        """
        Take some point cloud as input, and classify that using trained neural network

        :param split_size: size (in square meters) of each block
        :param stride: offset of the sliding windows approach
        :param num_points: number of points that are being used for inference
        :param potential_limit: maximum potential for a point to be processed. When reach this potential, discard.
        :param name: file name of the point cloud to classify
        :yield: blocks of points
        """

        # First step, load data
        data = self.load_point_cloud(name=name, train=False)

        # Assign indexes to each point
        idx = np.arange(0, data.shape[0])
        idx = idx[:, np.newaxis]
        data = np.concatenate((data, idx), axis=1)

        # Slice cloud into regions of equal size
        # min_points = 1 because we are on inference time, and so every point must be processed
        blocks = slice_cloud(data, split_size=split_size, stride=stride, min_points=1)

        # Once the cloud have been processed, create array of potentials
        potentials = np.zeros(data.shape[0], dtype=np.float)  # Initialize all potentials on zero
        potentials = np.concatenate((idx, potentials), axis=1)

        # All blocks are procesable
        procesable_blocks = blocks

        # Create point extractor and associate points with blocks
        point_extractor = PointExtractor()
        point_extractor.associate_points_blocks(potentials, blocks)

        # Process blocks until there is no any procesable block (procesable_blocs is empty)
        while True:
            # Blocks to delete: indexes
            blocks_delete = []

            for index, block in enumerate(procesable_blocks):
                extracted_points = point_extractor.extract_points(block, potentials=potentials, num_points=num_points,
                                                  potential_limit=potential_limit)

                """
                If some block has been processed enough times, discard block and delete it from procesable_blocks
                to avoid considering it on next iterations
                """
                if extracted_points == None:
                    blocks_delete.append(index)
                else:
                    yield extracted_points

            # Remove blocks that have been processed enough times
            for index in blocks_delete:
                del procesable_blocks[index]

            # Stop loop when there is no any procesable block
            if len(procesable_blocks) == 0:
                return

    def postprocess_cloud(self, name, result, voting_scheme):
        """
        Receive the results of the classification, and get full semantized point cloud. At point level, apply a voting
        scheme to assign a definitive class to the points.

        Possible solution: when match, add this point to a list of "process later" and after processing every point,
        assign to the point a new "voting scheme" according to its k nearest neighbors.
        :param voting_scheme: voting scheme to use, majority, ...
        :param result: result of the classification. List of tuples (point_index, class), for example
        :param name: name of the point cloud to postprocess

        :return:
        """

    def get_test_clouds(self):
        """
        Get the names of the clouds to apply inference.

        :return: array of string containing cloud names
        """

        return os.listdir(self.inference_dir)

class Semantic3DProcessor(Processor):

    def __init__(self, basepath, download=False, training_items='train_items'):
        super().__init__('Semantic3D', basepath, training_items)

        # In this dataset, labels file are separated from points file
        self.labels_dir = os.path.join(self.train_dir, 'labels')

        # Where to download compressed files
        self.compressed = os.path.join(self.train_dir, 'compressed')

        # labels description
        self.labels = {
            # 0: 'unlabeled point',
            1: 'man-made terrain',
            2: 'natural terrain',
            3: 'high vegetation',
            4: 'low vegetation',
            5: 'high vegetation',
            6: 'hard scape',
            7: 'scanning artifact',
            8: 'car'
        }

        # Try to download if not available
        if download:
            self.download()

    def load_point_cloud(self, name, train=True):
        """
        Method to load a Semantic3D point cloud given its name. This method have to look for both data & labels file,
        we must remember that a great majority of points in this dataset are labeled with 0, and those points must
        not be considered during training phase.

        So, in this method, we have to concatenate each point with its corresponding label and then discard all
        ones that are labeled with 0 label.

        :param name: name of the point cloud.
        :return: numpy array where each line is: x y z intensity r g b label. rgb are normalized to [0, 1] scale, and
        points are translated according to the point with the minimum coordinates (defining the corner of the bounding
        box).
        """

        # Call to method in superclass
        super().load_point_cloud(name, train)

        if train == False:
            # TODO: implement this later (when Neural Net is working)
            # Point cloud path
            data_file = os.path.join(self.inference_dir, name)

            # Load data
            data = np.loadtxt(data_file, delimiter=" ", dtype=np.float16)

            # Normalize RGB channels & translate point cloud to origin
            data[:, 4] /= 255.0  # R channel
            data[:, 5] /= 255.0  # G channel
            data[:, 6] /= 255.0  # B channel

            # Find minimum x y z to translate point cloud to the origin
            min_coord = np.amin(data[:, 0:3], axis=0)

            # Translate point cloud (center it)
            data[:, 0:3] -= min_coord

            return data

        # Configure paths
        data_file = os.path.join(self.train_dir, name)
        label_file = os.path.join(self.labels_dir, name.split('/')[-1].replace(".txt", ".labels"))

        # Load files
        data = np.loadtxt(data_file, delimiter=" ", dtype=np.float16)  # dtype avoids Out-Of-Memory failure
        labels = np.loadtxt(label_file, dtype=np.uint8)
        labels = labels[:, np.newaxis]

        # Concatenate points & labels
        data = np.concatenate((data, labels), axis=1)

        # Free labels memory
        labels = None

        # Filter points with label == 0
        data = data[data[:, 7] != 0]

        # Substract 1 to class (new range is [0, 7])
        data[:, 7] -= 1

        # Normalize rgb channels
        data[:, 4] /= 255.0  # R channel
        data[:, 5] /= 255.0  # G channel
        data[:, 6] /= 255.0  # B channel

        # Find minimum x y z to translate point cloud to the origin
        min_coord = np.amin(data[:, 0:3], axis=0)

        # Translate point cloud (center it)
        data[:, 0:3] -= min_coord

        return data

    def subsample_dataset(self, method='random'):
        """
        Redefinition of method declared in superclass.

        :param method: method to subsample point cloud (string)
        :param train: load train partition or test.
        :return:
        """

        # List base dir of this dataset
        pointclouds = os.listdir(self.train_dir)

        # List sampled dir: pointcloud that have been yet processed
        if os.path.isdir(self.sampled_clouds):
            processed_pointclouds = os.listdir(self.sampled_clouds)

            # Delete extensions of the files
            processed_pointclouds = [pointcloud[:-4] for pointcloud in processed_pointclouds]
        else:
            processed_pointclouds = []

        for pointcloud in pointclouds:
            if pointcloud[-4:] == ".txt":
                # Only read point clouds (as txt file)
                # Check if point cloud have been yet processed
                if pointcloud[:-4] not in processed_pointclouds:
                    print("{}: processing point cloud... {}".format(self.dset_name, pointcloud))
                    data = None  # Free memory before start to read data again. Avoid Out-Of-Memory in process.
                    data = self.load_point_cloud(pointcloud, train=True)
                    data = self.sample(data, method=method)
                    self.write_point_cloud(pointcloud=data, output_name=pointcloud[:-4], format="npy")

    def download(self):
        """
        Download Semantic3D from its website: Semantic3D.net
        """

        # Check if compressed directory exists
        if not os.path.isdir(self.compressed):
            os.mkdir(self.compressed)

        if not os.path.isdir(self.labels_dir):
            os.mkdir(self.labels_dir)

        # Check if data directory is empty
        pointclouds = os.listdir(self.train_dir)
        pointclouds = [pointcloud for pointcloud in pointclouds if pointcloud[-4:] == ".txt"]

        # Download pointcloud files as 7z
        urls = ["http://www.semantic3d.net/data/point-clouds/training1/bildstein_station1_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/bildstein_station3_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/bildstein_station5_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/domfountain_station1_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/domfountain_station2_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/domfountain_station3_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/neugasse_station1_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg27_station1_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg27_station2_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg27_station4_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg27_station5_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg27_station9_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/sg28_station4_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/untermaederbrunnen_station1_xyz_intensity_rgb.7z",
                "http://www.semantic3d.net/data/point-clouds/training1/untermaederbrunnen_station3_xyz_intensity_rgb.7z"]

        if len(pointclouds) == len(urls):
            # All point clouds have been downloaded
            return None

        labels_urls = "http://www.semantic3d.net/data/sem8_labels_training.7z"

        # Download pointclouds in ./compressed directory
        print("Downloading all files...")
        for url in urls:
            download(url=url, savepath=self.compressed)

        # Download label files in ./compressed directory
        download(labels_urls, savepath=self.compressed)

        # Decompress all files
        print("Decompressing all downloaded files...")
        compressed_files = os.listdir(self.compressed)
        for compressed_file in compressed_files:
            file_path = os.path.join(self.compressed, compressed_file)
            if 'labels' in compressed_file:
                # Save on self.labels_dir
                unzip(file_path, self.labels_dir)
            else:
                # Save on self.train_dir
                unzip(file_path, self.train_dir)


class LiDARProcessor(Processor):
    """
    This class deal with LiDAR point clouds: those whose format is .las or .laz.
    """

    def __init__(self, basepath, training_items='train_items'):
        super().__init__('LiDAR', basepath, training_items)

        self.labels = {
            0: 'base',
            1: 'building',
            2: 'vegetation',
            3: 'rechazo'
        }

    def load_point_cloud(self, name, train=True):
        """
        Load a point cloud given its name. Point cloud must be either .las or .laz format. For the sake of simplicity,
        this code is only considering .las files. This function must discard every point which is not valid for creating
        the train dataset, but this requires to prior know the domain of the problem. So, by the moment, this function
        is going to read every point in .las file.

        Once points have been read, every valid property must be extracted and concatenated in order to get one-dimensional
        arrays with as many data as read properties: X Y Z Intensity R G B Label.

        Points will be translated according to the dimensions of the MBB.

        :param name: the name (filename) of the pointcloud which is being read.
        :param train: if the point cloud is being considered for training or testing.
        :return: numpy array where each line is: x y z intensity r g b label. rgb are normalized to [0, 1] scale, and
        points are translated according to the point with the minimum coordinates (defining the corner of the bounding
        box).
        """

        # Only for debugging process
        super().load_point_cloud(name=name, train=train)

        if train == False:
            # TODO: implement this later (when Neural Net is working)
            pass

        # Configure paths
        data_file = os.path.join(self.train_dir, name)

        # Load files using LASPY library
        if name[-4:] == ".laz":
            # LAS compressed file
            # TODO this later
            pass
        else:
            file = LASFile(data_file, mode='r')

        # Get data as a concatenated numpy array. This gets a (8, N points) matrix.
        data = np.array((file.x, file.y, file.z, file.intensity, file.red, file.green, file.blue, file.classification))

        # Transpose previous matrix to get (N, 8) array.
        data = data.T

        # Discard those points whose classification is 0 or 1
        data = data[data[:, -1] != 0]
        data = data[data[:, -1] != 1]

        # Map classes of the point to a unique classification
        # data[data[:, -1] == 11][:, -1] = 2  # base -> floor
        # data[data[:, -1] == 4][:, -1] = 3  # vegetation
        # data[data[:, -1] == 5][:, -1] = 3  # vegetation
        # 
        # # Finally, assign same labels as used in self.labels
        # data[data[:, -1] == 2][:, -1] = 0
        # data[data[:, -1] == 6][:, -1] = 1
        # data[data[:, -1] == 3][:, -1] = 2
        # data[data[:, -1] == 9][:, -1] = 3
        # data[data[:, -1] > 3][:, -1] = 4

        # Remap point classes
        labels = data[:, -1]

        labels = np.where(labels == 11.0, 0.0, labels)
        labels = np.where(labels == 2.0, 0.0, labels)

        labels = np.where(labels == 6.0, 1.0, labels)

        labels = np.where(labels == 4.0, 2.0, labels)
        labels = np.where(labels == 5.0, 2.0, labels)

        labels = np.where(labels >= 3.0, 3.0, labels)

        # Assign labels to points after remap
        data[:, -1] = labels

        # Normalize Intensity & RGB channels
        data[:, 3] /= 65535.0  # Intensity is 16bit unsigned int [0, 65535]
        data[:, 4] /= 65535.0  # R channel
        data[:, 5] /= 65535.0  # G channel
        data[:, 6] /= 65535.0  # B channel

        # Get min coords in points
        min_coords = np.min(data[:, 0:3], axis=0)

        # Translate points to avoid too big measures that will harm the network performance
        data[:, 0:3] -= min_coords

        # Close LAS file
        file.close()

        return data

    def subsample_dataset(self, method='random'):
        """
        Redefinition of method declared in superclass. If method==None, cloud is processed without loss of points.

        :param method: method to subsample point cloud (string)
        :param train: load train partition or test.
        :return:
        """

        # List base dir of this dataset
        pointclouds = os.listdir(self.train_dir)

        for pointcloud in pointclouds:
            if pointcloud[-4:] == ".las" or pointcloud[-4:] == ".laz":
                # Only read point clouds in LiDAR format
                print("{}: processing point cloud... {}".format(self.dset_name, pointcloud))
                data = self.load_point_cloud(pointcloud, train=True)
                if method is not None:
                    data = self.sample(data, method=method)
                self.write_point_cloud(pointcloud=data, output_name=pointcloud[:-4], format="npy")
                # Free memory
                data = None
