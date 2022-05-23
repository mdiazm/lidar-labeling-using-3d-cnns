"""
Utility functions to display 3D point clouds.
"""

import open3d as o3d
import numpy as np

def visualize_cloud(pointcloud=None):
    """
    Visualize a point cloud using Open3D library

    :param pointcloud: point cloud data as numpy array
    :return: anything
    """

    # From NumPy to open3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 4:7])

    # Visualize geometries
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")

def visualize_cloud_splitted(pointcloud_list=None):
    """
    Visualize a point cloud by fragments. Each fragments correspond with a slice of the original point cloud.

    :param pointcloud_list: list of numpy arrays containing parts of the original point cloud
    :return: anything
    """

    # From list of numpy arrays to list of o3d.geometry.pointcloud
    pc_data = []
    for pointcloud in pointcloud_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 4:7])

        pc_data += [pcd]

    # Visualize geometries
    o3d.visualization.draw_geometries(pc_data, window_name="Point Cloud fragments viewer")

class PointCloudViewer:
    """
    To visualize point clouds during block generation or testing. An initial o3d.visualization.Visualizer is created
    in constructor, and the whole initial point cloud is viewed. After several calls to self.vis.update(), the geometry
    will be updated.

    The visualizer is closed when close() is called.
    """

    def __init__(self, pointcloud=None):
        """
        Constructor of the class.
        :param pointcloud: initial geometry (the whole point cloud) as a numpy array.
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud block generation viewer")

        # Create geometry Open 3D object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 4:7])

        self.vis.add_geometry(pcd)

    def __call__(self, updated_pointcloud):
        """
        The "update" method for updating geometry and renderer.

        :param updated_pointcloud: same point cloud but some points (the block points) are colour-changed.
        """

        # Create geometry object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(updated_pointcloud[:, 0:3])
        pcd.colors = o3d.utility.Vector3dVector(updated_pointcloud[:, 4:7])

        # Update geometry and renderer
        self.vis.update_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """
        Close point cloud viewer (destroy window).
        """

        self.vis.destroy_window()