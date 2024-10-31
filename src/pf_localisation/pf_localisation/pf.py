from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
import numpy as np
from sklearn.cluster import DBSCAN

class PFLocaliser(PFLocaliserBase):
    def __init__(self, logger, clock):
        super().__init__(logger, clock)
        
        # Motion model parameters
        self.ODOM_ROTATION_NOISE = 1/500
        self.ODOM_TRANSLATION_NOISE = 1/500
        self.ODOM_DRIFT_NOISE = 1/500

        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 5
        self.NUMBER_OF_PARTICLES = 1000
        self.NOISE_PARAMETER = 0.5

        # Pre-allocate numpy arrays for better performance
        self.particle_positions = np.zeros((self.NUMBER_OF_PARTICLES, 2))
        self.particle_orientations = np.zeros((self.NUMBER_OF_PARTICLES, 4))
    
    def initialise_particle_cloud(self, initialpose) -> PoseArray:
        """
        Initialize particle cloud using vectorized operations.
        """

        print("Initial pose is:", initialpose)
        # Generate all random values at once
        position_noise = np.random.normal(scale=self.NOISE_PARAMETER/2, size=(self.NUMBER_OF_PARTICLES, 2)) + np.random.standard_t(df=2, size=(self.NUMBER_OF_PARTICLES, 2)) * self.NOISE_PARAMETER/2
        orientation_angles = np.random.uniform(0, np.pi*2, self.NUMBER_OF_PARTICLES)
        
        # Create positions array
        initial_x = initialpose.pose.pose.position.x
        initial_y = initialpose.pose.pose.position.y
        self.particle_positions[:, 0] = initial_x + position_noise[:, 0]
        self.particle_positions[:, 1] = initial_y + position_noise[:, 1]
        
        # Create poses array efficiently
        new_poses = PoseArray()
        initial_orientation = initialpose.pose.pose.orientation
        
        for i in range(self.NUMBER_OF_PARTICLES):
            pose = Pose()
            pose.position.x = self.particle_positions[i, 0]
            pose.position.y = self.particle_positions[i, 1]
            pose.orientation = rotateQuaternion(initial_orientation, orientation_angles[i])
            new_poses.poses.append(pose)

        return new_poses

    def update_particle_cloud(self, scan) -> PoseArray:
        """
        Update particle cloud using numpy operations for better performance.
        """
        # Calculate weights using numpy array operations
        weights = np.array([self.sensor_model.get_weight(scan, pose) for pose in self.particlecloud.poses])

        weights = 20 ** weights      

        # Normalize weights
        normalized_weights = weights / np.sum(weights)
        
        # Calculate cumulative probabilities
        cumulative_weights = np.cumsum(normalized_weights)
        
        # Generate random numbers for resampling
        random_numbers = np.random.random(len(weights))
        
        # Resample using vectorized operations
        resampled_poses = PoseArray()
        for random_num in random_numbers:
            # Find the first weight greater than random number
            idx = np.searchsorted(cumulative_weights, random_num)
            if idx >= len(self.particlecloud.poses):
                idx = len(self.particlecloud.poses) - 1
            resampled_poses.poses.append(self.particlecloud.poses[idx])

        
        self.particlecloud = resampled_poses


    def estimate_pose(self) -> Pose:
        """
        Estimate pose using DBSCAN clustering for outlier handling.
        """
        # Extract positions and orientations
        positions = np.array([[pose.position.x, pose.position.y] for pose in self.particlecloud.poses])
        relevant_positions = np.array([[pose.position.x, pose.position.y, pose.orientation.z, pose.orientation.w] for pose in self.particlecloud.poses])
        
        # Use DBSCAN for clustering (more efficient than IsolationForest for this case)
        clustering = DBSCAN(eps=1, min_samples=10).fit(positions)

        
        # Find the largest cluster
        largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
        mask = clustering.labels_ == largest_cluster
        filtered_positions = relevant_positions[mask]

        # Create final pose
        final_pose = Pose()
        final_pose.position.x = np.mean(filtered_positions[:, 0])
        final_pose.position.y = np.mean(filtered_positions[:, 1])

        # # Calculate mean orientation (using circular mean for angles)
        final_pose.orientation = Quaternion()
        final_pose.orientation.z = np.mean(filtered_positions[:, 2]) # Simplified for demonstration
        final_pose.orientation.w = np.mean(filtered_positions[:, 3]) # Simplified for demonstration

        return final_pose