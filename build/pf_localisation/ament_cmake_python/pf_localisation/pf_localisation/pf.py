from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase
from . util import rotateQuaternion, getHeading
import numpy as np
from sklearn.cluster import DBSCAN

class PFLocaliser(PFLocaliserBase):
    def __init__(self, logger, clock):
        super().__init__(logger, clock)
        
        # Motion model parameters
        self.ODOM_ROTATION_NOISE = 0
        self.ODOM_TRANSLATION_NOISE = 0
        self.ODOM_DRIFT_NOISE = 0

        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 5
        self.NUMBER_OF_PARTICLES = 100
        self.NOISE_PARAMETER = 0.1

        # Pre-allocate numpy arrays for better performance
        self.particle_positions = np.zeros((self.NUMBER_OF_PARTICLES, 2))
        self.particle_orientations = np.zeros((self.NUMBER_OF_PARTICLES, 4))
       
    def initialise_particle_cloud(self, initialpose) -> PoseArray:
        """
        Initialize particle cloud using vectorized operations.
        """
        # Generate all random values at once
        position_noise = np.random.normal(scale=self.NOISE_PARAMETER, size=(self.NUMBER_OF_PARTICLES, 2))
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
        Update particle cloud using numpy operations with kidnapped robot recovery.
        Implements a dual-MCL approach with random particle injection when confidence is low.
        """
        # Calculate weights using numpy array operations
        weights = np.array([self.sensor_model.get_weight(scan, pose) for pose in self.particlecloud.poses])
        weights = 5 ** weights

        # Normalize weights
        normalized_weights = weights / np.sum(weights)
        
        # Detect kidnapped robot situation
        average_weight = np.mean(weights)
        weight_variance = np.var(weights)
        
        # Parameters for kidnapped robot detection
        WEIGHT_THRESHOLD = 0.1  # Adjust based on your specific scenario
        RANDOM_PARTICLE_RATIO = 0.2  # 20% random particles when kidnapped
        
        # Check if robot might be kidnapped (low weights indicate poor localization)
        is_kidnapped = average_weight < WEIGHT_THRESHOLD
        
        # Calculate number of particles to resample and inject
        n_particles = len(self.particlecloud.poses)
        n_random = int(n_particles * RANDOM_PARTICLE_RATIO) if is_kidnapped else 0
        n_resample = n_particles - n_random
        
        # Resample existing particles
        resampled_poses = PoseArray()
        if n_resample > 0:
            # Calculate cumulative probabilities
            cumulative_weights = np.cumsum(normalized_weights)
            
            # Generate random numbers for resampling
            random_numbers = np.random.random(n_resample)
            
            # Resample using vectorized operations
            for random_num in random_numbers:
                idx = np.searchsorted(cumulative_weights, random_num)
                if idx >= len(self.particlecloud.poses):
                    idx = len(self.particlecloud.poses) - 1
                resampled_poses.poses.append(self.particlecloud.poses[idx])
        
        # Add random particles if kidnapped
        if n_random > 0:
            # Get map bounds (assuming these are available as class attributes)
            map_bounds = self.get_map_bounds()  # You'll need to implement this
            
            for _ in range(n_random):
                random_pose = Pose()
                # Generate random position within map bounds
                random_pose.position.x = np.random.uniform(map_bounds['x_min'], map_bounds['x_max'])
                random_pose.position.y = np.random.uniform(map_bounds['y_min'], map_bounds['y_max'])
                
                # Generate random orientation
                angle = np.random.uniform(0, 2 * np.pi)
                random_pose.orientation = Quaternion()
                random_pose.orientation.z = np.sin(angle / 2.0)
                random_pose.orientation.w = np.cos(angle / 2.0)
                
                resampled_poses.poses.append(random_pose)
        
        # Add small random noise to all particles to prevent particle depletion
        for pose in resampled_poses.poses:
            pose.position.x += np.random.normal(0, 0.05)
            pose.position.y += np.random.normal(0, 0.05)
            current_angle = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)
            angle_noise = np.random.normal(0, 0.1)
            pose.orientation.z = np.sin((current_angle + angle_noise) / 2.0)
            pose.orientation.w = np.cos((current_angle + angle_noise) / 2.0)
        
        self.particlecloud = resampled_poses

    def estimate_pose(self) -> Pose:
        """
        Estimate pose using DBSCAN clustering for outlier handling.
        """
        # Extract positions and orientations
        positions = np.array([[pose.position.x, pose.position.y, pose.orientation.z, pose.orientation.w] for pose in self.particlecloud.poses])
        
        # Use DBSCAN for clustering (more efficient than IsolationForest for this case)
        clustering = DBSCAN(eps=1, min_samples=10).fit(positions)

        # Find the largest cluster
        largest_cluster = max(set(clustering.labels_), key=list(clustering.labels_).count)
        mask = clustering.labels_ == largest_cluster
        filtered_positions = positions[mask]

        # Create final pose
        final_pose = Pose()
        final_pose.position.x = np.mean(filtered_positions[:, 0])
        final_pose.position.y = np.mean(filtered_positions[:, 1])

        # # Calculate mean orientation (using circular mean for angles)
        final_pose.orientation = Quaternion()
        final_pose.orientation.z = np.mean(filtered_positions[:, 2]) # Simplified for demonstration
        final_pose.orientation.w = np.mean(filtered_positions[:, 3]) # Simplified for demonstration

        return final_pose
