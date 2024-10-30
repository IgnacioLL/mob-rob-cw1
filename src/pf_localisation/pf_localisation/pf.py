from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from . pf_base import PFLocaliserBase

from . util import rotateQuaternion, getHeading

import numpy as np
import pandas as pd
from copy import deepcopy


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self, logger, clock):
        # ----- Call the superclass constructor
        super().__init__(logger, clock)
        
        # ----- Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0 # Odometry model x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0 # Odometry model y axis (side-to-side) noise

 
        # ----- Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 20     # Number of readings to predict
        self.NUMBER_OF_PARTICLES = 100
        
       
    def initialise_particle_cloud(self, initialpose: Pose) -> PoseArray:
        """
        Set particle cloud to initialpose plus noise

        Called whenever an initialpose message is received (to change the
        starting location of the robot), or a new occupancy_map is received.
        self.particlecloud can be initialised here. Initial pose of the robot
        is also set here.
        
        :Args:
            | initialpose: the initial pose estimate
        :Return:
            | (geometry_msgs.msg.PoseArray) poses of the particles
        """
        new_poses = PoseArray()
        temp_pose = deepcopy(initialpose)

        for _ in range(self.NUMBER_OF_PARTICLES):
            temp_pose.pose.x = initialpose.pose.x + np.random.normal()
            temp_pose.pose.y = initialpose.pose.y + np.random.normal()

            temp_pose.orientation = rotateQuaternion(temp_pose, np.random.uniform(0, 2*np.pi))

            new_poses.poses.append(temp_pose)
        
        return new_poses


 
    
    def update_particle_cloud(self, scan) -> PoseArray:
        """
        This should use the supplied laser scan to update the current
        particle cloud. i.e. self.particlecloud should be updated.
        
        :Args:
            | scan (sensor_msgs.msg.LaserScan): laser scan to use for update

         """
        weights = [(pose, self.sensor_model.get_weight(scan, pose)) for pose in self.particlecloud]

        weights_pd = pd.DataFrame(weights, columns=["pose", "probability"])

        weights_pd['norm_prob'] = weights_pd['probability']/weights_pd['probability'].sum()
        weights_pd['upper_limit'] = weights_pd['norm_prob'].cumsum()
        weights_pd['lower_limit'] = weights_pd['upper_limit'].shift().fillna(0)


        resampled_poses = PoseArray()
        for random in np.random.random(weights_pd.shape[0]):
            for row in weights_pd.itertuples(index=False):
                if (random <= row['upper_limit']) & (random > row['lower_limit']):
                    resampled_poses.poses.append(row['pose'])
                    break
        
        self.particlecloud = resampled_poses


    def estimate_pose(self):
        """
        This should calculate and return an updated robot pose estimate based
        on the particle cloud (self.particlecloud).
        
        Create new estimated pose, given particle cloud
        E.g. just average the location and orientation values of each of
        the particles and return this.
        
        Better approximations could be made by doing some simple clustering,
        e.g. taking the average location of half the particles after 
        throwing away any which are outliers

        :Return:
            | (geometry_msgs.msg.Pose) robot's estimated pose.
         """
        data = [(pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orienation.w) for pose in self.particlecloud]
        dataframe = pd.DataFrame(data, columns=["px", 'py', 'pz', 'ox', 'oy', 'oz', 'ow'])
        
        final_point = Point()
        px = dataframe['px'].mean()
        py = dataframe['py'].mean()
        pz = dataframe['pz'].mean()
        
        final_point.x = px
        final_point.y = px
        final_point.z = px

        final_pose = Pose()
        final_pose.position = final_point


        final_orientation = Quaternion()
        ox = dataframe['ox'].mean()
        oy = dataframe['oy'].mean()
        oz = dataframe['oz'].mean()
        ow = dataframe['ow'].mean()

        final_orientation.x = ox
        final_orientation.y = oy
        final_orientation.z = oz
        final_orientation.w = ow

        final_pose.orientation = final_orientation

        return final_pose

        