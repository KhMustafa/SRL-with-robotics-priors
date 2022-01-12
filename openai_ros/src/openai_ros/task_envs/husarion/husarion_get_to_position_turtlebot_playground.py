"""
This script is responsible for getting observation vector from the robot's sensors, computing the reward function for
the RL agent, setting the action, and generating random desired targets for navigation
"""


import rospy
import numpy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from gym import spaces
from openai_ros.robot_envs import husarion_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from math import atan2
import random

timestep_limit_per_episode = 10000  # Can be any Value
camera_obs_pixels = (32, 24)
camera_obs_len = 32 * 24 * 3

register(
    id='HusarionGetToPosTurtleBotPlayGround-v0',
    entry_point='openai_ros.task_envs.husarion.husarion_get_to_position_turtlebot_playground:HusarionGetToPosTurtleBotPlayGroundEnv',
    max_episode_steps=timestep_limit_per_episode,
)


class HusarionGetToPosTurtleBotPlayGroundEnv(husarion_env.HusarionEnv):
    def __init__(self):
        """
        This Task Env is designed for having the husarion in the husarion world
        closed room with columns.
        closed room with columns.
        It will learn how to move around without crashing.
        """

        # NOTE: Gym can only have one low and high value, so this holds for both linear and angular speed
        # get the parameters from the yaml file
        low = rospy.get_param('/husarion/min_ang_speed')
        high = rospy.get_param('/husarion/max_ang_speed')
        self.init_linear_forward_speed = rospy.get_param('/husarion/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/husarion/init_linear_turn_speed')
        self.new_ranges = rospy.get_param('/husarion/new_ranges')
        self.max_laser_value = rospy.get_param('/husarion/max_laser_value')
        self.min_laser_value = rospy.get_param('/husarion/min_laser_value')
        self.work_space_x_max = rospy.get_param("/husarion/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/husarion/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/husarion/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/husarion/work_space/y_min")
        self.nsteps = rospy.get_param("/husarion/nsteps")
        self.precision = rospy.get_param('/husarion/precision')
        self.move_base_precision = rospy.get_param('/husarion/move_base_precision')

        self.action_space = spaces.Box(low, high, shape=[2])
        self.successful_episodes = 0
        self.reached_max_steps = 0

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        self.reached_des_pos = False

        # generate a random target position
        self.desired_position = Point()
        self.desired_position.x = random.uniform(0, 4)
        self.desired_position.y = random.uniform(-2.0, 2.0)

        self.precision_epsilon = 1.0 / (10.0 * self.precision)

        # We create the arrays for the laser readings
        # We also create the arrays for the odometer readings
        # We join them together.
        camera_image = self._check_camera_rgb_image_raw_ready()

        laser_scan = self._check_laser_scan_ready()
        num_laser_readings = len(laser_scan.ranges) / self.new_ranges
        high_laser = numpy.full((num_laser_readings), self.max_laser_value)
        low_laser = numpy.full((num_laser_readings), self.min_laser_value)

        # We place the Maximum and minimum values of the X,Y and YAW of the odometry
        # The odometry yaw can be any value in the circunference.
        high_odometry = numpy.array([self.work_space_x_max,
                                     self.work_space_y_max,
                                     3.14])
        low_odometry = numpy.array([self.work_space_x_min,
                                    self.work_space_y_min,
                                    -1 * 3.14])

        # Now we fetch the max and min of the Desired Position in 2D XY
        # We use the exact same as the workspace, just because make no sense
        # Consider points outside the workspace
        high_des_pos = numpy.array([self.work_space_x_max,
                                    self.work_space_y_max
                                    ])
        low_des_pos = numpy.array([self.work_space_x_min,
                                   self.work_space_y_min
                                   ])

        # We join both arrays
        high = numpy.concatenate([high_laser, high_odometry, high_des_pos])
        low = numpy.concatenate([low_laser, low_odometry, low_des_pos])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards
        self.distance_multiplier = rospy.get_param("/husarion/distance_multiplier")
        self.alive_reward = rospy.get_param("/husarion/alive_reward")
        self.end_episode_points = rospy.get_param("/husarion/end_episode_points")

        self.cumulated_steps = 0.0

        self.laser_filtered_pub = rospy.Publisher('/turtlebot/laser/scan_filtered', LaserScan, queue_size=1)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HusarionGetToPosTurtleBotPlayGroundEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(self.init_linear_forward_speed,
                       self.init_linear_turn_speed,
                       epsilon=self.move_base_precision,
                       update_rate=10)

        self.generate_desired_pos()

        return True

    def _get_desired_position(self):
        desired_position = [round(self.desired_position.x, 2),
                            round(self.desired_position.y, 2)]
        return desired_position

    def _get_target_position(self):
        return self.desired_position

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0

        self.index = 0

        odometry = self.get_odom()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(odometry.pose.pose.position,
                                                                                     self.desired_position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the SumitXl
        based on the action number given.
        :param action: The action variable contains both the linear and the angular speed
        """

        rospy.logdebug("Start Set Action ==>" + str(action))
        # We tell Husarion the linear and angular speed to set to execute
        linear_speed = 0.6 * action[0]
        angular_speed = 0.8 * action[1]
        if linear_speed < 0:
            linear_speed = -1.0 * linear_speed
        self.move_base(linear_speed, angular_speed, epsilon=self.move_base_precision, update_rate=10)

        rospy.logdebug("END Set Action ==>" + str(action))

    def _get_obs(self, action):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        HusarionEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        camera_image = self.get_camera_rgb_image_raw()
        # We get the laser scan data
        laser_scan = self.get_laser_scan()

        discretized_laser_scan = self.discretize_scan_observation(laser_scan, self.new_ranges)

        # We get the odometry so that SumitXL knows where it is.
        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y

        # We get the orientation of the cube in RPY
        roll, pitch, yaw = self.get_orientation_euler()
        # We round to only two decimals to avoid very big Observation space
        # We only want the X and Y position and the Yaw
        odometry_array = [round(x_position, 2),
                          round(y_position, 2),
                          round(yaw, 2)]

        # We fetch also the desired position because it conditions the learning
        # It also make it dynamic, because we can change the desired position and the
        # learning will be able to adapt.
        desired_position = [round(self.desired_position.x, 2),
                            round(self.desired_position.y, 2)]

        diff_x = self.desired_position.x - x_position
        diff_y = self.desired_position.y - y_position

        bridge = CvBridge()
        cam_cv = bridge.imgmsg_to_cv2(camera_image, desired_encoding="rgb8")
        cam_cv = cv2.resize(cam_cv, camera_obs_pixels, interpolation=cv2.INTER_AREA)
        cam_list = cam_cv.flatten().tolist()

        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0
        desired_position_array = Point()
        desired_position_array.x = desired_position[0]
        desired_position_array.y = desired_position[1]
        desired_position_array.z = 0.0
        state = discretized_laser_scan + odometry_array + desired_position + cam_list

        return state

    def _is_done(self, observations):
        """
        We consider that the episode has finished when:
        1) Husarion has moved ouside the workspace defined.
        2) Husarion is too close to an object
        3) Husarion has reached the desired position
        """

        # We fetch data through the observations
        # Its all the array except from the last four elements, which are XY odom and XY des_pos
        laser_readings = observations[:-5 - camera_obs_len]

        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        roll, pitch, yaw = self.get_orientation_euler()
        # We round to only two decimals to avoid very big Observation space
        # We only want the X and Y position and the Yaw
        odometry_array = [round(x_position, 2),
                          round(y_position, 2),
                          round(yaw, 2)]
        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0

        desired_position_array = [round(self.desired_position.x, 2),
                                  round(self.desired_position.y, 2)]
        desired_position = Point()
        desired_position.x = desired_position_array[0]
        desired_position.y = desired_position_array[1]
        desired_position.z = 0.0

        rospy.logwarn("is DONE? current_position=" + str(current_position))
        rospy.logwarn("is DONE? desired_position=" + str(desired_position))

        too_close_to_object = self.check_husarion_has_crashed(laser_readings)
        inside_workspace = self.check_inside_workspace(current_position)
        self.reached_des_pos = self.check_reached_desired_position(current_position,
                                                                   desired_position)

        is_done = too_close_to_object or not inside_workspace or self.reached_des_pos or (
                (self.cumulated_steps + 1) % self.nsteps) < 1

        return is_done

    def _compute_reward(self, observations, done):
        """
        We will reward the following behaviours:
        1) The distance to the desired point has increase from last step
        2) The robot has reached the desired point
        
        We will penalise the following behaviours:
        1) Ending the episode without reaching the desired pos. That means it has crashed
        or it has gone outside the workspace 
        
        """

        laser_readings = observations[:-5 - camera_obs_len]

        desired_position_array = [round(self.desired_position.x, 2),
                                  round(self.desired_position.y, 2)]

        odometry = self.get_odom()
        x_position = odometry.pose.pose.position.x
        y_position = odometry.pose.pose.position.y
        roll, pitch, yaw = self.get_orientation_euler()
        # We round to only two decimals to avoid very big Observation space
        # We only want the X and Y position and the Yaw
        odometry_array = [round(x_position, 2),
                          round(y_position, 2),
                          round(yaw, 2)]
        current_position = Point()
        current_position.x = odometry_array[0]
        current_position.y = odometry_array[1]
        current_position.z = 0.0

        desired_position = Point()
        desired_position.x = desired_position_array[0]
        desired_position.y = desired_position_array[1]
        desired_position.z = 0.0

        diff_x = self.desired_position.x - current_position.x
        diff_y = self.desired_position.y - current_position.y

        # We get the orientation of the cube in RPY
        roll, pitch, yaw = self.get_orientation_euler()

        # We use the rotation matrix to express the difference vector into the robots frame of reference
        [diff_x, diff_y] = rot([diff_x, diff_y], -1 * yaw)

        # Angle to the target can no be easily calculated withtout suffering from the -pi to pi shifts
        diff_angle = atan2(diff_y, diff_x)

        distance_from_des_point = self.get_distance_from_desired_point(current_position, desired_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point
        distance_from_wall = sum(laser_readings[:3] + laser_readings[-3:]) / 6

        if self.reached_des_pos:
            # reward = self.end_episode_points
            reward = 10
            self.successful_episodes += 1

        elif self.check_husarion_has_crashed(laser_readings):
            reward = -10
        else:
            reward_distance = -10 * distance_difference

            reward = reward_distance + self.alive_reward

        self.previous_distance_from_des_point = distance_from_des_point

        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return reward

    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    # Internal TaskEnv Methods
    def update_desired_pos(self, new_position):
        """
        With this method you can change the desired position that you want
        Usarion to be that initialy is set through rosparams loaded through
        a yaml file possibly.
        :new_position: Type Point, because we only value the position.
        """
        self.desired_position.x = new_position.x
        self.desired_position.y = new_position.y

    def generate_desired_pos(self):

        self.desired_position.x = random.uniform(0, 4)
        self.desired_position.y = random.uniform(-2.0, 2.0)

    def resizeImage(self, data, new_pixelsize):
        new_data = []
        reductionrate = len(data) / new_pixelsize
        for i in range(len(data)):
            if i % (reductionrate * 3) == 0:
                new_data.append(data[i])  # red
                new_data.append(data[i + 1])  # green
                new_data.append(data[i + 2])  # blue
        return new_data

    def discretize_scan_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """

        discretized_ranges = []
        mod = len(data.ranges) / new_ranges

        filtered_range = []

        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))

        nan_value = (self.min_laser_value + self.min_laser_value) / 2.0

        for i, item in enumerate(data.ranges):
            if i % mod == 0:
                if item == float('Inf') or numpy.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif numpy.isnan(item):
                    rospy.logerr("Nan Value=" + str(item) + "Assigning MIN value")
                    discretized_ranges.append(self.min_laser_value)
                else:
                    # We clamp the laser readings
                    if item > self.max_laser_value:
                        discretized_ranges.append(round(self.max_laser_value, 2))
                    elif item < self.min_laser_value:
                        discretized_ranges.append(round(self.min_laser_value, 2))
                    else:
                        discretized_ranges.append(round(item, 2))
                # We add last value appended
                filtered_range.append(discretized_ranges[-1])
            else:
                # We add value zero
                filtered_range.append(0.0)

        self.publish_filtered_laser_scan(laser_original_data=data,
                                         new_filtered_laser_range=filtered_range)

        return discretized_ranges

    def get_orientation_euler(self):
        # We convert from quaternions to euler
        orientation_list = [self.odom.pose.pose.orientation.x,
                            self.odom.pose.pose.orientation.y,
                            self.odom.pose.pose.orientation.z,
                            self.odom.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def get_distance_from_desired_point(self, current_position, desired_position):
        """
        Calculates the distance from the current position to the desired point
        :param current_position: 
        :param desired_position:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                desired_position)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def check_husarion_has_crashed(self, laser_readings):
        """
        Based on the laser readings we check if any laser readingdistance is below 
        the minimum distance acceptable.
        """
        husarion_has_crashed = False
        laser_readings_raw = self.get_laser_scan().ranges

        for laser_distance in laser_readings_raw:
            if laser_distance <= self.min_laser_value:
                husarion_has_crashed = True
                rospy.logwarn("HAS CRASHED==>" + str(laser_distance) + ", min=" + str(self.min_laser_value))
                break

            elif laser_distance < self.min_laser_value:
                rospy.logerr("Value of laser shouldnt be lower than min==>" + str(laser_distance) + ", min=" + str(
                    self.min_laser_value))
            # elif laser_distance > self.max_laser_value:
            # rospy.logerr("Value of laser shouldnt be higher than max==>"+str(laser_distance)+", max="+str(self.max_laser_value))

        self.previous_laser_readings = laser_readings
        return husarion_has_crashed

    def check_inside_workspace(self, current_position):
        """
        We check that the current position is inside the given workspace.
        """
        is_inside = False

        if self.work_space_x_min < current_position.x <= self.work_space_x_max:
            if self.work_space_y_min < current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside

    def check_reached_desired_position(self, current_position, desired_position, epsilon=0.4):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False
        distance_from_des_point = self.get_distance_from_desired_point(current_position, desired_position)
        if abs(distance_from_des_point) <= epsilon:
            is_in_desired_pos = True

        rospy.logdebug("###### IS DESIRED POS ? ######")
        rospy.logdebug("is_in_desired_pos" + str(is_in_desired_pos))
        rospy.logdebug("############")
        return is_in_desired_pos

    def publish_filtered_laser_scan(self, laser_original_data, new_filtered_laser_range):

        length_range = len(laser_original_data.ranges)
        length_intensities = len(laser_original_data.intensities)

        laser_filtered_object = LaserScan()

        h = Header()
        h.stamp = rospy.Time.now()  # Note you need to call rospy.init_node() before this will work
        h.frame_id = "chassis"

        laser_filtered_object.header = h
        laser_filtered_object.angle_min = laser_original_data.angle_min
        laser_filtered_object.angle_max = laser_original_data.angle_max
        laser_filtered_object.angle_increment = laser_original_data.angle_increment
        laser_filtered_object.time_increment = laser_original_data.time_increment
        laser_filtered_object.scan_time = laser_original_data.scan_time
        laser_filtered_object.range_min = laser_original_data.range_min
        laser_filtered_object.range_max = laser_original_data.range_max

        laser_filtered_object.ranges = []
        laser_filtered_object.intensities = []
        for item in new_filtered_laser_range:
            laser_filtered_object.ranges.append(item)
            laser_filtered_object.intensities.append(item)

        self.laser_filtered_pub.publish(laser_filtered_object)


def rot(x, theta):
    x = numpy.array(x)
    r = numpy.array(((numpy.cos(theta), -numpy.sin(theta)),
                     (numpy.sin(theta), numpy.cos(theta))))
    result = r.dot(x.transpose())
    return result.transpose()
