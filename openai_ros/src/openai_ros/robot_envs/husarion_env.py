"""
This script is responsible for setting up simulated robot. This includes the robot's publishers and
subscribers nodes and topics. Additionally, it controls the motion of the robot.
"""

import rospy
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class HusarionEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        """
        Initializes a new HusarionEnv environment.
        The Sensors: The sensors accessible are the ones considered usefully for learning.
        
        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/rgb/image_raw: RGB camera
        * /turtlebot/laser/scan: Laser Readings
        
        Actuators Topic List: /cmd_vel, 
        
        Args:
        """
        rospy.logdebug("Start HusarionEnv INIT...")

        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(HusarionEnv, self).__init__(controllers_list=self.controllers_list,
                                          robot_name_space=self.robot_name_space,
                                          reset_controls=False,
                                          start_init_physics_parameters=False)

        self.gazebo.unpauseSim()
        self._check_all_sensors_ready()
        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self._check_publishers_connection()
        self.gazebo.pauseSim()

        rospy.logdebug("Finished HusarionEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()

        rospy.logwarn("Waiting for env to reset..")
        rate = rospy.Rate(0.1)
        start_wait_time = rospy.get_rostime().to_sec()

        while not rospy.is_shutdown() and rospy.get_rostime().to_sec() - start_wait_time < 0.5:
            rate.sleep()

        rospy.logwarn("Waiting time elapsed...")
        return True

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_camera_rgb_image_raw_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")
            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")

        return self.odom

    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for /camera/rgb/image_raw to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message("/camera/rgb/image_raw", Image, timeout=5.0)
                rospy.logdebug("Current /camera/rgb/image_raw READY=>")

            except:
                rospy.logerr("Current /camera/rgb/image_raw not ready yet, retrying for getting camera_rgb_image_raw")
        return self.camera_rgb_image_raw

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan

    def _odom_callback(self, data):
        self.odom = data

    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=50):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logwarn("TurtleBot3 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        # self.wait_until_twist_achieved(cmd_vel_value,
        #                                 epsilon,
        #                                 update_rate)
        self.wait_until_time_elapsed(update_rate)

    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, angular_speed_noise=0.005):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        Bear in mind that the angular won't be controlled , because it's too imprecise.
        We will only consider to check if It's moving or not inside the angular_speed_noise fluctuations it has.
        from the odometer.
        param cmd_vel_value: Twist we want to wait to reach.
        param epsilon: Error acceptable in odometer readings.
        param update_rate: Rate at which we check the odometer.
        """
        rospy.logwarn("START wait_until_twist_achieved...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.01

        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))

        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + 1 * epsilon
        angular_speed_minus = angular_speed - 1 * epsilon

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z

            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + "," + str(
                linear_speed_plus) + "]")
            rospy.logdebug(
                "Angular VEL=" + str(odom_angular_vel) + ", angular_speed asked=[" + str(angular_speed) + "]")

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (
                    odom_angular_vel > angular_speed_minus)

            if linear_vel_are_close and angular_vel_are_close:
                rospy.logwarn("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            elif rospy.get_rostime().to_sec() - start_wait_time > 1:
                rospy.logwarn("Velocity timeout!")
                break

            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time - start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time) + "]")

        rospy.logwarn("END wait_until_twist_achieved...")

        return delta_time

    def wait_until_time_elapsed(self, update_rate):

        rospy.logwarn("START wait_until_time_elapsed...")

        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()

        while not rospy.is_shutdown():
            current_odometry = self._check_odom_ready()
            if rospy.get_rostime().to_sec() - start_wait_time > 0.5:
                rospy.logwarn("Time passed: " + str(rospy.get_rostime().to_sec() - start_wait_time))
                rospy.logwarn("Actual linear speed: " + str(current_odometry.twist.twist.linear.x))
                rospy.logwarn("Actual linear speed: " + str(current_odometry.twist.twist.angular.z))
                break

            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()

    def get_odom(self):
        return self.odom

    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw

    def get_laser_scan(self):
        return self.laser_scan
