"""
This script translates the executed actions to gazebo environment, spawns the robot at a random position at the
beginning of every episode, and reset the simulation environment when the episode is over.
"""
import rospy
import gym
import os
from gym.utils import seeding
from .gazebo_connection import GazeboConnection
from .controllers_connection import ControllersConnection
from openai_ros.msg import RLExperimentInfo
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from load_model import model_control
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Point
import random


class RobotGazeboEnv(gym.Env):

    def __init__(self, robot_name_space, controllers_list, reset_controls, start_init_physics_parameters=True,
                 reset_world_or_sim="SIMULATION"):

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # To reset Simulations
        rospy.logdebug("START init RobotGazeboEnv")
        self.gazebo = GazeboConnection(start_init_physics_parameters, reset_world_or_sim)
        self.controllers_object = ControllersConnection(namespace=robot_name_space, controllers_list=controllers_list)
        self.reset_controls = reset_controls
        self.seed()
        self.load_target_model = model_control()
        # Set up ROS related variables
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.reward_pub = rospy.Publisher('/openai/reward', RLExperimentInfo, queue_size=1)
        rospy.logdebug("END init RobotGazeboEnv")

    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Here we should convert the action num to movement action, execute the action in the
        simulation and get the observations result of performing that action.
        """
        rospy.logdebug("START STEP OpenAIROS")

        self.gazebo.unpauseSim()
        self._set_action(action)
        self.gazebo.pauseSim()
        obs = self._get_obs(action)
        done = self._is_done(obs)
        info = {}
        reward = self._compute_reward(obs, done)
        self.cumulated_episode_reward += reward

        rospy.logdebug("END STEP OpenAIROS")

        return obs, reward, done, info

    def reset(self):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        target = self._get_desired_position()
        self._update_episode()

        action = [0,
                  0]
        obs = self._get_obs(action)
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs, target

    def target(self):
        return self._get_desired_position()

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

    def _update_episode(self):
        """
        Publishes the cumulated reward of the episode and 
        increases the episode number by one.
        :return:
        """
        # change the initial pose of the robot every episode
        pose = Pose()

        # randomize the initial pose of the robot in each episode
        target = self._get_target_position()
        pose.position.x = random.uniform(0, 3.5)
        pose.position.y = random.uniform(-2, 2)
        roll = 0
        theta = random.uniform(-3.14, 3.14)
        pitch = 0
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w) = quaternion_from_euler(roll,
                                                                                                                 pitch,
                                                                                                                 theta)
        state = ModelState()
        state.model_name = "turtlebot3"
        state.pose = pose
        try:
            ret = self.set_model_srv(state)
            print ret.status_message
        except Exception, e:
            rospy.logerr('Error on calling service: %s', str(e))

        rospy.logwarn("PUBLISHING REWARD...")
        self._publish_reward_topic(
            self.cumulated_episode_reward,
            self.episode_num
        )
        rospy.logwarn("PUBLISHING REWARD...DONE=" + str(self.cumulated_episode_reward) + ",EP=" + str(self.episode_num))
        # get the relative position of the file
        parentDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        modeldir = os.path.join(parentDirectory, 'turtlebot3_gazebo/worlds/target.world')
        model_name = 'ground_plane_1'
        position = [target.x, target.y, 0]
        orientation = [0, 0, 0]
        self.load_target_model.delete_model(model_name)
        print("delete target")
        self.load_target_model.spawn_model(modeldir, model_name, "robot_namespace", position, orientation)
        print("spawn target")
        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _publish_reward_topic(self, reward, episode_number=1):
        """
        This function publishes the given reward in the reward topic for
        easy access from ROS infrastructure.
        :param reward:
        :param episode_number:
        :return:
        """
        reward_msg = RLExperimentInfo()
        reward_msg.episode_number = episode_number
        reward_msg.episode_reward = reward
        self.reward_pub.publish(reward_msg)

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls:
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _get_desired_position(self):
        raise NotImplementedError()

    def _generate_desired_pos(self, episode_num):
        raise NotImplementedError()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_done(self, observations):
        """Indicates whether or not the episode is done ( the robot has fallen for example).
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        raise NotImplementedError()
