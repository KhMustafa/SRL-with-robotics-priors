#!/usr/bin/env python

"""
This script is the main script in the repo. It asks the user whether to load an SRL learned parameters or to start
with a random policy. It also asks if the RL should start from scratch or to continue learning from a partially
learned parameters. Based on the user's input the learning process will start.
"""



from DDPG_agent import DDPGAgent as Agent
import sys
import numpy as np
import os
import gym
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from std_msgs.msg import Empty
from Logger import Logger
from StateRepresentation_Continuous_V2 import StateRepresentation as srl_py
from DDPG_agent import OUNoise
from utils import reshape_observation


def main(continue_rl, continue_srl, env):
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    nepisodes = rospy.get_param("/husarion/nepisodes")
    max_eps_len = rospy.get_param("/husarion/nsteps")
    forward_lim = rospy.get_param("/husarion/max_forward_speed")
    angular_lim = rospy.get_param("/husarion/max_ang_speed")
    # Handle saved data
    folder = os.path.join(rospkg.RosPack().get_path("rosbot_srl"), "training_results")
    logger = Logger(folder + "/trainingdata")
    # the low dimensional state size proposed by the SRL reults
    state_size = 9
    patchsize = [32, 24, 40]
    action_size = env.action_space.shape[0]

    print("loading agent...")
    agent = Agent(state_size, action_size, continuelearning=continue_rl)
    print("...finish loading agent")

    print("loading srl...")
    srl = srl_py(patchsize, usingros=True, continuelearning=continue_srl)
    print("...finish loading srl")

    sample_number = 0
    noise = OUNoise(action_size)

    # MAIN TRAINING LOOP ------------------------------------------------------------------------------------
    for episode in range(nepisodes):
        # Print information about the current episodes
        print('Starting episode {} / {}'.format(episode, nepisodes))
        # Initialize the environment and get first state of the robot
        state, desired_position = env.reset()

        o_srl = reshape_observation(state)
        initial_action = [0, 0]
        # concatenate the randomly generated target position with the learned state
        state = np.concatenate((srl.predict(o_srl), np.reshape(desired_position, (1, 2))), axis=1)
        state = np.concatenate((state, np.reshape(initial_action, (1, 2))), axis=1)

        noise.reset()

        # COLLECT DATA FROM THE ENVIRONMENT AND TRAIN THE AGENT --------------------------------------------------------
        ep_len, ep_ret = 0, 0
        for step in range(max_eps_len + 1):
            # 1. Compute an action according to policy and noise
            action = agent.act(state)
            action = [np.clip(action[0] + noise.evolve_state()[0], 0.0, forward_lim),
                      np.clip(action[1] + noise.evolve_state()[1], -angular_lim, angular_lim)]
            # 2. Take step in the env
            next_state, reward, done, _ = env.step(action)
            o2_srl = reshape_observation(next_state)
            next_state = np.concatenate((srl.predict(o2_srl), np.reshape(desired_position, (1, 2))), axis=1)
            next_state = np.concatenate((next_state, np.reshape(action, (1, 2))), axis=1)

            # Add some information to the logger for later evaluation
            logger.ulog("actions", action)
            logger.ulog("rewards", reward)
            logger.ulog("states", state)

            # 3. Update variables to keep track of what's happening
            ep_ret += reward
            ep_len += 1
            sample_number += 1

            # 4. Store experience to replay buffer
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Break out of the loop if the episode has finished and train the agent
            if done or (ep_len == max_eps_len):
                if sample_number > 20 * agent.batch_size:
                    for _ in range(ep_len):
                        agent.learn()
                    agent.save_model()
                rospy.logwarn("DONE")
                logger.ulog("episode_length", ep_len)
                logger.ulog("episode_reward", ep_ret)
                logger.ulog("done", done)
                break

        if step % 10 == 0 and step >= 10:
            logger.save_ulog()


if __name__ == '__main__':
    # ASK THE USER WHAT KIND OF RUN SHOULD BE DONE -----------------------------------------------------------------
    messageRL = " * Press (y) to continue RL from the saved data \n " + \
                "  Press (n) to discard saved data \n "
    messageSRL = " * Press (y) to continue stateRL from the saved data \n " + \
                 "  Press (n) to discard saved data \n "

    # Create input function which switches function depending on the python version"""
    input_fuction = raw_input if sys.version_info[0] < 3 else input

    # Ask the user for input each time you start a session
    continue_rl  = True if input_fuction(messageRL).lower() == 'y' else False
    continue_srl = True if input_fuction(messageSRL).lower() == 'y' else False

    # SETUP THE ENVIRONMENT ------------------------------------------------------------------------------------------
    # set up node
    rospy.init_node('husarion_maze_qlearn', anonymous=True, log_level=rospy.WARN)
    # Create reset odom command
    reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=10)
    # Create the Gym environment
    env = gym.make('HusarionGetToPosTurtleBotPlayGround-v0')
    rospy.loginfo("Gym environment done")
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('rosbot_srl')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    # CALL THE MAIN TRAINING LOOP ------------------------------------------------------------------------------------
    main(continue_rl, continue_srl, env)
