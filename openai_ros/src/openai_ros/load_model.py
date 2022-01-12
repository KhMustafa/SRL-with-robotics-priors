#!/usr/bin/env python

import rospy, tf
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import *
import re
from gazebo_ros import gazebo_interface

class model_control:
    def __init__(self):
        print("Wait for services")
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        print("Got it.")
        self.model_delete = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.model_spawn  = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    def spawn_model(self, modeldir, model_name, robot_namespace, position, orientation):
        with open(modeldir, "r") as f:
            model_xml = f.read()
        model_xml = re.sub("<\s*mesh\s+filename\s*=\s*([\"|'])package://","<mesh filename=\g<1>model://", model_xml)
        
        orient = Quaternion()
        quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        orient.x = quaternion[0]
        orient.y = quaternion[1]
        orient.z = quaternion[2]
        orient.w = quaternion[3]

        # calculate pose and spawn the model
        initial_pose = Pose(Point(x=position[0], y=position[1], z=position[2]), orient)
        print(initial_pose)
        self.model_spawn(model_name, model_xml, robot_namespace, initial_pose, "world")

    def delete_model(self, model_name):
        self.model_delete(model_name=model_name)
