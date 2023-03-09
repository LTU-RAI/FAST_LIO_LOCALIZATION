#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import _thread
import time

import numpy as np
import rospy
import tf
import tf.transformations
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from nav_msgs.msg import Odometry

import scipy.signal as ss
from collections import deque
from statistics import median


cur_odom_to_baselink = None
cur_map_to_odom = None
current_time = 0.0
last_time = 0.0

## Keep the last five poses
velocities_x = deque(maxlen=15)
velocities_y = deque(maxlen=15)
velocities_z = deque(maxlen=15)

MAX_VEL = 2.5

def velocity_filter(xyz, last_xyz):
    ## Global deques
    global velocities_x, velocities_y, velocities_z
    global current_time, last_time
    ## If last pose is empty
    if last_xyz is None:
        last_xyz = xyz
        return 0.0, 0.0, 0.0
    ## Calculate time difference
    dt = current_time - last_time
    ## Calculate dx
    dx, dy, dz = xyz[0] - last_xyz[0], xyz[1] - last_xyz[1], xyz[2] - last_xyz[2]
    ## Calculate velocity
    vx, vy, vz = dx/dt, dy/dt, dz/dt
    ## Keep velocities under threshold
    if abs(vx) < MAX_VEL:
        velocities_x.append(vx)
    if abs(vy) < MAX_VEL:
        velocities_y.append(vy)
    if abs(vz) < MAX_VEL:
        velocities_z.append(vz)
    ## Median scipy filter
    median_speed_x = ss.medfilt(velocities_x)
    median_speed_y = ss.medfilt(velocities_y)
    median_speed_z = ss.medfilt(velocities_z)

    return np.sum(median_speed_x)/len(median_speed_x), \
                np.sum(median_speed_y)/len(median_speed_y), \
                    np.sum(median_speed_z)/len(median_speed_z)
    

def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def transform_fusion():
    global cur_odom_to_baselink, cur_map_to_odom
    global last_time, current_time
    last_xyz = None
    br = tf.TransformBroadcaster()
    while True:
        time.sleep(1 / FREQ_PUB_LOCALIZATION)

        # TODO 这里注意线程安全
        cur_odom = copy.copy(cur_odom_to_baselink)
        if cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat(cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)

        br.sendTransform(tf.transformations.translation_from_matrix(T_map_to_odom),
                         tf.transformations.quaternion_from_matrix(T_map_to_odom),
                         rospy.Time.now(),
                         'camera_init', 'map')
        if cur_odom is not None:
            # 发布全局定位的odometry
            localization = Odometry()

            T_odom_to_base_link = pose_to_mat(cur_odom)

            current_time = cur_odom.header.stamp.to_sec()

            # 这里T_map_to_odom短时间内变化缓慢 暂时不考虑与T_odom_to_base_link时间同步
            T_map_to_base_link = np.matmul(T_map_to_odom, T_odom_to_base_link)
            xyz = tf.transformations.translation_from_matrix(T_map_to_base_link)
            quat = tf.transformations.quaternion_from_matrix(T_map_to_base_link)

            # Calculate velocities
            if (xyz != last_xyz).all():
                vx, vy, vz = velocity_filter(xyz, last_xyz)

            last_xyz = xyz
            last_time = current_time

            localization.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
            # localization.twist = cur_odom.twist
            localization.twist.twist.linear.x = vx
            localization.twist.twist.linear.y = vy 
            localization.twist.twist.linear.z = vz

            localization.header.stamp = cur_odom.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'body'
            # rospy.loginfo_throttle(1, '{}'.format(np.matmul(T_map_to_odom, T_odom_to_base_link)))
            pub_localization.publish(localization)


def cb_save_cur_odom(odom_msg):
    global cur_odom_to_baselink
    cur_odom_to_baselink = odom_msg


def cb_save_map_to_odom(odom_msg):
    global cur_map_to_odom
    cur_map_to_odom = odom_msg


if __name__ == '__main__':
    # tf and localization publishing frequency (HZ)
    FREQ_PUB_LOCALIZATION = 20

    rospy.init_node('transform_fusion')

    last_time = rospy.get_time()

    rospy.loginfo('Transform Fusion Node Inited...')

    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)
    rospy.Subscriber('/map_to_odom', Odometry, cb_save_map_to_odom, queue_size=1)

    pub_localization = rospy.Publisher('/localization', Odometry, queue_size=1)

    # 发布定位消息
    _thread.start_new_thread(transform_fusion, ())

    rospy.spin()
