#! /usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal

def create_trajectory(joint_names, positions_list, times_from_start):
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    
    for positions, time in zip(positions_list, times_from_start):
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(time)
        trajectory.points.append(point)
    
    return trajectory

def create_action_goal(trajectory):
    goal = FollowJointTrajectoryActionGoal()
    goal.goal.trajectory = trajectory
    goal.header.stamp = rospy.Time.now()
    return goal

def publish_trajectories(torso_pub,arm_pub):

    torso_trajectory = create_trajectory(
        ["torso_lift_joint"],
        [[0.35], [0.35], [0.35],[0.35]],
        [4.0, 8.5, 13.0,18.0]
    )
    
    arm_trajectory = create_trajectory(
        ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"],
        [
            [0.20, -1.34, -0.20, 1.94, -1.57, 1.37, 0.0],
            [0.10, 0.47, -0.20, 1.56, -1.58, 0.25, 0.0],
            [0.07, 0.31, -0.05, -0.00, -1.85, -0.22, -0.05],
            [0.1, 0.4, -1.41, 1.71, 0.43, -1.37, 1.7]
        ],
        [4.0, 8.5, 13.0,18.0]
    )
    
    torso_goal = create_action_goal(torso_trajectory)
    arm_goal = create_action_goal(arm_trajectory)
    
    torso_pub.publish(torso_goal)
    arm_pub.publish(arm_goal)
    rospy.loginfo("Trajectories published")

def publish_reverse_trajectories(torso_pub,arm_pub):
    
        torso_trajectory = create_trajectory(
            ["torso_lift_joint"],
            [[0.35], [0.35], [0.35]],
            [3.0, 8.5, 10.5]
        )
        
        arm_trajectory = create_trajectory(
            ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"],
            [
                [0.1, 0.4, -1.41, 1.71, 0.43, -1.37, 1.7],
                [0.07, 0.29, -1.75, -0.02, -1.49, -0.02, 0.0],
                [0.10, 0.47, -0.20, 1.56, -1.58, 0.25, 0.0],
                # [0.20, -1.34, -0.20, 1.94, -1.57, 1.37, 0.0]

            ],
            [3.0, 8.5, 10.5]
        )
        
        torso_goal = create_action_goal(torso_trajectory)
        arm_goal = create_action_goal(arm_trajectory)
        
        torso_pub.publish(torso_goal)
        arm_pub.publish(arm_goal)
        rospy.loginfo("Trajectories published")

def main():
    global torso_pub, arm_pub
    rospy.init_node('trajectory_publisher', anonymous=True)
    
    torso_pub = rospy.Publisher('/torso_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    arm_pub = rospy.Publisher('/arm_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
    
    # 等待发布器就绪
    rospy.sleep(1)
    
    # 发布一次完整的轨迹
    publish_trajectories(torso_pub,arm_pub)
    rospy.sleep(15)
    # publish_reverse_trajectories(torso_pub,arm_pub)
    
    rospy.loginfo("Trajectory execution completed. Press Ctrl+C to exit.")
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
