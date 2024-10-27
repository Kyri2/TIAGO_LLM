#!/usr/bin/env python3
import rospy,json,os,time,sys,cv2,copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import image_query
from tf.transformations import euler_from_quaternion
from tf.transformations import *
from tf2_geometry_msgs import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from move_package.msg import ObjectPoseList, ObjectPose
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectoryPoint
from noactionlib import publish_trajectories,create_trajectory,create_action_goal,publish_reverse_trajectories
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionGoal
from tf.transformations import quaternion_from_euler
from openai import OpenAI
import image_query
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from back_to_home import t_publish_trajectories,t_create_trajectory,t_create_action_goal
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest


class ObjectGraspPlanner:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('object_grasp_planner', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "arm_torso"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.bridge = CvBridge()

        
        self.move_group.set_planner_id("RRTConnectkConfigDefault")  
        self.move_group.set_pose_reference_frame("base_footprint")
        self.move_group.set_max_velocity_scaling_factor(0.1)  
        self.move_group.set_max_acceleration_scaling_factor(0.1) 
        self.move_group.set_planning_time(20.0)  
        self.move_group.allow_replanning(True)  
        self.move_group.set_num_planning_attempts(10)  

        self.torso_pub = rospy.Publisher('/torso_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
        self.arm_pub = rospy.Publisher('/arm_controller/follow_joint_trajectory/goal', FollowJointTrajectoryActionGoal, queue_size=1)
        self.end_effector_link = self.move_group.get_end_effector_link()
        rospy.loginfo(f"End effector link: {self.end_effector_link}")

        self.object_cor_record = []
        self.image_saver = rospy.Subscriber("/yolo_processed_frame", Image, self.image_callback)
        self.save_dir = "/home/ysx3/robot_gpt/src/move_package/scripts/Integration"
        self.get_image = False
        self.capture_requested = False
        self.done_flag = False
        self.object_pose_sub = rospy.Subscriber("/object_pose", ObjectPoseList, self.object_pose_callback)
        self.pose = None
        self.need_prepare = True
        rospy.loginfo("Object Grasp Planner has been initialized.")

    def add_response_to_prompt(self,prompt, content):
        content = {"role":"assistant", "content":content}
        return prompt.append(content)


    def prompt_condition(self,file):
        with open(file, 'r', encoding='utf-8') as file:
            content = file.read()
        prompt = [{"role":"user", "content":content}]
        return prompt

    def pre_setup(self):
        prompt = []
        setup_file_list = ['gpt_prompt_constraint','robot_action_lib','planning_example']
        for file in setup_file_list:
            with open(file, 'r', encoding='utf-8') as files:
                content = files.read()
            prompt.append({"role":"user", "content":content})
            prompt = self.gpt(prompt)
        return prompt


    def gpt(self,prompt):
        client = OpenAI(
        api_key="your_api_key",
        )

        completion = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=prompt,
            temperature=0,
            # response_format="json"
        )
        response = completion.choices[0].message.content
        self.add_response_to_prompt(prompt, response)
        return prompt
    
    def modify_teddy_bear_string(self,input_string):
        if input_string.startswith("teddy_bear_"):
            return "teddy bear" + input_string[10:]
        return input_string

    def extract_action_object(self,instruction):
        action, param = instruction.split('(') 
        object = param.rstrip(')').strip()
        return action, object
    
    def gpt_trail(self,instruction,img):
        add_offset = False
        obj_detected_name = ""
        for item in self.object_cor_record:
            obj_detected_name += item['name']
            obj_detected_name += " "
        print(obj_detected_name)
        prompt = self.pre_setup()
        limit = image_query.prompt_condition('gpt_img_prompt_constraint')
        limit += f"User's instruction:{instruction}"
        environment_description = image_query.gpt_image(img, limit)
        print(environment_description)
        environment_description += f"ONLY USE Object DETECTED: {obj_detected_name}, for example, if cup_1 is in Object detected, then ONLY USE cup_1 in the instructiona as the object name rather then merely cup."
        environment_description = {"role":"user", "content":environment_description}
        prompt.append(environment_description)
        prompt = self.gpt(prompt)
        prompt.append({"role":"user", "content":f"now start working,don't say anything else other than the answer in the asked format. instruction: {instruction}"})
        prompt = self.gpt(prompt)
        json_content = prompt[-1]['content'].strip('```json\n').strip('```')
        print(json_content)
        try:
            parsed_json = json.loads(json_content)
            for i in range(len(parsed_json['task_sequence'])):
                print(parsed_json['task_sequence'][i])
                self.capture_requested = True
                self.get_image = False
                rospy.loginfo("Capturing image...")
                rospy.sleep(2.0)
                image_path = "/home/ysx3/robot_gpt/src/move_package/scripts/Integration/captured_image.jpg"
                while not self.get_image:
                    try:
                        if os.path.exists(image_path):
                            image = image_path
                            self.get_image = True
                        else:
                            raise FileNotFoundError("Image not found")
                    except Exception as e:
                        rospy.logerr(f"Error capturing image: {e}")
                action,obj = self.extract_action_object(parsed_json['task_sequence'][i])
                if obj == "teddy_bear_0" or obj == "teddy_bear_1":
                    obj = self.modify_teddy_bear_string(obj)
                if not add_offset:
                    target_orientation = quaternion_from_euler(0, np.pi/2, 0)  # Pitch = 90 degrees
                    # self.pose.pose.orientation = Quaternion(*target_orientation)
                    # self.pose.pose.position.z += 0.18 #Gripper offset
                    # self.pose.pose.position.x += 0.03 #Gripper offset
                    # print(self.pose)
                    # add_offset = True
                    for item in self.object_cor_record:
                        print(item['name'])
                        item_pose = item['pose']
                        item_pose.pose.pose.position.z += 0.18
                        item_pose.pose.pose.position.x += 0.03
                        item_pose.pose.pose.orientation = Quaternion(*target_orientation)
                    add_offset = True
                if obj != "":
                    self.pose = [item['pose'] for item in self.object_cor_record if item['name'] == obj]
                    self.pose = self.pose[0].pose
                print(self.pose)
                # if self.need_prepare:
                #     self.prepare_by_trajectory()
                #     self.need_prepare = False
                if action == 'move_hand':
                    self.pose.pose.position.z += 0.12
                    self.move_hand(self.pose, "Moving to pre-grasp position")
                    self.pose.pose.position.z -= 0.10
                elif action == 'release_object':
                    self.open_gripper()
                elif action == 'grasp_object':
                    self.move_hand(self.pose, "Moving to grasp position")
                    rospy.sleep(2.0)
                    self.close_gripper()
                elif action == 'detach_from_plane':
                    self.pose.pose.position.z += 0.10
                    self.move_hand(self.pose, "Lifting object")
                    self.pose.pose.position.z -= 0.10
                elif action == 'back_to_home':
                    self.back_to_home()
                if i == len(parsed_json['task_sequence']) - 1:
                    self.done_flag = True
                time.sleep(1)
        except json.JSONDecodeError:
            print("Response was not a valid JSON:", json_content)
        # self.end_by_trajectory()


    def object_pose_callback(self, msg):
        self.capture_requested = True
        self.done_flag = True
        while(self.capture_requested):
            rospy.logwarn("Waiting for image capture...")
            rospy.sleep(3.0)
        if self.done_flag:
            self.object_cor_record = []
        for obj in msg.objects:
            rospy.loginfo(f"Object: {obj.name}")
            self.object_cor_record.append({"name": obj.name, "pose": obj})
        instruction = input("Enter the instruction: ")
        planner.gpt_trail(instruction,'captured_image.jpg')
        
    def image_callback(self, msg):
        if self.capture_requested: 
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                filename = os.path.join(self.save_dir, "captured_image.jpg")
                cv2.imwrite(filename, cv_image)
                rospy.loginfo(f"Image saved as {filename}")
                self.capture_requested = False
            except Exception as e:
                rospy.logerr(f"Error saving image: {e}")

    def prepare_by_trajectory(self):
        rospy.loginfo("Preparing for grasping by executing pregrasp trajectory")
        publish_trajectories(self.torso_pub,self.arm_pub)
        execution_time = 12.0
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < execution_time:
            rospy.sleep(0.1)  
        rospy.loginfo("Pregrasp trajectory execution completed")

    def end_by_trajectory(self):
        rospy.loginfo("Preparing for grasping by executing pregrasp trajectory")
        publish_reverse_trajectories(self.torso_pub,self.arm_pub)
        execution_time = 12.0
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < execution_time:
            rospy.sleep(0.1)  
        rospy.loginfo("Pregrasp trajectory execution completed")

            

    def plan_grasp(self, object_pose):
        pose_stamped = object_pose.pose
        width = object_pose.width
        height = object_pose.height
        name = object_pose.name
        target_orientation = quaternion_from_euler(0, np.pi/2, 0)  # Pitch = 90 degrees
        pose_stamped.pose.orientation = Quaternion(*target_orientation)
        print(pose_stamped)
      
        if pose_stamped.header.frame_id != "base_footprint":
            rospy.logwarn(f"Received pose in frame {pose_stamped.header.frame_id}. Expected base_footprint. Attempting to transform.")
        
        x, y, z = pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z
        orientation = pose_stamped.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w], 'ryxz')

        rospy.loginfo(f"Planning to grasp object at position: ({x:.2f}, {y:.2f}, {z:.2f})")
        rospy.loginfo(f"Object orientation (roll, pitch, yaw): ({roll:.2f}, {pitch:.2f}, {yaw:.2f})")
        rospy.loginfo(f"Object width: {width:.2f}m, height: {height:.2f}m")

        pose_stamped.pose.position.z += 0.18 #Gripper offset
        pose_stamped.pose.position.x += 0.05
        pre_grasp_pose = copy.deepcopy(pose_stamped)
        pre_grasp_pose.pose.position.z += 0.15

        self.move_hand(pre_grasp_pose, "Moving to pre-grasp position")

        
        ask_for_permission = input("Do you want to continue to grasp the object? (y/n): ")
        if ask_for_permission.lower() == 'y':
            self.move_hand(pose_stamped, "Moving to grasp position")
            self.close_gripper() # close gripper, should take object width as args

            print(pose_stamped)

            
            lift_pose = pose_stamped
            lift_pose.pose.position.z += 0.1  
            self.move_hand(lift_pose, "Lifting object")
        
        drop = input("Do you want to drop the object? (y/n): ")
        if drop.lower() == 'y':
            self.open_gripper()
            rospy.loginfo("Object dropped")
        

        rospy.loginfo("Grasp sequence completed.")


    def move_hand(self, pose_stamped, description):
        rospy.loginfo(f"{description}")
        self.move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose(self.end_effector_link).pose
        target_pose = pose_stamped.pose
        target_orientation = current_pose.orientation

        # Define the three stages
        lift_pose = self.create_pose_stamped(
            x=current_pose.position.x,
            y=current_pose.position.y,
            z=max(current_pose.position.z, target_pose.position.z),
            orientation=target_orientation
        )

        xy_pose = self.create_pose_stamped(
            x=target_pose.position.x,
            y=target_pose.position.y,
            z=lift_pose.pose.position.z,
            orientation=target_orientation
        )

        final_pose = self.create_pose_stamped(
            x=target_pose.position.x,
            y=target_pose.position.y,
            z=target_pose.position.z,
            orientation=target_orientation
        )

        stages = [
            (lift_pose, "Lifting"),
            (xy_pose, "Moving in XY plane"),
            (final_pose, "Descending to target")
        ]

        for stage_pose, stage_description in stages:
            if stage_description == "Moving in XY plane":
                waypoints = []
                start_pose = self.move_group.get_current_pose().pose
                waypoints.append(copy.deepcopy(start_pose))
                waypoints.append(xy_pose.pose)

                (plan_msg, fraction) = self.move_group.compute_cartesian_path(
                    waypoints,
                    0.01,
                    0.0
                )

                if fraction == 1.0:
                    rospy.loginfo(f"Cartesian path computed for {stage_description}. Executing...")
                    self.move_group.execute(plan_msg, wait=True)
                    rospy.loginfo(f"Successfully executed {stage_description}")
                    self.move_group.clear_pose_targets()
                else:
                    rospy.logwarn(f"Cartesian path planning failed for {stage_description}")
                    self.move_group.clear_pose_targets()
                    return False
            else:
                for attempt in range(3):
                    rospy.loginfo(f"Planning attempt {attempt + 1} for {stage_description} - {description}")
                    self.move_group.set_pose_target(stage_pose, self.end_effector_link)

                    plan_success, plan_msg, planning_time, error_code = self.move_group.plan()

                    if plan_success:
                        rospy.loginfo(f"Planning succeeded for {stage_description}. Execution starting...")
                        execution_success = self.move_group.execute(plan_msg, wait=True)
                        if execution_success:
                            rospy.loginfo(f"Successfully executed {stage_description}")
                            self.move_group.clear_pose_targets()
                            break
                        else:
                            rospy.logwarn(f"Execution failed for {stage_description}")
                    else:
                        rospy.logwarn(f"Planning failed for {stage_description}. Error code: {error_code}")

                    if attempt == 2:
                        rospy.logerr(f"Failed to plan and execute {stage_description} after 3 attempts")
                        self.move_group.clear_pose_targets()
                        return False

                    rospy.sleep(1)

        rospy.loginfo(f"Successfully completed all stages for {description}")
        return True


    def create_pose_stamped(self, x, y, z, orientation):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.move_group.get_planning_frame()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation = orientation
        return pose_stamped



    def close_gripper(self):
        rospy.loginfo("Closing gripper")
        pub_gripper_controller = rospy.Publisher(
        '/gripper_controller/command', JointTrajectory, queue_size=1)

    # loop continues until the grippers is closed
        for i in range(10):
            trajectory = JointTrajectory()

            # call joint group for take object
            trajectory.joint_names = [
                'gripper_left_finger_joint', 'gripper_right_finger_joint']

            trajectory_points = JointTrajectoryPoint()

            # define the gripper joints configuration
            trajectory_points.positions = [0.0, 0.0]

            # define time duration
            trajectory_points.time_from_start = rospy.Duration(1.0)

            trajectory.points.append(trajectory_points)

            pub_gripper_controller.publish(trajectory)


            rospy.sleep(0.1)

    
    def open_gripper(self):
        rospy.loginfo("Closing gripper")
        pub_gripper_controller = rospy.Publisher(
        '/gripper_controller/command', JointTrajectory, queue_size=1)

    # loop continues until the grippers is closed
        for i in range(10):
            trajectory = JointTrajectory()

            # call joint group for take object
            trajectory.joint_names = [
                'gripper_left_finger_joint', 'gripper_right_finger_joint']

            trajectory_points = JointTrajectoryPoint()

            # define the gripper joints configuration
            trajectory_points.positions = [0.045, 0.045]

            # define time duration
            trajectory_points.time_from_start = rospy.Duration(1.0)

            trajectory.points.append(trajectory_points)

            pub_gripper_controller.publish(trajectory)

            # interval to start next movement
            rospy.sleep(0.1)

            rospy.sleep(1)  
    def back_to_home(self):
        rospy.loginfo("Preparing for moving back to home")
        t_publish_trajectories(self.torso_pub,self.arm_pub)
        execution_time = 12.0
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < execution_time:
            rospy.sleep(0.1)  
        rospy.loginfo("Moving back to home completed")

    def get_end_effector_pose(self, joint_names, joint_positions):

        fk_service_name = '/compute_fk'
        rospy.wait_for_service(fk_service_name)
        try:
            fk_service = rospy.ServiceProxy(fk_service_name, GetPositionFK)
            fk_request = GetPositionFKRequest()
            fk_request.header.frame_id = self.move_group.get_planning_frame()
            fk_request.fk_link_names = [self.end_effector_link]
            fk_request.robot_state.joint_state.name = joint_names
            fk_request.robot_state.joint_state.position = joint_positions

            fk_response = fk_service(fk_request)
            if fk_response.error_code.val == fk_response.error_code.SUCCESS:
                pose = fk_response.pose_stamped[0].pose
                return pose
            else:
                rospy.logerr("FK计算失败，错误代码：{}".format(fk_response.error_code.val))
                return None
        except rospy.ServiceException as e:
            rospy.logerr("服务调用失败: %s" % e)
            return None

    def move_robot_with_cartesian_path(self):
        rospy.loginfo("Starting move_robot_with_cartesian_path...")

        eef_link = self.end_effector_link
        reference_frame = self.move_group.get_planning_frame()


        current_pose = self.move_group.get_current_pose(eef_link).pose


        joint_names = ["torso_lift_joint", "arm_1_joint", "arm_2_joint", "arm_3_joint",
                       "arm_4_joint", "arm_5_joint", "arm_6_joint", "arm_7_joint"]
        joint_positions = [0.35, 0.1, 0.4, -1.41, 1.71, 0.43, -1.37, 1.7]


        target_pose = self.get_end_effector_pose(joint_names, joint_positions)
        if not target_pose:
            rospy.logerr("cannot get target pose")
            return

        waypoints = []
        waypoints.append(copy.deepcopy(current_pose))
        waypoints.append(copy.deepcopy(target_pose))

        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            0.01,  # eef_step
            0.0    # jump_threshold
        )

        if fraction == 1.0:
            rospy.loginfo("execute plan")
            self.move_group.execute(plan, wait=True)
            rospy.loginfo("target pose reached")
        else:
            rospy.logwarn("only calculate {:.2f}%".format(fraction * 100))
            return


        return_waypoints = []
        return_waypoints.append(copy.deepcopy(target_pose))
        return_waypoints.append(copy.deepcopy(current_pose))

        (return_plan, return_fraction) = self.move_group.compute_cartesian_path(
            return_waypoints,
            0.01,
            0.0
        )

        if return_fraction == 1.0:
            rospy.loginfo("execute return plan")
            self.move_group.execute(return_plan, wait=True)
            rospy.loginfo("return to start pose")
        else:
            rospy.logwarn("only calculate {:.2f}%".format(return_fraction * 100))

        self.move_group.clear_pose_targets()


if __name__ == '__main__':
    try:
        planner = ObjectGraspPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
