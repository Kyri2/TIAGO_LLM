#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from ultralytics import YOLO
import message_filters
import cv2
import tf2_ros
import tf2_geometry_msgs
from move_package.msg import ObjectPoseList, ObjectPose

class ObjectDetector:
    def __init__(self, test=False):
        rospy.loginfo("Initializing ObjectDetector")
        if test:
            rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        rospy.loginfo("CvBridge initialized")

        # Load YOLO model (YOLOv8)
        rospy.loginfo("Loading YOLO model...")
        try:
            self.model = YOLO('yolov8x-worldv2.pt')  # or your preferred YOLOv8 model
            # self.model.set_classes(['bottle'])
            rospy.loginfo("YOLO model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Error loading YOLO model: {e}")
            raise
    
        # Subscribers
        rospy.loginfo("Setting up subscribers...")
        self.image_sub = message_filters.Subscriber("/xtion/rgb/image_rect_color", Image)
        self.pointCloud_sub = message_filters.Subscriber("/xtion/depth_registered/points", PointCloud2)
        self.processed_frame_pub = rospy.Publisher("/yolo_processed_frame", Image, queue_size=1)

        # Synchronizer
        rospy.loginfo("Setting up message synchronizer...")
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pointCloud_sub], 10, 0.1)
        self.ts.registerCallback(self.synchronized_callback)

        # Publisher
        rospy.loginfo("Setting up publisher...")
        self.object_pose_pub = rospy.Publisher("/object_pose", ObjectPoseList, queue_size=10)

        # set rate(30fps)
        self.rate = rospy.Rate(30)

        rospy.loginfo("Setting up tf buffer and listener...")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("ObjectDetector initialization complete")
        self.take_pic = False
    
    def take_picture(self):
        self.take_pic = True

    def transform_pose(self, pose_stamped, target_frame):
        rospy.loginfo(f"Transforming pose to frame: {target_frame}")
        try:
            transform = self.tf_buffer.lookup_transform(target_frame,
                                                        pose_stamped.header.frame_id,
                                                        rospy.Time(0),
                                                        rospy.Duration(1.0))
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            rospy.loginfo("Pose transformed successfully")
            return transformed_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Transform failed: {e}")
            return None

    def publish_object_pose(self, pose_msg):
        rospy.loginfo("Publishing object pose...")
        # Transform to arm base frame
        transformed_pose = self.transform_pose(pose_msg, "base_footprint")
        if transformed_pose:
            self.object_pose_pub.publish(transformed_pose)
            rospy.loginfo(f"Published transformed object pose: {transformed_pose.pose.position}")
        else:
            rospy.logwarn("Failed to transform object pose")

    def synchronized_callback(self, image_msg, pointCloud_msg):
        rospy.loginfo("Received synchronized image and point cloud")
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.loginfo("Image converted to OpenCV format")

            # Perform YOLO detection
            results = self.model(cv_image)
            rospy.loginfo(f"YOLO detection complete. Found {len(results[0].boxes)} objects")

            # Visualize results on the frame
            # annotated_frame = results[0].plot()
            annotated_frame = cv_image.copy()
            
            object_pose_list = ObjectPoseList()
            object_pose_list.header = pointCloud_msg.header



            # Process YOLO results for object detection
            for r in results:
                boxes = r.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if conf > 0.49:  # Confidence threshold
                        # rospy.loginfo(f"Processing detected object {i+1}")
                        class_name = self.model.names[int(cls)]
                        label = f"{class_name}_{i}"
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Calculate center of bounding box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        w = x2 - x1
                        h = y2 - y1
                        item_width_info = (x1, y1, w, h)
                        # Estimate object size
                        object_width, object_height = self.estimate_object_size(cv_image, item_width_info)

                        # Get 3D coordinates from point cloud
                        point_3d = self.get_3d_point(pointCloud_msg, cx, cy)

                        if point_3d:
                            # rospy.loginfo(f"3D point found for object {i+1}")
                            # Create and publish PoseStamped message
                            pose_msg = PoseStamped()
                            pose_msg.header = pointCloud_msg.header
                            pose_msg.pose.position.x = point_3d[0]
                            pose_msg.pose.position.y = point_3d[1]
                            pose_msg.pose.position.z = point_3d[2]
                            
                            pose_msg.pose.orientation.x = 0.0
                            pose_msg.pose.orientation.y = 0.0
                            pose_msg.pose.orientation.z = 0.0
                            pose_msg.pose.orientation.w = 1.0



                            transformed_pose = self.transform_pose(pose_msg, "base_footprint")
                            if transformed_pose:
                                world_coordinate_obj = ObjectPose()
                                world_coordinate_obj.name = label
                                world_coordinate_obj.pose = transformed_pose
                                world_coordinate_obj.width = object_width
                                world_coordinate_obj.height = object_height
                                object_pose_list.objects.append(world_coordinate_obj)
                                # rospy.loginfo(f"Added object of class {label} to the list")
            if object_pose_list.objects:
                self.object_pose_pub.publish(object_pose_list)
                # rospy.loginfo(f"Published list of {len(object_pose_list.objects)} object poses")
            else:
                rospy.loginfo("No objects detected in this frame")
            cv2.imshow("YOLO Detection", annotated_frame)
            cv2.waitKey(1)  # 使用1毫秒等待时间以避免阻塞
            processed_image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
                # Set the header to match the original image
            processed_image_msg.header = image_msg.header
                # Publish the processed frame
            self.processed_frame_pub.publish(processed_image_msg)
            if self.take_pic:
                cv2.imwrite("/home/ysx3/robot_gpt/src/move_package/scripts/Integration/captured_image.jpg", annotated_frame)
                self.take_pic = False
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in synchronized_callback: {e}")

    def get_3d_point(self, cloud_msg, x, y):
        rospy.loginfo(f"Getting 3D point for coordinates ({x}, {y})")
        width = cloud_msg.width
        height = cloud_msg.height

        if x < 0 or x >= width or y < 0 or y >= height:
            rospy.logwarn(f"Coordinates ({x}, {y}) are outside the point cloud boundaries.")
            return None

        points = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=[(x, y)])
        point_3d = next(points, None)

        if point_3d is None:
            rospy.logwarn(f"No valid 3D point at coordinates ({x}, {y})")
            return None

        rospy.loginfo(f"Found 3D point: {point_3d}")
        return point_3d

    def estimate_object_size(self, d_image, bbox):
        rospy.loginfo("Estimating object size")
        x, y, w, h = map(int, bbox)
        roi = d_image[y:y + h, x:x + w]

        # assume rectangle
        object_width = np.median(roi[:, 0]) - np.median(roi[:, -1])
        object_height = np.median(roi[0, :]) - np.median(roi[-1, :])

        # transform to meters
        object_width_m = object_width / 1000
        object_height_m = object_height / 1000

        rospy.loginfo(f"Estimated object size: {object_width_m}m x {object_height_m}m")
        return object_width_m, object_height_m

if __name__ == '__main__':
    rospy.loginfo("Starting object_detector node")
    try:
        detector = ObjectDetector(test=True)
        rospy.loginfo("ObjectDetector created, entering spin()")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
