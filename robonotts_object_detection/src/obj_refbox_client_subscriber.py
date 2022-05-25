#!/usr/bin/env python3

import rospy
import metrics_refbox_msgs.msg
from sensor_msgs.msg import Image
from metrics_refbox_msgs.msg import Command
from metrics_refbox_msgs.msg import Command
from metrics_refbox_msgs.msg import ObjectDetectionResult, PersonDetectionResult, ActivityRecognitionResult
from metrics_refbox_msgs.msg import GestureRecognitionResult, HandoverObjectResult, ReceiveObjectResult
from metrics_refbox_msgs.msg import ClutteredPickResult, AssessActivityStateResult, ItemDeliveryResult
from metrics_refbox_msgs.msg import BoundingBox2D
from darknet_ros_msgs.msg import BoundingBoxes
import time

class RefboxClientListener(object):
    person_box = BoundingBox2D()
    person_prob = 0

    image_saved = Image()

    def __init__(self):
        rospy.init_node('refbox_client_listener')
        rospy.Subscriber("/metrics_refbox_client/command", metrics_refbox_msgs.msg.Command, self.handle_command)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.update_bounding_boxes)
        rospy.Subscriber("/darknet_ros/detection_image", Image, self.update_image)
        self.person_publisher = rospy.Publisher("metrics_refbox_client/person_detection_result", PersonDetectionResult, queue_size=10)
        #self.send_person_detection_result()
        self.requested_person = False
        rospy.spin()
    
    def update_image(self, msg):
        self.image_saved = msg
    def update_bounding_boxes(self, msg):
        for b in msg.bounding_boxes:
            if b.Class == "person" and b.probability >= 0.6:
                print(f"Found a person with {b.probability} certainty")
                self.person_box.min_x = b.xmin
                self.person_box.min_y = b.ymin
                self.person_box.max_x = b.xmax
                self.person_box.max_y = b.ymax
                self.person_prob = b.probability
                if self.requested_person:
                    res = PersonDetectionResult()
                    res.message_type = res.RESULT
                    res.person_found = True
                    res.box2d = self.person_box
                    res.image = self.image_saved
                    self.person_publisher.publish(res)
                    print("Published person")
                    self.requested_person = False

    def handle_command(self, msg):
        print("Handling command")
        if msg.task == Command.OBJECT_DETECTION:
            self.send_object_detection_result()
        elif msg.task == Command.PERSON_DETECTION:
            self.send_person_detection_result()
        elif msg.task == Command.GESTURE_RECOGNITION:
            self.send_gesture_recognition_result()

    def send_object_detection_result(self):
        pass

    def send_person_detection_result(self):
        print("Received person request")
        self.requested_person = True

if __name__ == '__main__':
    client = RefboxClientListener()


