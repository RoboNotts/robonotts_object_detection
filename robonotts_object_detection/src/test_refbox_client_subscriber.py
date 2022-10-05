#!/usr/bin/env python3

from numpy import True_
import rospy
import json
import metrics_refbox_msgs.msg
from sensor_msgs.msg import Image
import random
from metrics_refbox_msgs.msg import Command
from metrics_refbox_msgs.msg import ObjectDetectionResult, PersonDetectionResult, ActivityRecognitionResult
from metrics_refbox_msgs.msg import GestureRecognitionResult, HandoverObjectResult, ReceiveObjectResult
from metrics_refbox_msgs.msg import ClutteredPickResult, AssessActivityStateResult, ItemDeliveryResult
from metrics_refbox_msgs.msg import BoundingBox2D
from drake.msg import DrakeResults
from darknet_ros_msgs.msg import BoundingBoxes
import time
import threading

classnames = [
    "cup",
    "plate",
    "bowl",
    "towel",
    "shoes",
    "sponge",
    "bottle",
    "toothbrush",
    "toothpaste",
    "tray",
    "sweater",
    "cellphone",
    "banana",
    "medicine bottle",
    "reading glasses",
    "flashlight",
    "pill box",
    "book",
    "knife",
    "cellphone charger",
    "shopping bag",
    "keyboard"
]

gesture_names = ["nodding", "pointing", "pull_hand_in_call_someone", "shaking_head", "stop_sign", "thumbs_down", "thumbs_up", "wave_someone_away", "waving_hand"]
activity_names = ["Opening the door and walking in/out",
"Putting on a jacket",
"Touching a hot surface",
"Opening the fridge",
"Drinking water",
"Colliding against something",
"Eating food with a fork",
"Coughing or sneezing",
"Wiping a table",
"Reading a book",
"Neck roll exercise",
"Freehand exercise",
"Lying down",
"Limping",
"Talking on the phone",
"Using a computer",
"Falling down",
"Brushing teeth",
"Writing"]

class RefboxClientListener(object):
    person_box = BoundingBox2D()
    result_msg = None # The message to be sent back to the refBox
    ready_to_pub = False 
    image_saved = Image() # The currently saved Image
    
    # Initialises the node, and subscribes to the needed topics
    def __init__(self):
        rospy.init_node('refbox_client_listener') 
        rospy.Subscriber("/metrics_refbox_client/command", metrics_refbox_msgs.msg.Command, self.handle_command) # Refbox Commands
        rospy.Subscriber("/drake/bounding_boxes", DrakeResults, self.drake_bounding_boxes) # Bounding boxes from Darknet
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_bounding_boxes)
        rospy.Subscriber("/locobot/camera/color/image_raw", Image, self.update_image) # Bucky's Camera
        self.publishers = {
            #Set up publishers for our results.
            "person": rospy.Publisher("metrics_refbox_client/person_detection_result", PersonDetectionResult, queue_size=10),
            "object": rospy.Publisher("metrics_refbox_client/object_detection_result", ObjectDetectionResult, queue_size=10),
            "gesture": rospy.Publisher("metrics_refbox_client/gesture_recognition_result", GestureRecognitionResult, queue_size=10),
            "activity": rospy.Publisher("metrics_refbox_client/activity_recognition_result", ActivityRecognitionResult, queue_size=10)
        }
        self.requested_person = False
        self.requested_object = False
        rospy.spin() #Keeps the python node from terminating until it is closed by ROS
    
    #Publishes the current image, or updates the current image if there is not one.
    def update_image(self, msg):
        if self.ready_to_pub:
            self.result_msg.image = self.image_saved
            self.publishers[self.pub_type].publish(self.result_msg)
            self.ready_to_pub = False
        else:
            self.image_saved = msg
    
    def darknet_bounding_boxes(self, msg):
        for b in msg.bounding_boxes:
            if self.requested_object and b.probability >= 0.6 and b.Class == self.search_object:
                print(f"found a {b.Class} with {b.probability} certainty")
                self.res = ObjectDetectionResult()
                self.res.message_type = self.res.RESULT
                self.res.object_found = True
                self.res.box2d.min_x = b.xmin
                self.res.box2d.min_y = b.ymin
                self.res.box2d.max_x = b.xmax
                self.res.box2d.max_y = b.ymax
                self.ready_to_pub = True
                self.pub_type = "object"
                self.requested_object = False
                break
            if b.Class == "person" and b.probability >= 0.6 and self.requested_person:
                print(f"Foud a person with {b.probability} certainty")
                self.res = PersonDetectionResult()
                self.res.message_type = self.res.RESULT
                self.res.person_found = True
                self.res.box2d.min_x = b.xmin
                self.res.box2d.min_y = b.ymin
                self.res.box2d.max_x = b.xmax
                self.res.box2d.max_y = b.ymax
                self.ready_to_pub = True
                self.pub_type = "person"
                self.requested_person = False
                break

    #Takes all the bounding boxes, and formats them into the appropriate refbox message
    def drake_bounding_boxes(self, msg):
        for b in msg.results:
            # If we are looking for an object...
            if self.requested_object and classnames[b.object_class] == self.search_object:
                print(f"Found a {classnames[b.object_class]} with {b.confidence} certainty") #Debug console statement
                self.result_msg = ObjectDetectionResult()
                self.result_msg.message_type = self.result_msg.RESULT
                self.result_msg.object_found = True
                self.result_msg.result_type = 1 # 2d bounding box
                self.result_msg.box2d.min_x = b.xmin
                self.result_msg.box2d.min_y = b.ymin
                self.result_msg.box2d.max_x = b.xmax
                self.result_msg.box2d.max_y = b.ymax
                self.ready_to_pub = True
                self.pub_type = "object"
                self.requested_object = False
                break   

    # Handle a refbox command
    def handle_command(self, msg):
        print("Handling command")
        if msg.command != 1: return # This means we don't need to handle it
        if msg.task == Command.OBJECT_DETECTION:
            self.search_object = json.loads(msg.task_config)["Target object"].lower()
            print(f"Searching for a {self.search_object}")
            self.send_object_detection_result()
        elif msg.task == Command.PERSON_DETECTION:
            self.send_person_detection_result()
        elif msg.task == Command.GESTURE_RECOGNITION:
            self.send_gesture_recognition_result()
        elif msg.task == Command.ACTIVITY_RECOGNITION:
            self.send_activity_recognition_result()

    # Send the result of our object detection
    def send_object_detection_result(self):
        print("Received object request")
        self.requested_object = True
        th = threading.Thread(target=self.timeout_object)
        th.start()
        th.join()

    # As above for people
    def send_person_detection_result(self):
        print("Received person request")
        self.requested_person = True
        th = threading.Thread(target=self.timeout_person)
        th.start()
        th.join()

    # As above for gestures
    def send_gesture_recognition_result(self):
        print("Received gesture request")
        self.result_msg = GestureRecognitionResult()
        self.result_msg.message_type = self.result_msg.RESULT
        self.result_msg.gestures = random.sample(gesture_names, 2)
        time.sleep(random.uniform(0.5,5))
        self.publishers["gesture"].publish(self.result_msg)

    def send_activity_recognition_result(self):
        print("Received activity request")
        self.result_msg = ActivityRecognitionResult()
        self.result_msg.message_type = self.result_msg.RESULT
        self.result_msg.activities = random.sample(activity_names, 2)
        time.sleep(random.uniform(0.5,5))
        self.publishers["activity"].publish(self.result_msg)

    # 8 second timer for detection
    def timeout_person(self):
        st = time.time()
        while time.time() - st < 8:
            if not self.requested_person:
                return # person already found

        # waited >= 8 secs for person but not found
        self.result_msg = PersonDetectionResult()
        self.result_msg.message_type = self.result_msg.RESULT
        self.result_msg.person_found = False
        self.publishers["person"].publish(self.result_msg)
        self.requested_person = False

    # 8 second timer for detection
    def timeout_object(self):
        st = time.time()
        while time.time() - st < 8:
            if not self.requested_object:
                return # objects already found

        #NO objects found
        self.result_msg = ObjectDetectionResult()
        self.result_msg.message_type = self.result_msg.RESULT
        self.result_msg.object_found = False
        self.publishers["object"].publish(self.result_msg)
        self.requested_object = False

if __name__ == '__main__':
    client = RefboxClientListener()


