from styx_msgs.msg import TrafficLight
import h5py
from keras.models import load_model
import cv2
import tensorflow as tf
#import rospy

class TLClassifier(object):
    def __init__(self):
        # load classifier
        self.model = load_model('./model.h5')
        self.graph = tf.get_default_graph()


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        with self.graph.as_default():
            light_state = (self.model.predict(image[None,:,:,:], batch_size=1))
            #rospy.loginfo(str(light_state)+' light_state')
            if light_state < 0.5:
                return -1
            else:
                return 0
