from styx_msgs.msg import TrafficLight
import h5py
from keras.models import load_model
import cv2
import tensorflow as tf
import os
import rospy

#model = None
#model = load_model('/home/ashre/CarND-Capstone/ros/src/tl_classification_model/LightStatusData/model.h5')

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        current_dir = os.getcwd()
        filesDir = os.path.join(current_dir, 'light_classification', 'model.h5')
        self.model = load_model(filesDir)
        self.graph = tf.get_default_graph()
        #self.model._make_predict_function()
        #self.model.summary()
        pass


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #f = h5py.File('model_temp.h5', mode='r')
        #rospy.loginfo(str(f.attrs.get('keras_version')))
        #model = load_model('model.h5')
        #image = cv2.imread('/home/ashre/CarND-Capstone/ros/src/tl_classification_model/LightStatusData/IMG/2.jpg')

        with self.graph.as_default():
            light_state = (self.model.predict(image[None,:,:,:], batch_size=1))
            #rospy.loginfo(str(light_state)+' light_state')
            if light_state < -0.5:
                return -1
            else:
                return 0

        #return light_state
