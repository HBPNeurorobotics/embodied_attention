#!/usr/bin/env python
PKG = 'saliency'
import roslib; roslib.load_manifest(PKG)

import cv2 as cv
import numpy as np
import skimage.transform
from keras.models import load_model
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys

height = 168
width = 224

def rescale(img, height, width):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    elif len(img.shape) == 2:
        h, w = img.shape

    if (float(height)/h) > (float(width)/w):
        img = skimage.transform.resize(img, (height, w*height/h),
                                       order=3, preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h*width/w, width),
                                       order=3, preserve_range=True)

    if len(img.shape) == 3:
        h, w, _ = img.shape
    elif len(img.shape) == 2:
        h, w = img.shape

    img = img[h//2-(height/2):h//2+(height/2), w//2-(width/2):w//2+(width/2)]

    if len(img.shape) == 3:
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68

    return img

def get_data():
    tmp = cv.imread('image.jpg', 1)
    tmp = rescale(tmp, height, width)
    tmp = tmp.astype(np.float32)
    tmp = tmp.transpose(2, 0, 1)

    return tmp

def use_model(img):
    model = load_model('/disk/no_backup/lesi/Saliency/model.hdf5')
    model.load_weights('/disk/no_backup/lesi/Saliency/weights.hdf5')
    tmp = model.predict(img[np.newaxis, :, :, :])

    return tmp


class Saliency():
    def __init__(self):
        self.model = load_model('/disk/no_backup/jkaiser/saliency/Saliency/model.hdf5')
        self.model.load_weights('/disk/no_backup/jkaiser/saliency/Saliency/weights.hdf5')
        image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback, queue_size=1,  buff_size=2**24)
        self.saliency_pub = rospy.Publisher("/saliency_map", Image, queue_size=1)

    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            frame = CvBridge().imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e

        tmp = rescale(frame, height, width)
        tmp = tmp.astype(np.float32)
        tmp = tmp.transpose(2, 0, 1)
        res = self.model.predict(tmp[np.newaxis, :, :, :])

        res = res.squeeze()
        res = (res - res.min()) / (res.max() - res.min()) * 255.
        res = np.uint8(res)
        res_msg = CvBridge().cv2_to_imgmsg(res, "mono8")

        self.saliency_pub.publish(res_msg)



def main(args):
    rospy.init_node("saliency")
    saliency = Saliency()
    rospy.spin()
    # img = get_data()

    # res = use_model(img)

    # make_plot(img, res)

if __name__ == '__main__':
    main(sys.argv)
