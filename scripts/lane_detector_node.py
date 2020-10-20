#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import cv2
import json
import torch
import utils.agent
import numpy as np
from copy import deepcopy
import time
from time import sleep
from config.parameters import Parameters
import utils.util as util
from tqdm import tqdm
import csaps
import argparse
import glob
import os

class LaneDetector:
    def __init__(self):
        print("Init()")
        print(" Info()")
        rospy.init_node('lane_detector_node', anonymous = True)

        self.param = Parameters()
        self.lane_agent = utils.agent.Agent()
        self.lane_detector_init()

    def lane_detector_init(self):
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param("~camera_topic")
        self.result_lane_topic = rospy.get_param("~result_lane_topic")
        self.image_path = rospy.get_param("~image_path")
        self.model_path = rospy.get_param("~model_path")
        self.epoch = rospy.get_param("~epoch")
        self.loss = rospy.get_param("~loss")

        path = self.model_path+str(self.epoch)+'_'+str(self.loss)+'_'+'lane_detection_network.pkl'

        print("  Model: {0:s}".format(path))

        sub_image = rospy.Subscriber(self.camera_topic, Image, self.imageCb, queue_size=1)
    
        # Load weights
        self.lane_agent.load_weights(path)

        # Check image
        self.camera_status = False
        
        if torch.cuda.is_available():
            self.lane_agent.cuda()

        self.lane_agent.evaluate_mode()

    def imageCb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        self.camera_status = True

    def generate_result(self, confidance, offsets,instance, thresh):
        gen_start = time.monotonic()
        mask = confidance > thresh

        grid = self.param.grid_location[mask]
        offset = offsets[mask]
        feature = instance[mask]

        lane_feature = []
        x = []
        y = []
        for i in range(len(grid)):
            if (np.sum(feature[i]**2))>=0:
                point_x = int((offset[i][0]+grid[i][0])*self.param.resize_ratio)
                point_y = int((offset[i][1]+grid[i][1])*self.param.resize_ratio)
                if point_x > self.param.x_size or point_x < 0 or point_y > self.param.y_size or point_y < 0:
                    continue
                if len(lane_feature) == 0:
                    lane_feature.append(feature[i])
                    x.append([point_x])
                    y.append([point_y])
                else:
                    flag = 0
                    index = 0
                    min_feature_index = -1
                    min_feature_dis = 10000
                    for feature_idx, j in enumerate(lane_feature):
                        dis = np.linalg.norm((feature[i] - j)**2)
                        if min_feature_dis > dis:
                            min_feature_dis = dis
                            min_feature_index = feature_idx
                    if min_feature_dis <= self.param.threshold_instance:
                        lane_feature[min_feature_index] = (lane_feature[min_feature_index]*len(x[min_feature_index]) + feature[i])/(len(x[min_feature_index])+1)
                        x[min_feature_index].append(point_x)
                        y[min_feature_index].append(point_y)
                    elif len(lane_feature) < 12:
                        lane_feature.append(feature[i])
                        x.append([point_x])
                        y.append([point_y])

        gen_time = time.monotonic() - gen_start

#        print("Generate result : {0:.2f}".format(gen_time))
                
        return x, y

    def eliminate_fewer_points(self, x, y):
        eliminate_start = time.monotonic()
        # eliminate fewer points
        out_x = []
        out_y = []
        for i, j in zip(x, y):
            if len(i)>2:
                out_x.append(i)
                out_y.append(j)     

        eliminate_time = time.monotonic() - eliminate_start
        
#        print("Eliminate : {0:0.2f}".format(eliminate_time*1000))
        return out_x, out_y   

    def test(self, lane_agent, test_images, thresh = 0.35, index= -1):

        result = lane_agent.predict_lanes_test(test_images)
        torch.cuda.synchronize()
        confidences, offsets, instances = result[index]
    
        num_batch = len(test_images)

        out_x = []
        out_y = []
        out_images = []
    
        for i in range(num_batch):
            # test on test data set
            image = deepcopy(test_images[i])
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0)*255.0
            image = image.astype(np.uint8).copy()

            confidence = confidences[i].view(self.param.grid_y, self.param.grid_x).cpu().data.numpy()

            offset = offsets[i].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)
        
            instance = instances[i].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            # generate point and cluster
            raw_x, raw_y = self.generate_result(confidence, offset, instance, thresh)

            # eliminate fewer points
            in_x, in_y = self.eliminate_fewer_points(raw_x, raw_y)
                
            # sort points along y 
            in_x, in_y = util.sort_along_y(in_x, in_y)  

            result_image = util.draw_points(in_x, in_y, deepcopy(image))

            out_x.append(in_x)
            out_y.append(in_y)
            out_images.append(result_image)
        
        return out_x, out_y, out_images

    def run_detector(self):
        # Waiting for image 
        while not self.camera_status:
            print("Waiting for {0:s} camera topic...".format(self.camera_topic))
            sleep(2)

        self.thresh = self.param.threshold_point

        while not rospy.is_shutdown():
            resize_start = time.monotonic()

            resized_image = cv2.resize(self.cv_image, (512,256))/255.0
            resized_image = np.rollaxis(resized_image, axis=2, start=0)
            resize_time = time.monotonic() - resize_start

#            print("Resize : {0:0.2f}".format(resize_time * 1000))

            test_start = time.monotonic()
            _, _, ti = self.test(self.lane_agent, np.array([resized_image]), self.thresh)
            test_time = time.monotonic() - test_start

#            print("Test : {0:0.2f}".format(test_time * 1000))

            display_img = cv2.cvtColor(ti[0], cv2.COLOR_BGR2RGB)
            cv2.imshow("Detection result", display_img)
            cv2.waitKey(1)

if __name__ == '__main__':
    lane_detector = LaneDetector()
    lane_detector.run_detector()
    rospy.spin()
