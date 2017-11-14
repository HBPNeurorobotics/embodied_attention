#include "ros/ros.h"
#include "std_msgs/String.h"
#include "image_transport/image_transport.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "std_msgs/Float64.h"
#include "sensor_msgs/CameraInfo.h"
#include "ros_holographic/NewObject.h"

#include <math.h>
#include <algorithm>
#include <sstream>
#include <iostream>

image_transport::Publisher pubkps;
cv::Ptr<cv::SimpleBlobDetector> detector;
image_geometry::PinholeCameraModel model;
sensor_msgs::CameraInfoConstPtr ptr;
ros::Publisher pub_neck_yaw_pos;
ros::Publisher pub_neck_yaw_vel;
ros::Publisher pub_neck_pitch_pos;
ros::Publisher pub_neck_pitch_vel;
std_msgs::Float64 neck_yaw_pos;
std_msgs::Float64 neck_yaw_vel;
std_msgs::Float64 neck_pitch_pos;
std_msgs::Float64 neck_pitch_vel;
double x = 0, y = 0, thresh = 0.05;
bool set = false;
bool block = false;
ros::ServiceClient memory_client;
ros_holographic::NewObject obj;
int counter = 0;

bool compare_size(cv::KeyPoint first, cv::KeyPoint second) {
        return first.size > second.size;
}

void imageCallback2(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& nptr) {
        ptr = nptr;
        set = true;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        // convert ros image to cv image
        cv::Mat mat = cv_bridge::toCvShare(msg, "bgr8")->image;
        // convert to gray scale image
        cv::cvtColor(mat, mat, CV_BGR2GRAY);
        // make too dark pixels black
        cv::threshold(mat, mat, 100, 255, 0);
        // resize
        cv::resize(mat, mat, cv::Size(msg->width, msg->height));
        // detect keypoints
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(mat, keypoints);
        // make keypoints visible
        cv::Mat kps;
        cv::drawKeypoints(mat, keypoints, kps, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::sort(keypoints.begin(), keypoints.end(), compare_size);
        // convert to gray scale image
        cv::cvtColor(kps, kps, CV_BGR2GRAY);
        // publish image
        sensor_msgs::ImageConstPtr ikps;
        ikps = cv_bridge::CvImage(std_msgs::Header(), "mono8", kps).toImageMsg();
        pubkps.publish(ikps);    
        if (keypoints.size() && set && !block) {
                block = true;
                ROS_ERROR_STREAM("" << model.fromCameraInfo(ptr));
                cv::Point3d p = model.projectPixelTo3dRay(keypoints[0].pt);
                double dx, dy;
                dx = atan(-p.x);
                dy = atan(-p.y / sqrt(p.x * p.x + 1)); // cartesian to polar coordinates
                ROS_ERROR("keypoints: %f %f", keypoints[0].pt.x, keypoints[0].pt.y);
                ROS_ERROR("differences: %f %f", dx, dy);
//                if (abs(dx) > thresh || abs(dy) > thresh) {
                        // send new positions
                        x += dx;
                        y += dy;
                        std::cout << "going to look at: " << x << ", " << y << "\n";
                        neck_yaw_pos.data = x;
                        neck_pitch_pos.data = y;
                        pub_neck_yaw_pos.publish(neck_yaw_pos);
                        pub_neck_pitch_pos.publish(neck_pitch_pos);

                        // send info to memory
                        obj.request.X = x;
                        obj.request.Y = y;
                        std::stringstream sstm;
                        sstm << "label no " << counter;
                        obj.request.Label = sstm.str();
                        std::cout << "going to label label " << counter << " at: " << x << ", " << y << "\n";
                        counter++;
                        memory_client.call(obj);
//                }
                ros::Duration(5).sleep();

                block = false;
        }
}

int main(int argc, char **argv) {
        // initialize blob detector
        cv::SimpleBlobDetector::Params params;
        params.minThreshold = 35;
        params.maxThreshold = 255;
        params.minDistBetweenBlobs = 0.0f;
        params.filterByInertia = false;
        params.filterByConvexity = false;
        params.filterByColor = true;
        params.blobColor = 255;
        params.filterByCircularity = false;
        params.filterByArea = true;
        params.minArea = 50;
        detector = cv::SimpleBlobDetector::create(params);
        
        // initialize neck msgs
        neck_yaw_pos.data = x;
        neck_yaw_vel.data = 0.25;
        neck_pitch_pos.data = y;
        neck_pitch_vel.data = 0.25;
        
        // initialize ros node
        ros::init(argc, argv, "get_rays");
        ros::NodeHandle nh;
        image_transport::ImageTransport it(nh);
        
        // be able to publish to /kps_map and robot neck controls
        pubkps = it.advertise("/kps_map",1);
        pub_neck_yaw_pos = nh.advertise<std_msgs::Float64>("/robot/neck_yaw/pos", 1); 
        pub_neck_yaw_vel = nh.advertise<std_msgs::Float64>("/robot/neck_yaw/vel", 1); 
        pub_neck_pitch_pos = nh.advertise<std_msgs::Float64>("/robot/neck_pitch/pos", 1); 
        pub_neck_pitch_vel = nh.advertise<std_msgs::Float64>("/robot/neck_pitch/vel", 1); 
        
        // publish first states
        pub_neck_yaw_pos.publish(neck_yaw_pos);
        pub_neck_yaw_vel.publish(neck_yaw_vel);
        pub_neck_pitch_pos.publish(neck_pitch_pos);
        pub_neck_pitch_vel.publish(neck_pitch_vel);
        ros::Duration(10).sleep();
        
        // subscribe and do stuff
        image_transport::Subscriber sub = it.subscribe("/saliency_map", 1, imageCallback);
        image_transport::CameraSubscriber csub = it.subscribeCamera("/icub_model/left_eye_camera/image_raw", 1, imageCallback2);

        // initialize service client
        memory_client = nh.serviceClient<ros_holographic::NewObject>("new_object");

        ros::spin();
        
        return 0;
}
