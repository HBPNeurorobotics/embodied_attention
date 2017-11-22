#include "ros/ros.h"
#include "std_msgs/String.h"
#include "image_transport/image_transport.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Float64.h"
#include "sensor_msgs/CameraInfo.h"
#include "ros_holographic/NewObject.h"

#include <math.h>
#include <algorithm>
#include <sstream>
#include <iostream>

sensor_msgs::JointState joint_cmd;
ros::Publisher joint_cmd_pub;

image_transport::Publisher keypoint_pub;
image_transport::Publisher roi_pub;
cv::Ptr<cv::SimpleBlobDetector> detector;
image_geometry::PinholeCameraModel model;
sensor_msgs::ImageConstPtr original_img;
sensor_msgs::CameraInfoConstPtr original_img_info;

std_msgs::Float64 neck_yaw_pos;
std_msgs::Float64 neck_pitch_pos;
ros::Publisher neck_yaw_pos_pub;
ros::Publisher neck_pitch_pos_pub;

double x = 0, y = 0, thresh = 0.05;
bool has_original_img = false;
bool block = false;

ros::ServiceClient memory_client;
ros_holographic::NewObject obj;
int counter = 0;

bool compare_size(cv::KeyPoint first, cv::KeyPoint second) {
  return first.size > second.size;
}

void image_raw_callback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& nptr) {
  original_img = msg;
  original_img_info = nptr;
  has_original_img = true;
}

void saliency_map_callback(const sensor_msgs::ImageConstPtr& msg) {
  // convert ros image to cv image
  cv::Mat mat = cv_bridge::toCvShare(msg, "bgr8")->image;
  // convert to gray scale image
  cv::cvtColor(mat, mat, CV_BGR2GRAY);
  // make too dark pixels black
  cv::threshold(mat, mat, 100, 255, 0);
  // resize
  cv::resize(mat, mat, cv::Size(original_img->width, original_img->height));
  // detect keypoints
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(mat, keypoints);
  // make keypoints visible
  cv::Mat keypoint_img;
  std::sort(keypoints.begin(), keypoints.end(), compare_size);
  cv::drawKeypoints(mat, keypoints, keypoint_img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // set and draw roi
  int k_x = keypoints.at(0).pt.x;
  int k_y = keypoints.at(0).pt.y;
  int size = keypoints.at(0).size/2;
  ROS_INFO("keypoint: x: %d, y: %d, size: %d", k_x, k_y, size);
  int x_1 = std::max(0, k_x - size);
  int x_2 = std::min(mat.cols - 1, k_x + size);
  int y_1 = std::max(0, k_y - size);
  int y_2 = std::min(mat.rows - 1, k_y + size);
  ROS_INFO("roi: x1: %d, y1: %d, x2: %d, y2: %d", x_1, y_1, x_2, y_2);
  cv::rectangle(keypoint_img, cv::Point(x_1, y_1), cv::Point(x_2, y_2), cv::Scalar(255, 0, 0));

  // convert to gray scale image
  cv::cvtColor(keypoint_img, keypoint_img, CV_BGR2GRAY);
  // publish image
  sensor_msgs::ImageConstPtr keypoint_msg;
  keypoint_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", keypoint_img).toImageMsg();
  keypoint_pub.publish(keypoint_msg);

  if (keypoints.size() && has_original_img && !block) {
    block = true;
    model.fromCameraInfo(original_img_info);
    cv::Point3d p = model.projectPixelTo3dRay(keypoints[0].pt);
    double dx, dy;
    dx = atan(-p.x);
    dy = atan(-p.y / sqrt(p.x * p.x + 1)); // cartesian to polar coordinates

    // if (abs(dx) > thresh || abs(dy) > thresh) {
    x += dx;
    y += dy;

    ROS_INFO("3dray: %f, %f, %f", p.x, p.y, p.z);
    ROS_INFO("dx: %f, dy: %f", dx, dy);
    ROS_INFO("x: %f, y: %f", x, y);
    ROS_INFO("---");

    // send new joint command
    joint_cmd.header.stamp = ros::Time::now();
    joint_cmd.position[0] = x;
    joint_cmd.position[1] = y;
    joint_cmd_pub.publish(joint_cmd);

    // send new positions
    neck_yaw_pos.data = x;
    neck_pitch_pos.data = y;
    neck_yaw_pos_pub.publish(neck_yaw_pos);
    neck_pitch_pos_pub.publish(neck_pitch_pos);

    // publish roi
    cv::Mat mat_2 = cv_bridge::toCvShare(original_img, "bgr8")->image;
    cv::Rect roi(x_1, y_1, x_2 - x_1, y_2 - y_1);
    cv::Mat region = mat_2(roi);
    sensor_msgs::ImageConstPtr cropped_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", region).toImageMsg();
    roi_pub.publish(cropped_img);

    // send info to memory
    obj.request.X = x;
    obj.request.Y = y;
    std::stringstream sstm;
    sstm << "label no " << counter;
    obj.request.Label = sstm.str();
    memory_client.call(obj);
    counter++;

    // }
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

  // initialize joint command
  joint_cmd.name.resize(2);
  joint_cmd.position.resize(2);
  joint_cmd.velocity.resize(2);
  joint_cmd.name[0] = "ptu_pan";
  joint_cmd.name[1] = "ptu_tilt";
  joint_cmd.velocity[0] = 0.25;
  joint_cmd.velocity[1] = 0.25;
  joint_cmd.position[0] = x;
  joint_cmd.position[1] = y;

  // initialize neck msgs
  neck_yaw_pos.data = x;
  neck_pitch_pos.data = y;

  // initialize ros node
  ros::init(argc, argv, "get_rays");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  // initialize memory client
  memory_client = nh.serviceClient<ros_holographic::NewObject>("new_object");

  // advertise topics
  keypoint_pub = it.advertise("/kps_map", 1);
  roi_pub = it.advertise("/roi", 1);
  joint_cmd_pub = nh.advertise<sensor_msgs::JointState>("/ptu/cmd", 1);
  neck_yaw_pos_pub = nh.advertise<std_msgs::Float64>("/robot/neck_yaw/pos", 1);
  neck_pitch_pos_pub = nh.advertise<std_msgs::Float64>("/robot/neck_pitch/pos", 1);

  // publish first joint commands
  joint_cmd.header.stamp = ros::Time::now();
  joint_cmd_pub.publish(joint_cmd);
  neck_yaw_pos_pub.publish(neck_yaw_pos);
  neck_pitch_pos_pub.publish(neck_pitch_pos);
  ros::Duration(10).sleep();

  // subscribe and do stuff
  image_transport::Subscriber sub = it.subscribe("/saliency_map", 1, saliency_map_callback);
  image_transport::CameraSubscriber csub = it.subscribeCamera("/icub_model/left_eye_camera/image_raw", 1, image_raw_callback);

  ros::spin();

  return 0;
}
