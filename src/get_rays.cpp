#include "ros/ros.h"
#include "std_msgs/String.h"
#include "image_transport/image_transport.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_geometry/pinhole_camera_model.h"
#include "sensor_msgs/JointState.h"
#include "sensor_msgs/CameraInfo.h"

#include <math.h>
#include <algorithm>
#include <sstream>

image_transport::Publisher  pubkps;
ros::Publisher  pubcmd;
cv::SimpleBlobDetector detector;
image_geometry::PinholeCameraModel model;
sensor_msgs::CameraInfoConstPtr ptr;
sensor_msgs::JointState j;
double x = 0, y = 0, thresh = 0.05;
bool set = false;
bool block = false;

bool compare_size(cv::KeyPoint first, cv::KeyPoint second)
{
  return first.size > second.size;
}

void imageCallback2(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& nptr){
    ptr = nptr;
    set = true;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
  cv::Mat mat = cv_bridge::toCvShare(msg, "bgr8")->image;
  cv::cvtColor(mat,mat, CV_BGR2GRAY );
  cv::threshold(mat,mat,100,255,0);
  cv::resize(mat,mat,cv::Size(ptr->width,ptr->height));
  std::vector<cv::KeyPoint> keypoints;
  detector.detect(mat, keypoints);
  cv::Mat kps;
  cv::drawKeypoints( mat, keypoints, kps, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  std::sort(keypoints.begin(),keypoints.end(),compare_size);
  cv::cvtColor(kps,kps, CV_BGR2GRAY );
  sensor_msgs::ImageConstPtr ikps;
  ikps = cv_bridge::CvImage(std_msgs::Header(), "mono8", kps).toImageMsg();
  pubkps.publish(ikps);    
  if (keypoints.size() && set && !block) {
    block = true;
    ROS_ERROR_STREAM("" << model.fromCameraInfo(ptr));
    cv::Point3d p = model.projectPixelTo3dRay(keypoints[0].pt);
    double jx, jy;
    jx = atan(- p.x);
    jy = atan(- p.y / sqrt(p.x * p.x + 1));
    ROS_ERROR("%f %f",keypoints[0].pt.x, keypoints[0].pt.y);
    ROS_ERROR("%f %f",jx,jy);
    if (true/*abs(jx) > thresh || abs(jy) > thresh*/){
      j.header.stamp = ros::Time::now();
      x += jx;
      y += jy;
      j.position[0] = x;
      j.position[1] = y;
      pubcmd.publish(j);
    }
    ros::Duration(5).sleep();
    block = false;
  }
}

int main(int argc, char **argv)
{
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
  detector = cv::SimpleBlobDetector(params);
  j.name.resize(2);
  j.position.resize(2);
  j.velocity.resize(2);
  j.name[0] = "ptu_pan";
  j.name[1] = "ptu_tilt";
  j.velocity[0] = 0.25;
  j.velocity[1] = 0.25;
  j.position[0] = x;
  j.position[1] = y;
  ros::init(argc, argv, "get_ray");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  pubkps = it.advertise("/kps_map",1);
  pubcmd = nh.advertise<sensor_msgs::JointState>("/ptu/cmd", 1); 
  j.header.stamp = ros::Time::now();
  pubcmd.publish(j);  
  ros::Duration(10).sleep();
  image_transport::Subscriber sub = it.subscribe("/saliency_map", 1, imageCallback);
  image_transport::CameraSubscriber csub = it.subscribeCamera("/rgb/image_raw", 1, imageCallback2);
  ros::spin();
  
  return 0;
}
