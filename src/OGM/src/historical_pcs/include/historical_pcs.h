//
// Created by ad on 3/27/18.
//

#ifndef DIAM_INTENSHIP_OVERHEAD_VIEW_IMAGE_PUBLISHER_H
#define DIAM_INTENSHIP_OVERHEAD_VIEW_IMAGE_PUBLISHER_H

#include <ros/ros.h>
#include <string>
#include <pthread.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ros/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <tf/tf.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace historical_pcs {
  class HistoricalPcs {
  public:
    HistoricalPcs(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    virtual ~HistoricalPcs();
    void MainLoop();

  private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    std::vector<sensor_msgs::PointCloud2> vec;

    std::string pointcloud_raw_topic_name_;

    tf::StampedTransform transform;
    tf::TransformListener listener;
    pthread_mutex_t mutex;

    bool pointcloud_raw_loaded_;
    sensor_msgs::PointCloud2::ConstPtr pointcloud2_raw_ptr_;

    // publisher
    //ros::Publisher points_publisher_;

    std::vector<ros::Publisher> points_publisher_;

    // callbacks
    void callback_pointcloud_raw(const sensor_msgs::PointCloud2 input);

    void getParameters();
  };

} // namespace overhead_view_occupancy_grid_publisher

#endif //DIAM_INTENSHIP_OVERHEAD_VIEW_IMAGE_PUBLISHER_H
