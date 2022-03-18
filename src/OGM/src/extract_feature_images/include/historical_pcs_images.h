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
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

namespace historical_pcs_images {
  class HistoricalPcsImages {
  public:
    HistoricalPcsImages(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    virtual ~HistoricalPcsImages();
    void MainLoop();

  private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    std::vector<sensor_msgs::PointCloud2> vec;
    std::vector<sensor_msgs::PointCloud2> vec_ego;
    std::vector<sensor_msgs::PointCloud2> vec_obj;

    std::string pointcloud_raw_topic_name_;
    int horizontal_number_of_cells_, vertical_number_of_cells_, scan_resolution_;
    double cell_width_, cell_height_, min_height_, max_height_, min_abs_x_, min_abs_y_, offset_x_, offset_y_, scan_step_inv_;
    double* scanned_line_;

    tf::StampedTransform transform;
    tf::TransformListener listener;
    pthread_mutex_t mutex;
    int seq;

    bool pointcloud_raw_loaded_;
    sensor_msgs::PointCloud2::ConstPtr pointcloud2_raw_ptr_;
    sensor_msgs::PointCloud2 hd_maps;
    sensor_msgs::PointCloud2 way;
    sensor_msgs::PointCloud2 obj;

    // publisher
    std::vector<image_transport::Publisher> publisher_;
    std::vector<image_transport::Publisher> publisher_ego;
    std::vector<image_transport::Publisher> publisher_obj;
    image_transport::Publisher publisher_map;
    image_transport::Publisher publisher_way;
    std::vector<ros::Publisher> points_publisher_;

    // callbacks
    void callback_pointcloud_raw(const sensor_msgs::PointCloud2 input);
    void callback_hd_map(const visualization_msgs::MarkerArray input_a);
    void callback_way(const visualization_msgs::MarkerArray input_a);
    void callback_obj(const visualization_msgs::MarkerArray input_a);

    void getParameters();
    cv::Mat Pc2Image(const sensor_msgs::PointCloud2 pointcloud2_raw);
  };

} // namespace overhead_view_occupancy_grid_publisher

#endif //DIAM_INTENSHIP_OVERHEAD_VIEW_IMAGE_PUBLISHER_H
