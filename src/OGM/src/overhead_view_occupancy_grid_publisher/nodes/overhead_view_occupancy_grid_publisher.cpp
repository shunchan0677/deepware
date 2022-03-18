//
// Created by ad on 3/27/18.
//

#include <ros/ros.h>
#include <overhead_view_occupancy_grid_publisher.h>

namespace overhead_view_occupancy_grid_publisher {

  OverheadViewOccupancyGridPublisher::OverheadViewOccupancyGridPublisher(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  {
    nh_ = nh;
    pnh_ = pnh;
  }

  OverheadViewOccupancyGridPublisher::~OverheadViewOccupancyGridPublisher()
  {
    // deconstruct
  }

  void OverheadViewOccupancyGridPublisher::callback_pointcloud_raw(const sensor_msgs::PointCloud2::ConstPtr& input)
  {
    // Convert the data type(from sensor_msgs to pcl).
//    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::PointCloud<pcl::PointXYZI> input_filtered;
//    pcl::fromROSMsg(*input, *input_cloud);
//
//    // downsample
//    double leaf_size = std::min(cell_width_, cell_height_) / 3.0;
//    pcl::VoxelGrid<pcl::PointXYZI> sor;
//    sor.setInputCloud (input_cloud);
//    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
//    sor.filter (input_filtered);
//    sensor_msgs::PointCloud2::Ptr input_filtered_ptr(new sensor_msgs::PointCloud2);
//    pcl::toROSMsg(input_filtered, *input_filtered_ptr);
//
//    pointcloud2_raw_ptr_ = input_filtered_ptr;
    pointcloud2_raw_ptr_ = input;
    scan();
    createImage();
  }

  void OverheadViewOccupancyGridPublisher::scan()
  {
//    ROS_DEBUG_STREAM("scanning points");
    std::fill_n(scanned_line_, scan_resolution_, std::numeric_limits<double>::max());

    pcl::PointCloud<pcl::PointXYZI> current_points;
    pcl::fromROSMsg(*pointcloud2_raw_ptr_, current_points);
    int index = 0;
    double squared_distance, arctan;
    for (unsigned int i=0; i<current_points.points.size(); i++) {
      if ((current_points.points[i].z < min_height_) || (current_points.points[i].z > max_height_))
        continue;
      if (((std::abs(current_points.points[i].x) < min_abs_x_)) && (std::abs(current_points.points[i].y) < min_abs_y_))
        continue;
//      ROS_DEBUG_STREAM("x: " << current_points.points[i].x << ", y: " << current_points.points[i].y);
      if (current_points.points[i].x == 0) {
        if (current_points.points[i].y >= 0)
          arctan = M_PI / 2.0;
        else
          arctan = (-1.0) * M_PI / 2.0;
      }else {
        arctan = std::atan(current_points.points[i].y / current_points.points[i].x);
      }
      if (current_points.points[i].x < 0)
        arctan += M_PI;
      arctan += M_PI / 2;
      index = (int) std::round(arctan * scan_step_inv_);
//      ROS_DEBUG_STREAM("index_pre: " << index);
//      ROS_DEBUG_STREAM("index: " << index);
      squared_distance = std::pow(current_points.points[i].x, 2) + std::pow(current_points.points[i].y, 2);
      if (squared_distance < scanned_line_[index])
        scanned_line_[std::abs(index)] = squared_distance;
    }
    current_points.clear();

  }

  void OverheadViewOccupancyGridPublisher::createImage()
  {

//    ROS_DEBUG_STREAM("creating an overhead view image");
    pcl::PointCloud<pcl::PointXYZI> points_merged;

    // prepare image
    double x, y;
    int index;
    double squared_distance, arctan;
    int horizontal_number_of_cells_half = horizontal_number_of_cells_ / 2;
    int vertical_number_of_cells_half = vertical_number_of_cells_ / 2;
    cv::Mat image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);

    // add to image
    for (int i=0; i<vertical_number_of_cells_; i++) {
      for (int j=0; j<horizontal_number_of_cells_; j++) {
        x = (vertical_number_of_cells_half - i) * cell_height_ + offset_x_;
        y = (horizontal_number_of_cells_half - j) * cell_width_ + offset_y_;
//        ROS_DEBUG_STREAM("x: " << x << ", y: " << y);
        if (x == 0) {
          if (y >= 0)
            arctan = M_PI / 2.0;
          else
            arctan = (-1.0) * M_PI / 2.0;
        }else {
          arctan = std::atan(y / x);
        }
        if (x < 0)
          arctan += M_PI;
        arctan += M_PI / 2;
        index = (int) std::round(arctan * scan_step_inv_);
        squared_distance = std::pow(x, 2) + std::pow(y, 2);
        if (std::abs(squared_distance - scanned_line_[std::abs(index)]) < 10.0){
          image.at<uchar>(i, j) = 255;}
        if (squared_distance < scanned_line_[std::abs(index)]){
          image.at<uchar>(i, j) = 0;}
      }
    }

    // publish the image
    cv_bridge::CvImage out_msg;
    out_msg.header   = pointcloud2_raw_ptr_->header;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image    = image;
    publisher_.publish(out_msg.toImageMsg());

    // publish points for debugging
    sensor_msgs::PointCloud2 points_merged_msg;
    pcl::toROSMsg(points_merged, points_merged_msg);
    points_merged_msg.header.frame_id = "/velodyne";
    points_merged_msg.header.stamp = pointcloud2_raw_ptr_->header.stamp;
    points_publisher_.publish(points_merged_msg);

    points_merged.clear();

  }

  void OverheadViewOccupancyGridPublisher::getParameters()
  {
    std::string colormap;
    pnh_.param<std::string>("pointcloud_raw_topic_name", pointcloud_raw_topic_name_, "points_no_ground");
    pnh_.param<int>("horizontal_number_of_cells", horizontal_number_of_cells_,256);
    pnh_.param<int>("vertical_number_of_cells", vertical_number_of_cells_, 256);
    pnh_.param<double>("cell_width", cell_width_, 0.35);   //0.35
    pnh_.param<double>("cell_height", cell_height_, 0.35); //0.35
    pnh_.param<int>("scan_resolution", scan_resolution_, 360);
    pnh_.param<double>("min_height", min_height_, -1.2);
    pnh_.param<double>("max_height", max_height_, 3.0);
    pnh_.param<double>("min_abs_x", min_abs_x_, 0.0);
    pnh_.param<double>("min_abs_y", min_abs_y_, 0.0);
    pnh_.param<double>("offset_x", offset_x_, 120.0*0.15);
    pnh_.param<double>("offset_y", offset_y_, 0.0);

    scan_step_inv_ = scan_resolution_ / (2 * M_PI);

    if ((sizeof(scanned_line_)/sizeof(*scanned_line_)) != scan_resolution_)
        scanned_line_ = new double[scan_resolution_];
  }


  void OverheadViewOccupancyGridPublisher::MainLoop()
  {

    // debug mode
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
      ros::console::notifyLoggerLevelsChanged();
    }

    // get parameters
    getParameters();

    // init
    scanned_line_ = new double[scan_resolution_];

    // advertise publisher
    points_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>("overhead_view_occupancy_grid_points", 1);
    image_transport::ImageTransport it(nh_);
    publisher_ = it.advertise("overhead_view_occupancy_grid", 1);

    // subscribe points_raw and current_pose
    ros::Subscriber points_sub = nh_.subscribe<sensor_msgs::PointCloud2>
        (pointcloud_raw_topic_name_, 10000, &OverheadViewOccupancyGridPublisher::callback_pointcloud_raw, this);

    ros::Rate loop_rate(100);

    while (ros::ok()) {
//      ros::spin();
      ros::spinOnce();
      loop_rate.sleep();
//      getParameters();  // calling getParameters() every step makes memory leak
    }

  }



} // end of namespace: overhead_view_image_publisher
