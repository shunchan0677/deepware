//
// Created by ad on 3/27/18.
//

#include <ros/ros.h>
#include <historical_pcs_images.h>

namespace historical_pcs_images {

  HistoricalPcsImages::HistoricalPcsImages(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  {
    nh_ = nh;
    pnh_ = pnh;
  }

  HistoricalPcsImages::~HistoricalPcsImages()
  {
    // deconstruct
  }

  void HistoricalPcsImages::callback_pointcloud_raw(const sensor_msgs::PointCloud2 input)
  {
    std::cout << "Call" << std::endl;
    sensor_msgs::PointCloud2 pc2_transformed;

    //listener.lookupTransform("map", "velodyne", ros::Time(0), transform);
    

    try{
        listener.waitForTransform("map", "base_link", input.header.stamp, ros::Duration(4.0));
        listener.lookupTransform("map", "base_link", input.header.stamp, transform);
    }
    catch (tf2::ExtrapolationException ex){
        ROS_ERROR("%s",ex.what());
    }
    //listener.lookupTransform("old_tf_5", "map", ros::Time(0), transform);
    pcl_ros::transformPointCloud("map", transform, input, pc2_transformed);


    vec.push_back(pc2_transformed);
    if(vec.size() >= 11){
        vec.erase(vec.begin());
        seq ++;
        try{
            listener.waitForTransform("base_link", "map", input.header.stamp, ros::Duration(4.0));
            listener.lookupTransform("base_link", "map", input.header.stamp - ros::Duration(7*0.1), transform);
        }
        catch (tf2::ExtrapolationException ex){
            ROS_ERROR("%s",ex.what());
        }
        for (int i = 0; i < 10; i++) {
            points_publisher_[i].publish(vec[i]);


        sensor_msgs::PointCloud2 pc_before;
        sensor_msgs::PointCloud2 pc_transformed;
        pc_before =  vec[i];


        pcl_ros::transformPointCloud("base_link", transform, pc_before, pc_transformed);
        //cv::Mat conv_image;
        //conv_image = Pc2Image(vec[5]);

        pcl::PointCloud<pcl::PointXYZI> current_points;
        pcl::fromROSMsg(pc_transformed, current_points);
        cv::Mat image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);


        std::cout << "Create image" << std::endl;
        std::cout << current_points.points.size()/2 << std::endl;
        unsigned int index;
        for (index=0; index<int(current_points.points.size()/2); index++) {
            if ((current_points.points[index*2].z < min_height_) || (current_points.points[index*2].z > max_height_))
              continue;
            int x = int(horizontal_number_of_cells_/2.0 - current_points.points[index*2].x / cell_height_ + offset_x_);
            int y = int(horizontal_number_of_cells_/2.0 - current_points.points[index*2].y / cell_height_ + offset_y_);
            if(abs(x) < vertical_number_of_cells_ && abs(y) < horizontal_number_of_cells_ && x > 0 && y > 0){
                image.at<uchar>(x, y) = 255;
            }
        }
        std::cout << index << std::endl;

        // publish the image
        cv_bridge::CvImage out_msg;
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = image;
        publisher_[i].publish(out_msg.toImageMsg());
        }
    }

  }


  cv::Mat HistoricalPcsImages::Pc2Image(sensor_msgs::PointCloud2 pointcloud2_raw)
  {
    pcl::PointCloud<pcl::PointXYZI> current_points;
    pcl::fromROSMsg(pointcloud2_raw, current_points);
    cv::Mat image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);
    for (unsigned int i=0; i<current_points.points.size(); i++) {
      //if ((current_points.points[i].z < min_height_) || (current_points.points[i].z > max_height_))
      //  continue;
      int x = int(current_points.points[i].x * vertical_number_of_cells_ / 60.0);
      int y = int(current_points.points[i].y * horizontal_number_of_cells_ / 60.0);
      if(abs(x) < vertical_number_of_cells_ && abs(y) < horizontal_number_of_cells_ && x > 0 && y > 0)
        image.at<uchar>(x, y) = 255;
    }
    return image;
  }



  void HistoricalPcsImages::getParameters()
  {
    std::string colormap;
    seq = 0;
    pnh_.param<std::string>("pointcloud_raw_topic_name", pointcloud_raw_topic_name_, "filtered_points");
    pnh_.param<int>("horizontal_number_of_cells", horizontal_number_of_cells_,400);
    pnh_.param<int>("vertical_number_of_cells", vertical_number_of_cells_, 400);
    pnh_.param<double>("cell_width", cell_width_, 80/400.0);   //0.35
    pnh_.param<double>("cell_height", cell_height_, 80.0/400.0); //0.35
    pnh_.param<int>("scan_resolution", scan_resolution_, 360);
    pnh_.param<double>("min_height", min_height_, -0.9);
    pnh_.param<double>("max_height", max_height_, 1.0);
    pnh_.param<double>("min_abs_x", min_abs_x_, 0.0);
    pnh_.param<double>("min_abs_y", min_abs_y_, 0.0);
    pnh_.param<double>("offset_x", offset_x_, 120);
    pnh_.param<double>("offset_y", offset_y_, 0.0);
    pnh_.param<std::string>("pointcloud_raw_topic_name", pointcloud_raw_topic_name_, "points_raw");

  }


  void HistoricalPcsImages::MainLoop()
  {

    // debug mode
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
      ros::console::notifyLoggerLevelsChanged();
    }

    // get parameters
    getParameters();

    // advertise publisher

    for (int i = 0; i < 10; i++) {
        ros::Publisher p_publisher_;
        std::stringstream ss;
        ss << "historical_points_" << i;
        p_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(ss.str(), 1);
        points_publisher_.push_back(p_publisher_);
        image_transport::ImageTransport it(nh_);
        std::stringstream ss2;
        ss2 << "occupancy_grid_" << i;
        publisher_.push_back(it.advertise(ss2.str(), 1));

    }



    // subscribe points_raw and current_pose
    ros::Subscriber points_sub = nh_.subscribe<sensor_msgs::PointCloud2>
        (pointcloud_raw_topic_name_, 10000, &HistoricalPcsImages::callback_pointcloud_raw, this);

    ros::Rate loop_rate(10);

    while (ros::ok()) {
//      ros::spin();
      ros::spinOnce();
      loop_rate.sleep();
//      getParameters();  // calling getParameters() every step makes memory leak
    }

  }



} // end of namespace: overhead_view_image_publisher
