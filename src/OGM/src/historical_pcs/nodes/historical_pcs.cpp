//
// Created by ad on 3/27/18.
//

#include <ros/ros.h>
#include <historical_pcs.h>

namespace historical_pcs {

  HistoricalPcs::HistoricalPcs(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  {
    nh_ = nh;
    pnh_ = pnh;
  }

  HistoricalPcs::~HistoricalPcs()
  {
    // deconstruct
  }

  void HistoricalPcs::callback_pointcloud_raw(const sensor_msgs::PointCloud2 input)
  {
    sensor_msgs::PointCloud2 pc2_transformed;
    listener.lookupTransform("map", "base_link", ros::Time(0), transform);
    pcl_ros::transformPointCloud("map", transform, input, pc2_transformed);


    vec.push_back(pc2_transformed);
    if(vec.size() >= 11){
        vec.erase(vec.begin());
        for (int i = 0; i < 10; i++) {
            std::stringstream ss1;
            ss1 << "old_tf_" << i;
            std::cout << vec.size();
            sensor_msgs::PointCloud2 pc_transformed;
            pcl_ros::transformPointCloud(ss1.str(), transform, vec[i], pc_transformed);
            points_publisher_[i].publish(vec[i]);

        }
    }
  }


  void HistoricalPcs::getParameters()
  {
    std::string colormap;
    pnh_.param<std::string>("pointcloud_raw_topic_name", pointcloud_raw_topic_name_, "points_raw");

  }


  void HistoricalPcs::MainLoop()
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
        points_publisher_.push_back(p_publisher_);}

    // subscribe points_raw and current_pose
    ros::Subscriber points_sub = nh_.subscribe<sensor_msgs::PointCloud2>
        (pointcloud_raw_topic_name_, 10000, &HistoricalPcs::callback_pointcloud_raw, this);

    ros::Rate loop_rate(10);

    while (ros::ok()) {
//      ros::spin();
      ros::spinOnce();
      loop_rate.sleep();
//      getParameters();  // calling getParameters() every step makes memory leak
    }

  }



} // end of namespace: overhead_view_image_publisher
