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

  void HistoricalPcsImages::callback_hd_map(const visualization_msgs::MarkerArray input_a)
  {
    std::cout << "Call_hd_map" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width    = input_a.markers.size()*2;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
        for (int i = 0; i < input_a.markers.size(); i++) {
            cloud.points[i*2].x = input_a.markers[i].points[0].x;
            cloud.points[i*2].y = input_a.markers[i].points[0].y;
            cloud.points[i*2].z = input_a.markers[i].points[0].z;
            cloud.points[i*2+1].x = input_a.markers[i].points[1].x;
            cloud.points[i*2+1].y = input_a.markers[i].points[1].y;
            cloud.points[i*2+1].z = input_a.markers[i].points[1].z;
        }

        pcl::toROSMsg(cloud, hd_maps);

  }

  void HistoricalPcsImages::callback_way(const visualization_msgs::MarkerArray input_a)
  {
    std::cout << "Call_way" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width    = input_a.markers.size();
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
        for (int i = 0; i < input_a.markers.size(); i++) {
            cloud.points[i].x = input_a.markers[i].pose.position.x;
            cloud.points[i].y = input_a.markers[i].pose.position.y;
            cloud.points[i].z = input_a.markers[i].pose.position.z;
        }

        pcl::toROSMsg(cloud, way);

  }

  void HistoricalPcsImages::callback_obj(const visualization_msgs::MarkerArray input_a)
  {
    std::cout << "Call_obj" << std::endl;
    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width    = input_a.markers.size()*4;
    cloud.height   = 1;
    cloud.is_dense = false;
    cloud.points.resize (cloud.width * cloud.height);

    pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
        for (int i = 0; i < input_a.markers.size(); i++) {
            float yaw = tf::getYaw(input_a.markers[i].pose.orientation);
            float size_x = input_a.markers[i].scale.x;
            float size_y = input_a.markers[i].scale.y;
            float x = size_x*cos(yaw) - size_y*sin(yaw);
            float y = size_x*sin(yaw) + size_y*cos(yaw);
            cloud.points[4*i].x = input_a.markers[i].pose.position.x + x/2.0;
            cloud.points[4*i].y = input_a.markers[i].pose.position.y + y/2.0;
            cloud.points[4*i].z = input_a.markers[i].pose.position.z;

            size_x = -input_a.markers[i].scale.x;
            size_y = input_a.markers[i].scale.y;
            x = size_x*cos(yaw) - size_y*sin(yaw);
            y = size_x*sin(yaw) + size_y*cos(yaw);
            cloud.points[4*i+1].x = input_a.markers[i].pose.position.x + x/2.0;
            cloud.points[4*i+1].y = input_a.markers[i].pose.position.y + y/2.0;
            cloud.points[4*i+1].z = input_a.markers[i].pose.position.z;

            size_x = -input_a.markers[i].scale.x;
            size_y = -input_a.markers[i].scale.y;
            x = size_x*cos(yaw) - size_y*sin(yaw);
            y = size_x*sin(yaw) + size_y*cos(yaw);
            cloud.points[4*i+2].x = input_a.markers[i].pose.position.x + x/2.0;
            cloud.points[4*i+2].y = input_a.markers[i].pose.position.y + y/2.0;
            cloud.points[4*i+2].z = input_a.markers[i].pose.position.z;

            size_x = input_a.markers[i].scale.x;
            size_y = -input_a.markers[i].scale.y;
            x = size_x*cos(yaw) - size_y*sin(yaw);
            y = size_x*sin(yaw) + size_y*cos(yaw);
            cloud.points[4*i+3].x = input_a.markers[i].pose.position.x + x/2.0;
            cloud.points[4*i+3].y = input_a.markers[i].pose.position.y + y/2.0;
            cloud.points[4*i+3].z = input_a.markers[i].pose.position.z;

        }

        pcl::toROSMsg(cloud, obj);


  }

  void HistoricalPcsImages::callback_pointcloud_raw(const sensor_msgs::PointCloud2 input)
  {
    //std::cout << "Call" << std::endl;
    sensor_msgs::PointCloud2 pc2_transformed;


    pcl::PointCloud<pcl::PointXYZ> ego_cloud;
    sensor_msgs::PointCloud2 ego_before;
    sensor_msgs::PointCloud2 ego_trans;

    ego_cloud.width    = 4;
    ego_cloud.height   = 1;
    ego_cloud.is_dense = false;
    ego_cloud.points.resize (ego_cloud.width * ego_cloud.height);

    ego_cloud.points[0].x = 4.36/2;
    ego_cloud.points[0].y = 1.695/2;
    ego_cloud.points[0].z = 0;

    ego_cloud.points[1].x = 4.36/2;
    ego_cloud.points[1].y = -1.695/2;
    ego_cloud.points[1].z = 0;

    ego_cloud.points[2].x = -4.36/2;
    ego_cloud.points[2].y = -1.695/2;
    ego_cloud.points[2].z = 0;

    ego_cloud.points[3].x = -4.36/2;
    ego_cloud.points[3].y = 1.695/2;
    ego_cloud.points[3].z = 0;


    pcl::toROSMsg(ego_cloud, ego_before);

    

    try{
        listener.waitForTransform("map", "base_link", input.header.stamp, ros::Duration(1.0));
        listener.lookupTransform("map", "base_link", input.header.stamp, transform);
    }
    catch (tf2::ExtrapolationException ex){
        ROS_ERROR("now : %s",ex.what());
    }
    //listener.lookupTransform("old_tf_5", "map", ros::Time(0), transform);
    pcl_ros::transformPointCloud("map", transform, input, pc2_transformed);
    pcl_ros::transformPointCloud("map", transform, ego_before, ego_trans);

    vec.push_back(pc2_transformed);
    vec_ego.push_back(ego_trans);
    vec_obj.push_back(obj);
    if(vec.size() >= 21){
        vec.erase(vec.begin());
        vec_ego.erase(vec_ego.begin());
        vec_obj.erase(vec_obj.begin());
        seq ++;
        try{
            listener.waitForTransform("base_link", "map", input.header.stamp, ros::Duration(1.0));
            listener.lookupTransform("base_link", "map", input.header.stamp - ros::Duration(0.6), transform);//TODO
        }
        catch (tf2::ExtrapolationException ex){
            ROS_ERROR("past : %s",ex.what());
        }
        for (int i = 0; i < 10; i++) {
        //   points_publisher_[i].publish(vec[i]);


        sensor_msgs::PointCloud2 pc_before;
        sensor_msgs::PointCloud2 pc_transformed;
        pc_before =  vec[2*i+1];

        pcl_ros::transformPointCloud("base_link", transform, pc_before, pc_transformed);

        sensor_msgs::PointCloud2 ego_before;
        sensor_msgs::PointCloud2 ego_transformed;
        ego_before =  vec_ego[2*i+1];

        pcl_ros::transformPointCloud("base_link", transform, ego_before, ego_transformed);

        sensor_msgs::PointCloud2 obj_before;
        sensor_msgs::PointCloud2 obj_transformed;
        obj_before =  vec_obj[2*i+1];

        pcl_ros::transformPointCloud("base_link", transform, obj_before, obj_transformed);




        //cv::Mat conv_image;
        //conv_image = Pc2Image(vec[5]);

        pcl::PointCloud<pcl::PointXYZI> current_points;
        pcl::fromROSMsg(pc_transformed, current_points);
        cv::Mat image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);


        pcl::PointCloud<pcl::PointXYZ> current_ego_points;
        pcl::fromROSMsg(ego_transformed, current_ego_points);
        cv::Mat ego_image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);


        pcl::PointCloud<pcl::PointXYZ> current_obj_points;
        pcl::fromROSMsg(obj_transformed, current_obj_points);
        cv::Mat obj_image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);


        //std::cout << "Create image" << std::endl;
        //std::cout << current_points.points.size()/2 << std::endl;
        unsigned int index;
        for (index=0; index<int(current_points.points.size()/2); index++) {
            if ((current_points.points[index*2].z < min_height_) || (current_points.points[index*2].z > max_height_))
              continue;
            int x = int(horizontal_number_of_cells_/2.0 - current_points.points[index*2].x / cell_height_ + offset_x_);
            int y = int(horizontal_number_of_cells_/2.0 - current_points.points[index*2].y / cell_height_ + offset_y_);
            if(abs(x+1) < vertical_number_of_cells_ && abs(y+1) < horizontal_number_of_cells_ && x-1 > 0 && y-1 > 0){
                image.at<uchar>(x, y) = 255;
                image.at<uchar>(x+1, y) = 255;
                image.at<uchar>(x, y+1) = 255;
                image.at<uchar>(x-1, y) = 255;
                image.at<uchar>(x, y-1) = 255;
                //cv::circle(image,cv::Point(y,x),1,cv::Scalar(255,255,255),2);
            }
        }
        //std::cout << index << std::endl;

       cv::Point pt[4];
       for (index=0; index<4; index++) {
       int x = horizontal_number_of_cells_/2.0 - current_ego_points.points[index].x / cell_height_ + offset_x_;
       int y = horizontal_number_of_cells_/2.0 - current_ego_points.points[index].y / cell_height_ + offset_y_;
       pt[index] = cv::Point(y, x);
       }
       cv::fillConvexPoly(ego_image, pt,4,cv::Scalar(255,255,255));


       for (index=0; index<int(current_obj_points.points.size()/4); index++) {
       cv::Point pt[4];
       for (int j =0; j<4; j++) {
       int x = horizontal_number_of_cells_/2.0 - current_obj_points.points[index*4+j].x / cell_height_ + offset_x_;
       int y = horizontal_number_of_cells_/2.0 - current_obj_points.points[index*4+j].y / cell_height_ + offset_y_;
       pt[j] = cv::Point(y, x);
       }
       
       cv::fillConvexPoly(obj_image, pt,4,cv::Scalar(255,255,255));
       }
        


        // publish the image
        cv_bridge::CvImage out_msg;
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = image;
        publisher_[i].publish(out_msg.toImageMsg());


        // publish the ego image
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = ego_image;
        publisher_ego[i].publish(out_msg.toImageMsg());



        // publish the obj image
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = obj_image;
        publisher_obj[i].publish(out_msg.toImageMsg());

        }
        
        //std::cout << "Create map image" << std::endl;
        pcl::PointCloud<pcl::PointXYZ> tmp_cloud1;

        sensor_msgs::PointCloud2 hd_maps_base;
        pcl_ros::transformPointCloud("base_link", transform, hd_maps, hd_maps_base);
        cv::Mat map_image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);
        pcl::PointCloud<pcl::PointXYZ> tmp_cloud;
        pcl::fromROSMsg(hd_maps_base, tmp_cloud);
        for (int i = 0; i < tmp_cloud.points.size()/2; i++) {
            float start_x = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i*2].x / cell_height_ + offset_x_;
            float start_y = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i*2].y / cell_height_ + offset_y_;
            float end_x = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i*2+1].x / cell_height_ + offset_x_;
            float end_y = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i*2+1].y / cell_height_ + offset_y_;
            cv::line(map_image,cv::Point(start_y, start_x),cv::Point(end_y, end_x),cv::Scalar(255,255,255),2,4);
        }

        // publish the image
        cv_bridge::CvImage out_msg;
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = map_image;
        publisher_map.publish(out_msg.toImageMsg());
        
        //std::cout << "Create way image" << std::endl;
        sensor_msgs::PointCloud2 way_base;
        pcl_ros::transformPointCloud("base_link", transform, way, way_base);
        cv::Mat way_image = cv::Mat::zeros(vertical_number_of_cells_, horizontal_number_of_cells_, CV_8UC1);
        pcl::fromROSMsg(way_base, tmp_cloud);
        for (int i = 0; i < tmp_cloud.points.size()/4+20; i++) {
            float start_x = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i].x / cell_height_ + offset_x_;
            float start_y = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i].y / cell_height_ + offset_y_;
            float end_x = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i+1].x / cell_height_ + offset_x_;
            float end_y = horizontal_number_of_cells_/2.0 - tmp_cloud.points[i+1].y / cell_height_ + offset_y_;
            cv::line(way_image,cv::Point(start_y, start_x),cv::Point(end_y, end_x),cv::Scalar(255,255,255),8,4);
        }

        // publish the image
        out_msg.header   = input.header;
        out_msg.header.seq   = seq;
        out_msg.encoding = sensor_msgs::image_encodings::MONO8;
        out_msg.image    = way_image;
        publisher_way.publish(out_msg.toImageMsg());

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
      int x = int(current_points.points[i].x * vertical_number_of_cells_ / 80.0);
      int y = int(current_points.points[i].y * horizontal_number_of_cells_ / 80.0);
      if(abs(x) < vertical_number_of_cells_ && abs(y) < horizontal_number_of_cells_ && x > 0 && y > 0)
        //image.at<uchar>(x, y) = 255;
        cv::circle(image,cv::Point(x,y),2,cv::Scalar(255,255,255));
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
        image_transport::ImageTransport it(nh_);
        std::stringstream ss2;
        ss2 << "occupancy_grid_" << i;
        publisher_.push_back(it.advertise(ss2.str(), 1));

        std::stringstream ss3;
        ss3 << "ego_bbox_" << i;
        publisher_ego.push_back(it.advertise(ss3.str(), 1));

        std::stringstream ss4;
        ss4 << "obj_bbox_" << i;
        publisher_obj.push_back(it.advertise(ss4.str(), 1));

    }

    //ros::NodeHandle nh_;
    image_transport::ImageTransport it(nh_);
    //ros::Publisher publisher_map;
    publisher_map = it.advertise("map_image", 1);
    publisher_way = it.advertise("way_image", 1);




    // subscribe points_raw and current_pose
    ros::Subscriber points_sub = nh_.subscribe<sensor_msgs::PointCloud2>
        (pointcloud_raw_topic_name_, 10000, &HistoricalPcsImages::callback_pointcloud_raw, this);

    ros::Subscriber hd_sub = nh_.subscribe<visualization_msgs::MarkerArray>
        ("/vector_map_carla", 10000, &HistoricalPcsImages::callback_hd_map, this);


    ros::Subscriber way_sub = nh_.subscribe<visualization_msgs::MarkerArray>
        ("/global_waypoints_mark", 10000, &HistoricalPcsImages::callback_way, this);

    ros::Subscriber obj_sub = nh_.subscribe<visualization_msgs::MarkerArray>
        ("/object_markers", 10000, &HistoricalPcsImages::callback_obj, this);

    ros::Rate loop_rate(10);

    while (ros::ok()) {
//      ros::spin();
      ros::spinOnce();
      loop_rate.sleep();
//      getParameters();  // calling getParameters() every step makes memory leak
    }

  }



} // end of namespace: overhead_view_image_publisher
