//
// Created by d-hayashi on 3/27/18.
//

#include <ros/ros.h>
#include <overhead_view_occupancy_grid_publisher.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "overhead_view_occupancy_grid_publisher");
  ros::NodeHandle nh(""), pnh("~");
  overhead_view_occupancy_grid_publisher::OverheadViewOccupancyGridPublisher publisher(nh, pnh);
  publisher.MainLoop();

  return (0);
}
