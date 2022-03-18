//
// Created by d-hayashi on 3/27/18.
//

#include <ros/ros.h>
#include <historical_pcs_images.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "historical_pcs_images");
  ros::NodeHandle nh(""), pnh("~");
  historical_pcs_images::HistoricalPcsImages publisher(nh, pnh);
  publisher.MainLoop();

  return (0);
}
