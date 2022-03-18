## Occupancy Grid Map Feature

This is a converter of velodyne points clouds to occupancy grid map.

overhead_view_occupancy_grid_publisher was developed by Daiki Hayashi.

historical_pcs and historical_pcs_images was developed by Shunya Seiya.

OGM prediction modules
* Paper : https://arxiv.org/pdf/1812.09395.pdf 
* Video : https://www.youtube.com/watch?v=Bskd0Z7eLFE&feature=youtu.be

/points_raw -> /overhead_view_occupancy_grid


## How to use

"cd ChauffuerNet/OGM"

"catkin_make"

"source devel/setup.bash"

"rosrun overhead_view_occupancy_grid_publisher overhead_view_occupancy_grid_publisher_node"


## Related works

* Deep Tracking
   * Paper : http://www.robots.ox.ac.uk/~mobile/Papers/2017_IJRR_Dequaire.pdf
   * Code : https://github.com/pondruska/DeepTracking
