# ros2_localization

- /src/ekf_lzpが自己位置推定パッケージで、/src/sim_envはgazeoシミュレーション環境立ち上げ用

- rviz上に自己位置表示させるのをテスト中

- input  
/initial_pose (geometry_msgs/PoseStamed)  
/imu (sensor_msgs/Imu)  
/odom (nav_msgs/Odometry)  
/tf(/base_link(robot frame) → /imu_link(imu frame))  
- output  
/curent_pose (geometry_msgs/PoseStamped)  

