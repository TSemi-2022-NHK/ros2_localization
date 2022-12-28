# ros2_localization

- /src/ekf_lzpが自己位置推定パッケージで、/src/sim_envはgazeoシミュレーション環境立ち上げ用

- 現状、うまく自己位置を拾えているぽいが回転の向きに対応できてない気がする

- input  
/initial_pose (geometry_msgs/PoseStamed)  
/imu (sensor_msgs/Imu)  
/odom (nav_msgs/Odometry)  
/tf(/base_link(robot frame) → /imu_link(imu frame))  
- output  
/curent_pose (geometry_msgs/PoseStamped)  

