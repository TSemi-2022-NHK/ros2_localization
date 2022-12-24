import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3Stamped,TransformStamped, PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ekf_lzp.ekf import ekf
import numpy as np

#EKFノードの本体、Lidarデータはまだ使ってない
#imuを入力、odomを観測値とし、その２つの情報をekfで統合する

class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")

        #publisher, subscriber定義
        self.x_est = self.create_publisher(Pose, '/current_pose', 10) #推定位置

        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback,10) #imuデータ
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #odomデータ
        self.initial_state = self.create_subscription(Pose, '/initial_pose', self.initialpose_callback,10) #初期位置

        #もろもろの変数
        self.acc_in = np.matrix([0.0, 0.0, 0.0])
        self.acc_out = np.matrix([0.0, 0.0, 0.0])
        self.w_in = np.matrix([0.0, 0.0, 0.0])
        self.W_out = np.matrix([0.0, 0.0, 0.0])

        self.odompose = np.matrix([0.0, 0.0, 0.0])
        self.odomtwist = np.matrix([0.0, 0.0, 0.0])
        self.odompose_out = np.matrix([0.0, 0.0, 0.0])
        self.odomtwist_out = np.matrix([0.0, 0.0, 0.0])


        self.robot_frame_id = "base_link" #ロボットのframeidを指定

        #ekf初期化
        self.ekf = ekf.Ekf()

        #timer
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        #tf2用のバッファとリスナ
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)


    #コールバック関数たち
    #初期位置を取得
    def initialpose_callback(self, msg):
        self.ekf.x[0] = msg.position.x
        self.ekf.x[1] = msg.position.y
        self.ekf.x[2] = msg.position.z
        self.ekf.x[6] = msg.orientation.x
        self.ekf.x[7] = msg.orientation.y
        self.ekf.x[8] = msg.orientation.z
        self.ekf.x[9] = msg.orientation.w


    #imuメッセージを処理
    def imu_callback(self, msg):
        self.acc_in[0] = msg.linear_acceleration.x
        self.acc_in[1] = msg.linear_acceleration.y
        self.acc_in[2] = msg.linear_acceleration.z
        self.w_in[0] = msg.angular_velocity.x
        self.w_in[1] = msg.angular_velocity.y
        self.w_in[2] = msg.angular_velocity.z

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nsec
        time_point = Duration(sec = sec, nanoseconds = nanosec)

        now = self.get_clock().now() - time_point

        transform = self.buffer.lookup_transform( #2つのframeidからtransformを取得するらしい
                      target_frame = self.robot_frame_id,
                      source_frame = msg.header.frame_id,
                      now = now,
                      time=time_point
                    )

        self.acc_out = self.buffer.transform(transform, self.acc_in, time_point) #tf変換処理
        self.w_out = self.buffer.transform(transform, self.w_in, time_point) 

        transformed_msg = Imu()
        transformed_msg.header.stamp = msg.header.stamp
        transformed_msg.angular_velocity.x = self.w_out.vector.x
        transformed_msg.angular_velocity.y = self.w_out.vector.y           
        transformed_msg.angular_velocity.z = self.w_out.vector.z
        transformed_msg.linear_acceleration.x = self.acc_out.vector.x
        transformed_msg.linear_acceleration.y = self.acc_out.vector.y
        transformed_msg.linear_acceleration.z = self.acc_out.vector.z

        self.ekf.predict(transformed_msg) #tf変換処理してからpredictへ

        self.get_logger().info("Subscribe : " + msg.data)


    #odomメッセージを処理
    def odom_callback(self, msg):
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nsec
        time_point = Duration(sec = sec, nanoseconds = nanosec)

        now = self.get_clock().now() - time_point

        trans= self.buffer.lookup_transform( #2つのframeidからtransformを取得するらしい buffer２ついる?
                      target_frame = self.robot_frame_id,
                      source_frame = msg.header.frame_id,
                      now = now,
                      time=time_point
                    )

        self.odompose[0] = msg.pose.pose.position.x
        self.odompose[1] = msg.pose.pose.position.y
        self.odompose[2] = msg.pose.pose.position.z
        self.odomtwist[0] = msg.twist.twist.linear.x
        self.odomtwist[1] = msg.twist.twist.linear.y
        self.odomtwist[2] = msg.twist.twist.linear.z

        self.odompose_out = self.buffer.transform(trans, self.odompose, time_point) #tf変換処理
        self.odomtwist_out = self.buffer.transform(trans, self.odomtwist, time_point) 

        transformed_msg = Odometry()
        transformed_msg.header.stamp = msg.header.stamp
        transformed_msg.pose.pose.position.x = self.odompose_out.vector.x
        transformed_msg.pose.pose.position.y = self.odompose_out.vector.y           
        transformed_msg.pose.pose.position.z = self.odompose_out.vector.z
        transformed_msg.twist.twist.linear.x = self.odomtwist_out.vector.x
        transformed_msg.twist.twist.linear.y = self.odomtwist_out.vector.y
        transformed_msg.twist.twist.linear.z = self.odomtwist_out.vector.z

        z = np.matrix([self.odompose_out.vector.x, self.odompose_out.vector.y, self.odompose_out.vector.z])

        self.ekf.ekf_estimation(z) #観測値に相当

        self.get_logger().info("Subscribe : " + msg.data)


    #1秒ごとに実行
    def timer_callback(self):
        position = Pose() 

        #更新後の位置を取得
        position.position.x = self.ekf.x[0]
        position.position.y = self.ekf.x[1]
        position.position.z = self.ekf.x[2]

        self.x_est.publish(position) #更新後の位置をpublish




