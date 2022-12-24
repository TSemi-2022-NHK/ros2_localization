import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3Stamped,TransformStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ekf_lzp.ekf import ekf
import numpy as np
from datetime import datetime, timedelta

#EKFノードの本体、Lidarデータはまだ使ってない
#imuを入力、odomを観測値とし、その２つの情報をekfで統合する

class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_main")

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

        self.odom_x = np.zeros(10)

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
        nanosec = msg.header.stamp.nanosec
        time_point = datetime.fromtimestamp(sec) + timedelta(microseconds=nanosec/1000)

        transform = self.buffer.lookup_transform( #2つのframeidからtransformを取得するらしい
                      target_frame = self.robot_frame_id,
                      source_frame = msg.header.frame_id,
                      time=time_point
                    )

        acc_out = self.buffer.transform(transform, self.acc_in, time_point) #tf変換処理
        w_out = self.buffer.transform(transform, self.w_in, time_point) 

        transformed_msg = Imu()
        transformed_msg.header.stamp = msg.header.stamp

        transformed_msg.angular_velocity.x = w_out.vector.x
        transformed_msg.angular_velocity.y = w_out.vector.y           
        transformed_msg.angular_velocity.z = w_out.vector.z

        transformed_msg.linear_acceleration.x = acc_out.vector.x
        transformed_msg.linear_acceleration.y = acc_out.vector.y
        transformed_msg.linear_acceleration.z = acc_out.vector.z

        self.ekf.predict(transformed_msg) #tf変換処理してからpredictへ


    #odomメッセージを処理
    def odom_callback(self, msg):
        self.odom_x[0] = msg.pose.pose.position.x
        self.odom_x[1] = msg.pose.pose.position.y
        self.odom_x[2] = msg.pose.pose.position.z

        self.odom_x[3] = msg.twist.twist.linear.x
        self.odom_x[4] = msg.twist.twist.linear.y
        self.odom_x[5] = msg.twist.twist.linear.z

        z = np.matrix([self.odom_x[0], self.odom_x[1], self.odom_x[2]])

        self.ekf.ekf_estimation(z) #観測値に相当


    #1秒ごとに実行
    def timer_callback(self):
        position = Pose() 

        #更新後の位置を取得
        position.position.x = self.ekf.x[0]
        position.position.y = self.ekf.x[1]
        position.position.z = self.ekf.x[2]

        self.x_est.publish(position) #更新後の位置をpublish




