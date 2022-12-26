import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3Stamped, TransformStamped, PoseStamped
from builtin_interfaces.msg import Time 
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from ekf_lzp.ekf import ekf
import numpy as np

#EKFノードの本体、Lidarデータはまだ使ってない
#imuを入力、odomを観測値とし、その２つの情報をekfで統合する

class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")

        #publisher, subscriber定義
        self.x_est = self.create_publisher(PoseStamped, '/current_pose', 10) #推定位置

        self.imu_sub = self.create_subscription(Imu, '/first_robot/imu', self.imu_callback,10) #imuデータ
        self.odom_sub = self.create_subscription(Odometry, '/first_robot/odom', self.odom_callback, 10) #odomデータ
        self.initial_state = self.create_subscription(PoseStamped, '/initial_pose', self.initialpose_callback,10) #初期位置


        #もろもろの変数
        #self.acc_in = np.matrix([0.0, 0.0, 0.0]).T
        #self.acc_out = np.matrix([0.0, 0.0, 0.0]).T
        #self.w_in = np.matrix([0.0, 0.0, 0.0]).T
        #self.w_out = np.matrix([0.0, 0.0, 0.0]).T

        self.acc_in = Vector3Stamped()
        self.w_in = Vector3Stamped()
        self.imu_time = Time()

        #self.odompose = np.matrix([0.0, 0.0, 0.0]).T
        #self.odomtwist = np.matrix([0.0, 0.0, 0.0]).T
        #self.odompose_out = np.matrix([0.0, 0.0, 0.0]).T
        #self.odomtwist_out = np.matrix([0.0, 0.0, 0.0]).T

        self.odompose = Vector3Stamped()
        self.odomtwist = Vector3Stamped()


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
        self.ekf.x[0, 0] = msg.pose.position.x
        self.ekf.x[1, 0] = msg.pose.position.y
        self.ekf.x[2, 0] = msg.pose.position.z
        self.ekf.x[6, 0] = msg.pose.orientation.x
        self.ekf.x[7, 0] = msg.pose.orientation.y
        self.ekf.x[8, 0] = msg.pose.orientation.z
        self.ekf.x[9, 0] = msg.pose.orientation.w


    #imuメッセージを処理
    def imu_callback(self, msg):
        #変換前
        self.imu_time = msg.header.stamp
        self.acc_in.vector.x = msg.linear_acceleration.x
        self.acc_in.vector.y = msg.linear_acceleration.y
        self.acc_in.vector.z = msg.linear_acceleration.z
        self.w_in.vector.x = msg.angular_velocity.x
        self.w_in.vector.y = msg.angular_velocity.y
        self.w_in.vector.z = msg.angular_velocity.z

        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        time_point = Duration(seconds = sec, nanoseconds = nanosec)

        now = self.get_clock().now() - time_point

        transform = self.buffer.lookup_transform( #2つのframeidからtransformを取得するらしい,　transformはTransformStamped型
                      target_frame = msg.header.frame_id,
                      source_frame = self.robot_frame_id,
                      time = now,
                      timeout =time_point
                    )

        acc_out = tf2_geometry_msgs.do_transform_vector3(self.acc_in, transform) #transformstamped型をvector3型に直す処理
        w_out = tf2_geometry_msgs.do_transform_vector3(self.w_in,  transform) 

        #変換後
        transformed_msg = Imu()
        transformed_msg.header.stamp = self.imu_time
        transformed_msg.angular_velocity.x = w_out.vector.x
        transformed_msg.angular_velocity.y = w_out.vector.y           
        transformed_msg.angular_velocity.z = w_out.vector.z
        transformed_msg.linear_acceleration.x = acc_out.vector.x
        transformed_msg.linear_acceleration.y = acc_out.vector.y
        transformed_msg.linear_acceleration.z = acc_out.vector.z

        self.ekf.predict(transformed_msg) #tf変換処理してからpredictへ


    #odomメッセージを処理
    def odom_callback(self, msg):
        #sec = msg.header.stamp.sec
        #nanosec = msg.header.stamp.nanosec
        #time_point = Duration(seconds = sec, nanoseconds = nanosec)

        #now = self.get_clock().now() - time_point

        #trans= self.buffer.lookup_transform( #2つのframeidからtransformを取得するらしい
        #              target_frame = msg.header.frame_id,
        #              source_frame = self.robot_frame_id,
        #              time = now,
        #              timeout =time_point
        #            )

        self.odompose.vector.x = msg.pose.pose.position.x
        self.odompose.vector.y = msg.pose.pose.position.y
        self.odompose.vector.z = msg.pose.pose.position.z
        self.odomtwist.vector.x = msg.twist.twist.linear.x
        self.odomtwist.vector.y = msg.twist.twist.linear.y
        self.odomtwist.vector.z = msg.twist.twist.linear.z

        #odompose_out = tf2_geometry_msgs.do_transform_vector3(self.odompose, trans) #transformstamped型をvector3型に直す処理
        #odomtwist_out = tf2_geometry_msgs.do_transform_vector3(self.odomtwist, trans) 

        #変換後
        #transformed_msg = Odometry()
        #transformed_msg.header.stamp = msg.header.stamp
        #transformed_msg.pose.pose.position.x = odompose_out.vector.x
        #transformed_msg.pose.pose.position.y = odompose_out.vector.y           
        #transformed_msg.pose.pose.position.z = odompose_out.vector.z
        #transformed_msg.twist.twist.linear.x = odomtwist_out.vector.x
        #transformed_msg.twist.twist.linear.y = odomtwist_out.vector.y
        #transformed_msg.twist.twist.linear.z = odomtwist_out.vector.z

        #z = np.matrix([self.odompose_out.vector.x, self.odompose_out.vector.y, self.odompose_out.vector.z])

        z = np.matrix([self.odompose.vector.x, self.odompose.vector.y, self.odompose.vector.z])

        self.ekf.ekf_estimation(z) #観測値に相当

    #1秒ごとに実行
    def timer_callback(self):
        position = PoseStamped() 

        #更新後の状態量を取得
        position.header.stamp = self.imu_time #timestampはimuのものを使用
        position.header.frame_id = 'odom'
        position.pose.position.x = self.ekf.x[0, 0]
        position.pose.position.y = self.ekf.x[1, 0]
        position.pose.position.z = self.ekf.x[2, 0]
        position.pose.orientation.x = self.ekf.x[6, 0]
        position.pose.orientation.y = self.ekf.x[7, 0]
        position.pose.orientation.z = self.ekf.x[8, 0]
        position.pose.orientation.w = self.ekf.x[9, 0]

        self.x_est.publish(position) #更新後の位置をpublish



