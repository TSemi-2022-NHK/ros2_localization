import numpy as np
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation

g = np.matrix([0, 0, 9.80665]).T

#カルマンフィルタ処理を行う
class Ekf:
    def __init__(self, initial_x = np.zeros(10), initial_p = np.eye(9) * 100):
        self.x = initial_x #初期状態 np.matrix(x, y, z, vx, vy, vz, qx,qy, qz, qw)　= (p v q)
        self.p = initial_p #共分散

        self.Q = np.eye(3) #適当な初期化
        self.dt = 0.1 #imuからの時間、適当な初期化
        self.current_time_imu = 0
        self.previous_time_imu = 0

    def skew(w): #3次元ベクトルを歪対称行列に変換するだけの関数
        V = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]
        ])

        return V

    def motion_model(self, u): #動作モデル
        p = np.matrix(self.x[0], self.x[1], self.x[2]).T
        v = np.matrix(self.x[3], self.x[4], self.x[5]).T
        q = np.matrix(self.x[6], self.x[7], self.x[8], self.x[9]).T

        #timestep
        self.current_time_imu = u.header.stamp.sec + u.header.stamp.nanosec * 1e-9
        self.dt = self.current_time_imu - self.previous_time_imu
        self.previous_time_imu = self.current_time_imu

        a_imu = np.matrix(u.linear_acceleration.x, u.linear_acceleration.y, u.linear_acceleration.z).T
        w_imu = np.matrix(u.angular_velocity.x, u.angular_velocity.y, u.angular_velocity.z).T
        dt_vec = np.matrix([self.dt, self.dt, self.dt]).T

        #predict処理
        v = v + (Rotation.from_quat(q) * a_imu - g)* self.dt
        p = p + v * self.dt + 0.5 * (Rotation.from_quat(q) * a_imu - g) * self.dt * self.dt
        q = Rotation.from_rotvec(w_imu * dt_vec) * q
 
        #Qの計算もここでしておく
        i3 = np.eye(3)
        self.Q = np.matrix([[0.33 * i3, 0], #分散は0.33に設定
                            [0, 0.33 * i3],
        ])
        self.Q = self.Q * self.dt * self.dt

        return np.matrix([p, v, q])   
    
    #予測値を計算
    def predict(self, u): #uはimuからの情報
        i3 = np.eye(3)

        self.x = self.motion_model(u) #ここでdtも計算

        jf = self.calc_jf(self.x, u)

        L = np.matrix([[0, 0],
                       [i3, 0],
                       [0, i3]
        ])

        self.p = jf * self.p * jf.T + L * self.Q * L.T

        return self.x, self.p
    
    #ヤコビアンjf計算
    def calc_jf(self, x, u):
        w_imu = np.matrix(u.angular_velocity.x, u.angular_velocity.y, u.angular_velocity.z).T
        q = np.matrix(x[6], x[7], x[8], x[9]).T
        i3 = np.eye(3)

        jf = np.matrix([[i3, self.dt * i3, 0],
                        [0, i3, -Rotation.from_quat(q) * self.skew(w_imu) * self.dt],
                        [0, 0, i3]
        ])

        return jf

    #ヤコビアンjc計算
    def calc_jc(self):
        jc = np.matrix([np.eye(3), 0, 0])

        return jc
    
    #観測値を統合
    def ekf_estimation(self, z): #z:観測値(px, py, pz)
        jc = self.calc_jc(self.x)
        y = z - jc * self.x
        
        R = 0.1 * np.eye(3)
        K = self.p * jc.T * np.linalg.inv(jc * self.p * jc.T + R) #カルマンゲイン(ここのR注意)

        self.x = self.x + K * y #状態量更新

        I = np.eye(9)
        self.p = (I - K * jc) * self.p #共分散更新

        return self.x 
