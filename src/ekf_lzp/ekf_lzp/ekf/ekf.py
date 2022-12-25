import numpy as np
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation
import quaternion #クオータニオン積計算用、こいつにも結構注意

g = np.matrix([0, 0, 9.80665]).T

#カルマンフィルタ処理を行う
class Ekf:
    def __init__(self, initial_x = np.matrix(np.zeros(10)).T, initial_p = np.eye(9) * 100):
        self.x = initial_x #初期状態 np.matrix(x, y, z, vx, vy, vz, qx,qy, qz, qw).T　= (p v q).T
                           # = (位置座標 速度 クオータニオン)
        self.p = initial_p #共分散

        self.Q = np.eye(3) #適当な初期化
        self.dt = 0.1 #imuからの時間、適当な初期化
        self.current_time_imu = 0
        self.previous_time_imu = 0


    def skew(self, w): #3次元ベクトルを歪対称行列に変換するだけの関数
        V = np.matrix([[0, -w[2, 0], w[1, 0]],
                      [w[2, 0], 0, -w[0, 0]],
                      [-w[1, 0], w[0, 0], 0]
        ])

        return V


    def motion_model(self, u): #動作モデル
        p = np.matrix([self.x[0, 0], self.x[1, 0], self.x[2, 0]]).T
        v = np.matrix([self.x[3, 0], self.x[4, 0], self.x[5, 0]]).T
        q = np.matrix([self.x[6, 0], self.x[7, 0], self.x[8, 0], self.x[9, 0]]) #クオータニオンは基本転地しない

        #timestep
        self.current_time_imu = u.header.stamp.sec + u.header.stamp.nanosec * 1e-9
        self.dt = self.current_time_imu - self.previous_time_imu
        self.previous_time_imu = self.current_time_imu

        a_imu = np.matrix([u.linear_acceleration.x, u.linear_acceleration.y, u.linear_acceleration.z]).T
        w_imu = np.matrix([u.angular_velocity.x, u.angular_velocity.y, u.angular_velocity.z]).T

        #wdt_quatの作成
        rot_x = Rotation.from_rotvec([u.angular_velocity.x * self.dt, 0, 0]).as_matrix()
        rot_y = Rotation.from_rotvec([0, u.angular_velocity.y * self.dt, 0]).as_matrix()
        rot_z = Rotation.from_rotvec([0, 0, u.angular_velocity.z * self.dt]).as_matrix()
        rot = rot_x * rot_y * rot_z
        wdt_quat = Rotation.from_matrix(rot).as_quat() #wdt合成クオータニオン(転地する必要あるかも?)

        #predict処理
        if np.all(q == 0):
            q = np.matrix([0, 0, 0, 1])
        
        q = q.squeeze()
        q = q.tolist() #Rotation.from_quatに渡すときのデータ型注意
        Ro = Rotation.from_quat(q).as_matrix() #回転行列

        #更新処理
        v = v + (Ro * a_imu - g) * self.dt
        p = p + v * self.dt + 0.5 * (Ro * a_imu - g) * self.dt * self.dt
        q = np.array(q) #わざわざリストに変換してからまた戻すというあれ
        q = quaternion.as_quat_array(q)
        wdt_quat = quaternion.as_quat_array(wdt_quat)
        q = wdt_quat * q[0] #クオータニオン積
        
 
        #Qの計算もここでしておく
        self.Q = np.zeros((6, 6))
        self.Q[0:3, 0:3] = np.eye(3) * 0.33 #分散は0.33に設定
        self.Q[3:6, 3:6] = np.eye(3) * 0.33
        self.Q = np.matrix(self.Q) #ちゃんとmatrixにする
    
        self.Q = self.Q * self.dt * self.dt

        return np.matrix([p[0,0],p[1,0],p[2,0], v[0,0],v[1,0],v[2,0], q.w,q.x,q.y,q.z]).T  
    

    #予測値を計算
    def predict(self, u): #uはimuからの情報
        self.x = self.motion_model(u) #ここでdtも計算

        jf = self.calc_jf(self.x, u)

        L = np.zeros((9, 6))
        L[3:6, 0:3] = np.eye(3)
        L[6:9, 3:6] = np.eye(3)
        L = np.matrix(L)

        self.p = jf * self.p * jf.T + L * self.Q * L.T

        return self.x, self.p
    

    #ヤコビアンjf計算
    def calc_jf(self, x, u):
        w_imu = np.matrix([u.angular_velocity.x, u.angular_velocity.y, u.angular_velocity.z]).T
        q = np.matrix([x[6, 0], x[7, 0], x[8, 0], x[9, 0]])

        if np.all(q == 0):
            q = np.matrix([0, 0, 0, 1])
        
        q = q.squeeze()
        q = q.tolist() #Rotation.from_quatに渡すときのデータ型注意
    
        jf = np.zeros((9, 9))
        jf[0:3, 0:3] = np.eye(3)
        jf[0:3, 3:6] = np.eye(3) + self.dt
        jf[3:6, 3:6] = np.eye(3)
        jf[3:6, 6:9] = Rotation.from_quat(q).as_matrix() * self.skew(w_imu) * self.dt * -1
        jf[6:9, 6:9] = np.eye(3)

        jf = np.matrix(jf)

        return jf


    #ヤコビアンjc計算
    def calc_jc(self):
        jc = np.zeros((3, 9))
        jc[0:3, 0:3] = np.eye(3)

        jc = np.matrix(jc)

        return jc
    

    #観測値を統合
    def ekf_estimation(self, z): #z:観測値(px, py, pz)
        jc = self.calc_jc()
        y = z - np.matrix([self.x[0, 0], self.x[1, 0], self.x[2, 0]])
        
        R = 0.1 * np.eye(3)
        K = self.p * jc.T * np.linalg.inv(jc * self.p * jc.T + R) #カルマンゲイン(ここのR注意)

        dx =  K * y.T #誤差状態量(dx, dy, dz, dvx, dvy, dvz, dθx, dθy, dθz).T

        #状態量更新
        #p
        self.x[0, 0] = self.x[0, 0] + dx[0, 0]
        self.x[1, 0] = self.x[1, 0] + dx[1, 0]
        self.x[2, 0] = self.x[2, 0] + dx[2, 0]
        #v
        self.x[3, 0] = self.x[3, 0] + dx[3, 0]
        self.x[4, 0] = self.x[4, 0] + dx[4, 0]
        self.x[5, 0] = self.x[5, 0] + dx[5, 0]
        #q
        rot_q = Rotation.from_rotvec([dx[6, 0], dx[7, 0], dx[8, 0]]).as_quat()
        q = np.matrix([self.x[6, 0], self.x[7, 0], self.x[8, 0], self.x[9, 0]])

        if np.all(q == 0):
            q = np.matrix([0, 0, 0, 1])

        q = quaternion.as_quat_array(q)
        rot_q = quaternion.as_quat_array(rot_q)
        
        q = q[0] * rot_q

        self.x[6, 0] = q.w
        self.x[7, 0] = q.x
        self.x[8, 0] = q.y
        self.x[9, 0] = q.z

        I = np.eye(9)
        self.p = (I - K * jc) * self.p #共分散更新

        return self.x 
