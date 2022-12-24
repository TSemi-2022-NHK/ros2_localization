import rclpy
from ekf_lzp.ekf_node import ekf_node

def main(args=None):
    rclpy.init(args=args)

    node = ekf_node.EkfNode() #ekfノード起動

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()