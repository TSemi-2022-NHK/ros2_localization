import rclpy
from rclpy.node import Node

def main():

    rclpy.init()

    node = Node('hellonode')

    node.get_logger().info("helloworld")

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

if __name__ == "main":
    main()    