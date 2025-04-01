# @author Nico Schroeder
# this script is extracting point cloud frames from the rosbag
# but it's for ros2 bag files (should be ending with .db3)
# https://docs.openvins.com/dev-ros1-to-ros2.html
# steps to use:
#### 1. pip3 install rosbags>=0.9.11
#### 2. rosbags-convert <ros1.bag> --dst <ros2_bag_folder>
#### 3. python3 scripts/ros2bag2ply.py -i <ros2_bag.db3> -o <output/ply> -t /os_cloud_node/points -p


import os
import argparse
import numpy as np
import open3d as o3d
import rosbag2_py
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from rclpy.serialization import deserialize_message

from module import ply

def rosbag2ply(args):
    os.makedirs(args.output_folder, 0o755, exist_ok=True)

    if args.output_pcd:
        output_folder_pcd = args.output_folder + "_pcd"
        os.makedirs(output_folder_pcd, 0o755, exist_ok=True)

    print("Start extraction")

    storage_options = rosbag2_py.StorageOptions(uri=args.input_bag, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    begin_flag = False
    shift_timestamp = None

    total_num_plys = 0

    while reader.has_next():
        topic, serialized_msg, t = reader.read_next()

        if topic == args.topic:
            msg = deserialize_message(serialized_msg, PointCloud2)
            if not isinstance(msg, PointCloud2):
                raise TypeError("Deserialized message is not a PointCloud2 message")

            points = point_cloud2.read_points_list(msg, field_names=None, skip_nans=True)
            array = np.array([[p.x, p.y, p.z, p.intensity, p.t] for p in points])

            timestamps = array[:, 4] # for hilti, vbr, and others

            if not begin_flag:
                shift_timestamp = timestamps[0]
                begin_flag = True

            timestamps_shifted = timestamps - shift_timestamp

            field_names = ['x', 'y', 'z', 'intensity', 'timestamp']
            ply_file_path = os.path.join(args.output_folder, str(t) + ".ply")
            if ply.write_ply(ply_file_path, [array[:, :4], timestamps_shifted], field_names):
                total_num_plys += 1
                print(f"Export ply {total_num_plys}: {ply_file_path}")
            else:
                print("ply.write_ply() failed")

            if args.output_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(array[:, :3])
                pcd_file_path = os.path.join(output_folder_pcd, str(t) + ".pcd")
                o3d.io.write_point_cloud(pcd_file_path, pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_bag', required=True, help="Path to the input ROS 2 bag file")
    parser.add_argument('-o', '--output_folder', required=True, help="Path for output folder")
    parser.add_argument('-t', '--topic', required=True, help="Name of the point cloud topic in the ROS 2 bag")
    parser.add_argument('-p', '--output_pcd', action='store_true', help="Also output the PCD file")
    args = parser.parse_args()
    print("Usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to output folder] -t [topic name]")
    
    rosbag2ply(args)
