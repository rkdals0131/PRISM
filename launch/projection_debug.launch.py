#!/usr/bin/env python3
"""PRISM Projection Debug Node Launch File for visualizing LiDAR projections on camera images."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Generate launch description for PRISM projection debug node."""
    
    # Get package share directory
    prism_share_dir = get_package_share_directory('prism')
    config_dir = os.path.join(prism_share_dir, 'config')
    
    # Declare launch arguments
    params_file_default = os.path.join(prism_share_dir, 'config', 'prism_params.yaml')
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=params_file_default,
        description='Path to the ROS2 parameters YAML (controls all settings)'
    )
    
    # PRISM projection debug node
    projection_debug_node = Node(
        package='prism',
        executable='prism_projection_debug_node',
        name='projection_debug_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            # 실제 카메라 토픽에 맞게 리매핑
            ('/camera/camera_1/image_raw', '/usb_cam_1/image_raw'),
            ('/camera/camera_2/image_raw', '/usb_cam_2/image_raw'),
            ('/camera/camera_1/camera_info', '/usb_cam_1/camera_info'),
            ('/camera/camera_2/camera_info', '/usb_cam_2/camera_info'),
        ]
    )
    
    # Image transport republish nodes for compressed images
    # usb_cam_1 압축 해제
    image_republish_1 = Node(
        package='image_transport',
        executable='republish',
        name='image_republish_usb_cam_1',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', '/usb_cam_1/image_raw/compressed'),
            ('out', '/usb_cam_1/image_raw')
        ]
    )
    
    # usb_cam_2 압축 해제
    image_republish_2 = Node(
        package='image_transport',
        executable='republish',
        name='image_republish_usb_cam_2',
        arguments=['compressed', 'raw'],
        remappings=[
            ('in/compressed', '/usb_cam_2/image_raw/compressed'),
            ('out', '/usb_cam_2/image_raw')
        ]
    )
    
    return LaunchDescription([
        params_file_arg,
        # 압축 이미지를 raw로 변환
        image_republish_1,
        image_republish_2,
        # 투영 디버그 노드
        projection_debug_node,
    ])