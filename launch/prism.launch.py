#!/usr/bin/env python3
"""
PRISM Fusion Node Launch File
"""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for PRISM fusion node."""
    
    # Get package share directory using pathlib for robust path handling
    prism_share_dir = Path(get_package_share_directory('prism'))
    
    # Path to default parameter file
    default_params_file = str(prism_share_dir / 'config' / 'prism_params.yaml')
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the ROS2 parameters file'
    )
    
    # PRISM fusion node (Phase 5 - Full Pipeline)
    prism_fusion_node = Node(
        package='prism',
        executable='prism_fusion_node',
        name='prism_fusion_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')]
        # remapping 제거 - 모든 토픽 이름은 config 파일에서 설정
    )
    
    # Note: image_transport republish nodes are intentionally NOT launched here.
    # Use external alias/command when playing rosbag with compressed images.
    # Projection debug node (params strictly from YAML only)
    projection_debug_node = Node(
        package='prism',
        executable='prism_projection_debug_node',
        name='projection_debug_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('/camera/camera_1/image_raw', '/usb_cam_1/image_raw'),
            ('/camera/camera_2/image_raw', '/usb_cam_2/image_raw'),
            ('/camera/camera_1/camera_info', '/usb_cam_1/camera_info'),
            ('/camera/camera_2/camera_info', '/usb_cam_2/camera_info'),
        ]
    )

    return LaunchDescription([
        params_file_arg,
        projection_debug_node,
        prism_fusion_node
    ])