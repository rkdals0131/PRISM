#!/usr/bin/env python3
"""
PRISM Fusion Node Launch File
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for PRISM fusion node."""
    
    # Get package share directory
    prism_share_dir = get_package_share_directory('prism')
    
    # Path to default parameter file
    default_params_file = os.path.join(
        prism_share_dir, 'config', 'prism_params.yaml'
    )
    
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
    # Optional projection debug node (enabled via arg)
    enable_debug_arg = DeclareLaunchArgument(
        'enable_projection_debug',
        default_value='true',
        description='Launch projection debug node alongside fusion'
    )

    projection_debug_node = Node(
        package='prism',
        executable='prism_projection_debug_node',
        name='projection_debug_node',
        output='screen',
        parameters=[{
            'camera_ids': ['camera_1', 'camera_2'],
            'lidar_topic': '/ouster/points',
            'output_topic_prefix': '/prism/projection_debug',
            'calibration_directory': os.path.join(prism_share_dir, 'config'),
            'time_tolerance_sec': 0.2,
            'enable_time_sync': True,
            'enable_status_overlay': True,
            'point_radius': 3,
            'overlay_alpha': 0.8,
            'projection.min_depth': 0.5,
            'projection.max_depth': 100.0,
            'projection.margin_pixels': 5,
            'projection.enable_frustum_culling': True,
            'projection.enable_distortion_correction': True,
            'projection.enable_debug_visualization': True,
        }],
        remappings=[
            ('/camera/camera_1/image_raw', '/usb_cam_1/image_raw'),
            ('/camera/camera_2/image_raw', '/usb_cam_2/image_raw'),
            ('/camera/camera_1/camera_info', '/usb_cam_1/camera_info'),
            ('/camera/camera_2/camera_info', '/usb_cam_2/camera_info'),
        ],
        condition=None  # always include; enable/disable by parameter internally if needed
    )

    return LaunchDescription([
        params_file_arg,
        enable_debug_arg,
        projection_debug_node,
        prism_fusion_node
    ])