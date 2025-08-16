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
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/ouster/points',
        description='LiDAR point cloud topic'
    )
    
    # PRISM projection debug node
    projection_debug_node = Node(
        package='prism',
        executable='prism_projection_debug_node',
        name='projection_debug_node',
        output='screen',
        parameters=[{
            # Camera configuration - 실제 토픽 이름에 맞게 설정
            'camera_ids': ['camera_1', 'camera_2'],
            
            # Topic configuration  
            'lidar_topic': LaunchConfiguration('lidar_topic'),
            'output_topic_prefix': '/prism/projection_debug',
            
            # Calibration
            'calibration_directory': config_dir,
            
            # Synchronization (increased tolerance for bagfile replay)
            'time_tolerance_sec': 0.2,  # Increased from 0.1 to handle bagfile initial sync
            'enable_time_sync': True,
            
            # Visualization parameters
            'enable_status_overlay': True,
            'point_radius': 3,
            'overlay_alpha': 0.8,
            
            # Projection configuration
            'projection.min_depth': 0.5,
            'projection.max_depth': 50.0,
            'projection.margin_pixels': 5,
            'projection.enable_frustum_culling': True,
            'projection.enable_distortion_correction': True,
            'projection.enable_debug_visualization': True,
        }],
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
        lidar_topic_arg,
        # 압축 이미지를 raw로 변환
        image_republish_1,
        image_republish_2,
        # 투영 디버그 노드
        projection_debug_node,
    ])