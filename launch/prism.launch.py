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
    
    # PRISM interpolation node (Phase 2)
    prism_interpolation_node = Node(
        package='prism',
        executable='prism_interpolation_node',
        name='prism_interpolation_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('/ouster/points', '/ouster/points'),
            ('/prism/interpolated_points', '/prism/interpolated_points')
        ]
    )
    
    # PRISM fusion node (will be implemented in Phase 5)
    # prism_fusion_node = Node(
    #     package='prism',
    #     executable='prism_fusion_node',
    #     name='prism_fusion_node',
    #     output='screen',
    #     parameters=[LaunchConfiguration('params_file')],
    #     remappings=[
    #         ('/ouster/points', '/ouster/points'),
    #         ('/usb_cam_1/image_raw', '/usb_cam_1/image_raw'),
    #         ('/usb_cam_2/image_raw', '/usb_cam_2/image_raw'),
    #         ('/ouster/points/colored', '/ouster/points/colored')
    #     ]
    # )
    
    return LaunchDescription([
        params_file_arg,
        prism_interpolation_node,  # Phase 2 node active
        # prism_fusion_node  # Will be added in Phase 5
    ])