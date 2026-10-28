"""Launch RoboMaster EP (AP), AprilTag 3 navigation, and annotated camera."""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    home = os.path.expanduser('~')
    python = os.path.join(home, 'rm_venv', 'bin', 'python3')
    workspace = os.path.join(home, 'ros2_ws')

    ep_launch = os.path.join(
        get_package_share_directory('robomaster_ros'),
        'launch',
        'ep.launch'
    )

    driver = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(ep_launch),

        launch_arguments={
            'name': '',
            'driver_prefix': python,
            'model': 'ep',
            'conn_type': 'ap',
            'tof_0': 'true',
            'arm': 'true',
            'arm_rate': '5',
            'camera': 'true',
            'video_raw': '1',
            'camera_calibration_file': LaunchConfiguration('camera_calibration_file'),
            'video_ffmpeg': '0',
            'with_model_description': 'true',
            'chassis_timeout': '0.45',
        }.items()
    )

    navigation = ExecuteProcess(
        cmd=[
            python,
            '-u',
            '-m',
            'rm_flatness_nav.nav_node',
            '--ros-args',

            '-p',
            'tag_id:=3',
            '-p', 'fsm.v_max:=0.16',
            '-p', ['generate_plots:=', LaunchConfiguration('generate_plots')],
            '-p', ['output_dir:=', LaunchConfiguration('output_dir')],

            '-p',
            [
                'tag_size:=',
                LaunchConfiguration('tag_size')
            ],

            '-p',
            [
                'enable_motion:=',
                LaunchConfiguration('enable_motion')
            ],

            '-p',
            [
                'fsm.search_rate:=',
                LaunchConfiguration('search_rate')
            ],

            '-p',
            [
                'fsm.arrival_width_min:=',
                LaunchConfiguration('arrival_width_min')
            ],

        ],

        cwd=workspace,
        output='screen',
        emulate_tty=True
    )

    camera_view = Node(
        package='image_view',
        executable='image_view',
        name='annotated_camera_view',

        remappings=[
            ('image', '/camera/annotated')
        ],

        condition=IfCondition(
            LaunchConfiguration('view_camera')
        ),

        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_calibration_file', default_value=os.path.join(
            workspace, 'src', 'rm_flatness_nav', 'config',
            'camera_calibration.yaml')),
        DeclareLaunchArgument('generate_plots', default_value='true'),
        DeclareLaunchArgument('output_dir', default_value=os.path.join(
            workspace, 'log', 'assessment2',
            'run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))),

        DeclareLaunchArgument(
            'enable_motion',
            default_value='true'
        ),

        DeclareLaunchArgument(
            'view_camera',
            default_value='true'
        ),

        DeclareLaunchArgument(
            'tag_size',
            default_value='0.15'
        ),

        DeclareLaunchArgument(
            'search_rate',
            default_value='0.15'
        ),

        DeclareLaunchArgument(
            'arrival_width_min',
            default_value='0.17'
        ),


        driver,
        navigation,
        camera_view,
    ])