from glob import glob
from setuptools import find_packages, setup

setup(
    name='rm_flatness_nav',
    version='0.2.0',
    packages=find_packages(),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/rm_flatness_nav']
        ),
        (
            'share/rm_flatness_nav',
            ['package.xml']
        ),
        (
            'share/rm_flatness_nav/launch',
            glob('launch/*.launch.py')
        )
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='student',
    maintainer_email='student@example.com',

    description=(
        'AprilTag 3 pursuit with confirmed arrival '
        'and AP RoboMaster ROS driver'
    ),

    license='MIT',

    entry_points={
        'console_scripts': [
            'nav_node = rm_flatness_nav.nav_node:main',
            'calibrate_camera = rm_flatness_nav.calibrate_camera:main'
        ]
    }
)