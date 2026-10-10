from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nodes = ['camera', 'face_detector', 'ollama', 'whisper_og', 'voz']
    return LaunchDescription([
        Node(package='vision', executable=name, name=name, output='screen')
        for name in nodes
    ])
