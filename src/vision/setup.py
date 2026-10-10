import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'documentos'),
            glob('vision/documentos/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usuario',
    maintainer_email='usuario@todo.todo',
    description='Pipeline HRI Candidates: vision, voz y RAG',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hello_node = vision.hello_node:main',
            'camera = vision.camera:main',
            'face_detector = vision.face_detector:main',
            'identity = vision.identity:main',
            'ollama = vision.ollama:main',
            'whisper_og = vision.whisper_og:main',
            'voz = vision.voz:main',
        ],
    },
)
