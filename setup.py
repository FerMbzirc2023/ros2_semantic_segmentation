from setuptools import setup

package_name = 'ros2_semantic_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='jelena.tabak@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment = ros2_semantic_segmentation.segment:main',
            'mbzirc_semantic_segmentation = ros2_semantic_segmentation.mbzirc_semantic_segmentation:main'
        ],
    },
)
