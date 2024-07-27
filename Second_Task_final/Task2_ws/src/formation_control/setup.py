from setuptools import find_packages, setup
from glob import glob

package_name = 'formation_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch_folder/Task2_2.py")),
        ("share/" + package_name, glob("launch_folder/Task2_3.py")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gabriele',
    maintainer_email='gabriele@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "generic_agent = formation_control.the_agent:main",
            "plotter = formation_control.plotter:main",
            "agent_task2_3 = formation_control.agent_task2_3:main",
            "plotter_task2_3 = formation_control.plotter_task2_3:main",
        ],
    },
)
