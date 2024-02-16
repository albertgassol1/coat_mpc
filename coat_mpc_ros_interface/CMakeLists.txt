cmake_minimum_required(VERSION 2.8.3)
project(coat_mpc_ros_interface)

# Find all dependencies in package.xml and import them
find_package(catkin_simple REQUIRED)
find_package(catkin REQUIRED COMPONENTS
    rospy
)
catkin_simple(ALL_DEPS_REQUIRED)

catkin_python_setup()
catkin_package()
