cmake_minimum_required(VERSION 3.5)
project(zivid_description)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

install(
    DIRECTORY config launch meshes urdf
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
