cmake_minimum_required(VERSION 3.5)
project(02-visualodometry)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./build")

# Set build type and C++ standard
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(PCL 1.2 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

link_libraries(${OpenCV_LIBS})

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

# --- Executable for visual odometry test ---
add_executable(vo
        exec/vo.cpp
        src/cam.cpp
        src/camera.cpp
        src/picp_solver.cpp
        src/my_utilities.cpp
)
target_link_libraries(vo opencv_highgui Eigen3::Eigen)

add_executable(match_points_test
        exec/match_points_test.cpp
        src/cam.cpp
        src/camera.cpp
        src/picp_solver.cpp
        src/my_utilities.cpp
)
target_link_libraries(match_points_test opencv_highgui Eigen3::Eigen)

add_executable(triangulate_points_test
        exec/triangulate_points_test.cpp
        src/cam.cpp
        src/camera.cpp
        src/picp_solver.cpp
        src/my_utilities.cpp
)
target_link_libraries(triangulate_points_test opencv_highgui Eigen3::Eigen)

add_executable(icp_test
        exec/icp_test.cpp
        src/cam.cpp
        src/camera.cpp
        src/picp_solver.cpp
        src/my_utilities.cpp
)
target_link_libraries(icp_test opencv_highgui Eigen3::Eigen)
