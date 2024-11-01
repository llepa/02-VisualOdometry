# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Project name
project(02-visualodometry)

# Set CMake build flags
set(CMAKE_BUILD_TYPE Debug)  # or Release for optimized builds
set(CMAKE_CXX_STANDARD 17)   # C++17, adjust as necessary
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find and set up dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Set the executable output directory (modern CMake way)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")

# Add your project's source files
set(SOURCES src/main.cpp src/cam.cpp src/camera.cpp src/picp_solver.cpp src/my_utilities.cpp)

# Create the executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the required libraries (OpenCV and Eigen3)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

# Set target-specific includes
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Optionally, include your own source directory
target_include_directories(${PROJECT_NAME} PRIVATE src)
