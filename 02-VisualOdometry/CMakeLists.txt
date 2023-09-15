# Project name
# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.5)

# Project name
project(02-visualodometry)

# Set CMake build flags
set(CMAKE_BUILD_TYPE Debug)  # or Release for optimized release builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3 -DNDEBUG")

# Find and set up dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set internal includes (your project's source files)
include_directories(src)

# Set executable output path
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build")

# Add your project's source files
set(SOURCES src/main.cpp)

# Create the executable target
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the required libraries (OpenCV)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
