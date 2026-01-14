cmake_minimum_required(VERSION 3.13)  # CMake version check
project(simple_example)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

# Setting project and application name
set(project_name "MLPCpp")
set(app_name ${project_name})
project(${project_name})

# Adding source code directories
set(APP_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${APP_INCLUDE_DIRS})
