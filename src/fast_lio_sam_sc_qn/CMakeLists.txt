cmake_minimum_required(VERSION 3.10)
project(fast_lio_sam_sc_qn LANGUAGES CXX)
set(FAST_LIO_SAM_WORKSPACE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# --------------------------
# Build type & C++ standard
# --------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 常用编译/链接选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra ${CMAKE_CXX_FLAGS_RELEASE}")

# --------------------------
# OpenMP
# --------------------------
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP not found; continuing without OpenMP.")
endif()

# --------------------------
# Dependencies
# --------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  rosbag
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf
  tf_conversions
  pcl_ros
  pcl_conversions
  message_filters
  nano_gicp
  cv_bridge            # <--- 新增
  image_transport      # <--- 建议顺带加上
  message_generation
)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)

# Torch (libtorch)
set(Torch_DIR "" CACHE PATH "Path to the libtorch CMake package directory (the folder that contains TorchConfig.cmake)")
if(Torch_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")
elseif(DEFINED ENV{Torch_DIR} AND NOT "$ENV{Torch_DIR}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{Torch_DIR}")
endif()
find_package(Torch REQUIRED)

# 让所有目标都继承 Torch 的编译选项（如 _GLIBCXX_USE_CXX11_ABI）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_message_files(
  FILES
  BevFrame.msg            
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)


# --------------------------
# catkin package
# --------------------------
catkin_package(
  CATKIN_DEPENDS
    roscpp roslib rosbag std_msgs geometry_msgs nav_msgs sensor_msgs
    visualization_msgs tf tf_conversions pcl_ros pcl_conversions
    message_filters nano_gicp
    cv_bridge image_transport     # <--- 新增
)

# --------------------------
# Include directories
# --------------------------
include_directories(
  include
  ${FAST_LIO_SAM_WORKSPACE_SRC_DIR}/third_party/LighterBEV
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

# --------------------------
# External subprojects
# --------------------------
# LighterBEV
# ⚠️ 请在 third_party/LighterBEV/CMakeLists.txt 中将
# add_library(LighterBEVManager MODULE …)
# 改为：add_library(LighterBEVManager STATIC …)
set(LighterBEV_SOURCE_DIR ${FAST_LIO_SAM_WORKSPACE_SRC_DIR}/third_party/LighterBEV)
set(LighterBEV_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/LighterBEV)
add_subdirectory(${LighterBEV_SOURCE_DIR} ${LighterBEV_BINARY_DIR})

# --------------------------
# Targets
# --------------------------
set(COMMON_LIBS
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  gtsam
)

add_library(${PROJECT_NAME}_modules SHARED
  src/loop_closure.cpp
  src/fast_lio_sam_sc_qn.cpp
)
set_target_properties(${PROJECT_NAME}_modules PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${PROJECT_NAME}_modules
  ${COMMON_LIBS}
  ${TORCH_LIBRARIES}
  LighterBEVManager
)
add_dependencies(${PROJECT_NAME}_modules
  ${PROJECT_NAME}_generate_messages_cpp
  ${catkin_EXPORTED_TARGETS}
)

add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${COMMON_LIBS}
  ${PROJECT_NAME}_modules
)
add_dependencies(${PROJECT_NAME}_node
  ${PROJECT_NAME}_generate_messages_cpp
  ${catkin_EXPORTED_TARGETS}
)

# --------------------------
# Diagnostics (optional)
# --------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libs   : ${TORCH_LIBRARIES}")
