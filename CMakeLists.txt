cmake_minimum_required(VERSION 3.16)
project(prism)

# Compiler configuration
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
  # Optional native optimization
  option(PRISM_ENABLE_NATIVE_OPT "Enable native CPU optimizations" OFF)
  if(PRISM_ENABLE_NATIVE_OPT)
    add_compile_options(-march=native)
  endif()
endif()


# C++17 requirement for advanced features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)

# Third-party dependencies
find_package(PCL 1.10 REQUIRED COMPONENTS common io kdtree search filters)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Core library - Phase 1 & 2 components
add_library(${PROJECT_NAME}_core SHARED
  src/core/point_cloud_soa.cpp
  src/core/memory_pool.cpp
  src/core/calibration_manager.cpp
  src/execution_mode.cpp
  # Phase 2 interpolation components
  src/interpolation/interpolation_engine.cpp
  src/interpolation/catmull_rom_interpolator.cpp
  src/interpolation/simd_kernels.cpp
  src/interpolation/beam_altitude_manager.cpp
  # Future Phase 2+ components:
  # src/projection/projection_manager.cpp
  # src/fusion/prism_fusion.cpp
)

target_link_libraries(${PROJECT_NAME}_core
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  yaml-cpp
)


# Node executables
# Phase 2 Interpolation Node
add_executable(prism_interpolation_node
  src/nodes/prism_interpolation_node.cpp
)

target_link_libraries(prism_interpolation_node
  ${PROJECT_NAME}_core
  ${PCL_LIBRARIES}
  ${rclcpp_LIBRARIES}
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(prism_interpolation_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

# Phase 5 Fusion Node (will be added later)
# add_executable(prism_fusion_node
#   src/nodes/prism_fusion_node.cpp
# )

# target_link_libraries(prism_fusion_node
#   ${PROJECT_NAME}_core
# )

# Dependencies for ROS2 components
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  pcl_ros
  pcl_conversions
  message_filters
)

# Installation
install(TARGETS
  ${PROJECT_NAME}_core
  DESTINATION lib
)

install(TARGETS
  prism_interpolation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Testing configuration
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  ament_lint_auto_find_test_dependencies()
  
  # Phase 1 Unit tests
  ament_add_gtest(test_memory_pool
    test/unit/test_memory_pool.cpp
  )
  target_link_libraries(test_memory_pool
    ${PROJECT_NAME}_core
  )
  
  ament_add_gtest(test_calibration_manager
    test/unit/test_calibration_manager.cpp
  )
  target_link_libraries(test_calibration_manager
    ${PROJECT_NAME}_core
  )
  
  ament_add_gtest(test_execution_mode
    test/unit/test_execution_mode.cpp
  )
  target_link_libraries(test_execution_mode
    ${PROJECT_NAME}_core
  )
  
  # Phase 2 Interpolation tests
  ament_add_gtest(test_interpolation_engine
    test/unit/test_interpolation_engine.cpp
  )
  target_link_libraries(test_interpolation_engine
    ${PROJECT_NAME}_core
  )
  
  # Phase 2+ Unit tests (will be added in later phases)
  # ament_add_gtest(test_point_cloud_soa
  #   test/unit/test_point_cloud_soa.cpp
  # )
  # target_link_libraries(test_point_cloud_soa
  #   ${PROJECT_NAME}_core
  # )
  
  # ament_add_gtest(test_cubic_interpolation
  #   test/unit/test_cubic_interpolation.cpp
  # )
  # target_link_libraries(test_cubic_interpolation
  #   ${PROJECT_NAME}_core
  # )
  
  # Benchmark tests (will be added in later phases)
  # add_executable(benchmark_interpolation
  #   test/benchmark/benchmark_interpolation.cpp
  # )
  # target_link_libraries(benchmark_interpolation
  #   ${PROJECT_NAME}_core
  # )
endif()

ament_package()
