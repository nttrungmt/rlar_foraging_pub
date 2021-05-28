add_library(foraging_bot_controller MODULE footbot_foraging.h footbot_foraging.cpp)

target_link_libraries(foraging_bot_controller
  argos3core_simulator
  argos3plugin_simulator_footbot
  argos3plugin_simulator_genericrobot
  /opt/ros/$ENV{ROS_DISTRO}/lib/libroscpp.so)

add_library(foraging_loop_function MODULE
    foraging_loop_functions.h foraging_loop_functions.cpp
    footbot_foraging.h footbot_foraging.cpp
    id_qtuser_functions.h id_qtuser_functions.cpp
    ros_helpers.h shared_queue.hpp client.h client.cpp)
target_link_libraries(foraging_loop_function
    jsoncpp
    argos3core_simulator
    argos3plugin_simulator_footbot
    argos3plugin_simulator_genericrobot
    argos3plugin_simulator_dynamics2d
    argos3plugin_simulator_entities
    argos3plugin_simulator_footbot
    argos3plugin_simulator_genericrobot
    argos3plugin_simulator_media
    /opt/ros/$ENV{ROS_DISTRO}/lib/libroscpp.so)
