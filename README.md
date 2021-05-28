# rlar_foraging_pub
This public source repository is accompanied with the paper "<b>Reinforcement Learning as a Rehearsal for Swarm Foraging</b>" authored by Trung T. Nguyen and Bikramjit Banerjee

<h1>Dependencies</h1>
<h2>Argos 3</h2>
Argos 3 must be installed from https://github.com/ilpincy/argos3.
<h2>ROS</h2>
ROS Kinetic is required which can be found at http://wiki.ros.org/kinetic/Installation
<h2>Python Dependencies</h2>
Note that ROS does not support Python 3, so all python dependencies must be installed for Python 2.
<h2>Directory Listing</h2>
<h3>argos_worlds</h3>
Define the foraging world settings in ARGOS3
<h3>msg</h3>
Define the Observation object format to be used to communicate by ARGOS3 environment, ROS and Python bindings code
<h3>plugin</h3>
Implement robot controller and loop functions of foraging experiments (ARGOS3 simulation) 
<h3>run</h3>
Bash scripts to execute the experiments
<h3>scripts</h3>
Implement reinforcement learning code for making decision in foraging and MDN code to train rehearsal features
<h3>srv</h3>
Implement AIService to communicate between Python code and ARGOS3 experiment C++ code through ROS service
================================================ INSTALL ROS ===========================================
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-melodic-desktop
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
$ sudo rosdep init
$ rosdep update
$ source /opt/ros/melodic/setup.bash
$ source devel/setup.bash
$ echo $ROS_PACKAGE_PATH
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
================================================ INSTALL ARGOS3 ===========================================
$ git clone https://github.com/ilpincy/argos3.git
$ sudo apt-get install cmake libfreeimage-dev libfreeimageplus-dev qt5-default freeglut3-dev libxi-dev libxmu-dev liblua5.3-dev lua5.3 doxygen graphviz graphviz-dev asciidoc
$ cd argos3
$ mkdir build_simulator
$ cd build_simulator
$ cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=~/argos-dist \
        -DARGOS_BUILD_FOR=simulator \
        -DARGOS_BUILD_NATIVE=OFF \
        -DARGOS_THREADSAFE_LOG=ON \
        -DARGOS_DYNAMIC_LOADING=ON \
        -DARGOS_USE_DOUBLE=ON \
        -DARGOS_DOCUMENTATION=ON \
        -DARGOS_INSTALL_LDSOCONF=ON \
        ../src
$ make
$ make doc
$ make install
================================================ COMPILE THIS PROJECT ===========================================
$ cd ~/catkin_ws/src
$ git clone https://github.com/nttrungmt/ma_foraging/
$ cd ..
$ sudo apt-get install libgsl-dev
$ sudo apt-get install libjsoncpp-dev
$ catkin_make
This project is licensed under the terms of GNU GENERAL PUBLIC LICENSE v3.
