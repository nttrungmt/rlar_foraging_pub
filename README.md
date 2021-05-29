# rlar_foraging_pub
This public source repository is accompanied with the paper "<b>Reinforcement Learning as a Rehearsal for Swarm Foraging</b>" authored by Trung T. Nguyen and Bikramjit Banerjee, 2021

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
<br>
================================================ INSTALL ROS ===========================================<br>
$ sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'<br>
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654<br>
$ sudo apt update<br>
$ sudo apt install ros-melodic-desktop<br>
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc<br>
$ source ~/.bashrc<br>
$ sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential<br>
$ sudo rosdep init<br>
$ rosdep update<br>
$ source /opt/ros/melodic/setup.bash<br>
$ source devel/setup.bash<br>
$ echo $ROS_PACKAGE_PATH<br>
$ mkdir -p ~/catkin_ws/src<br>
$ cd ~/catkin_ws/<br>
$ catkin_make<br>

================================================ INSTALL ARGOS3 ===========================================<br>
$ git clone https://github.com/ilpincy/argos3.git<br>
$ sudo apt-get install cmake libfreeimage-dev libfreeimageplus-dev qt5-default freeglut3-dev libxi-dev libxmu-dev liblua5.3-dev lua5.3 doxygen graphviz graphviz-dev asciidoc<br>
$ cd argos3<br>
$ mkdir build_simulator<br>
$ cd build_simulator<br>
$ cmake -DCMAKE_BUILD_TYPE=Release \<br>
        -DCMAKE_INSTALL_PREFIX=~/argos-dist \<br>
        -DARGOS_BUILD_FOR=simulator \<br>
        -DARGOS_BUILD_NATIVE=OFF \<br>
        -DARGOS_THREADSAFE_LOG=ON \<br>
        -DARGOS_DYNAMIC_LOADING=ON \<br>
        -DARGOS_USE_DOUBLE=ON \<br>
        -DARGOS_DOCUMENTATION=ON \<br>
        -DARGOS_INSTALL_LDSOCONF=ON \<br>
        ../src<br>
$ make<br>
$ make doc<br>
$ make install<br>

================================================ COMPILE THIS PROJECT ===========================================<br>
$ cd ~/catkin_ws/src<br>
$ git clone https://github.com/nttrungmt/rlar_foraging_pub/<br>
$ cd ..<br>
$ sudo apt-get install libgsl-dev<br>
$ sudo apt-get install libjsoncpp-dev<br>
$ catkin_make<br>
<br>
This project is licensed under the terms of GNU GENERAL PUBLIC LICENSE v3.
