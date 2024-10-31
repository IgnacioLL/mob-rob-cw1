# Install script for directory: /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/install/Stage")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/stage/worlds" TYPE FILE FILES
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/amcl-sonar.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/autolab.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/camera.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/everything.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/lsp_test.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/mbicp.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/nd.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/roomba.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/simple.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/test.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/uoa_robotics_lab.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/vfh.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/wavefront-remote.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/wavefront.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/wifi.cfg"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/SFU.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/autolab.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/camera.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/circuit.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/everything.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/fasr.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/fasr2.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/fasr_plan.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/large.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/lsp_test.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/mbicp.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/pioneer_flocking.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/pioneer_follow.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/pioneer_walle.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/roomba.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/sensor_noise_demo.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/sensor_noise_module_demo.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/simple.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/uoa_robotics_lab.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/wifi.world"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/beacons.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/chatterbox.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/hokuyo.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/irobot.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/map.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/objects.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/pantilt.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/pioneer.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/sick.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/ubot.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/uoa_robotics_lab_models.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/walle.inc"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/cfggen.sh"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/test.sh"
    "/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/worlds/worldgen.sh"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/worlds/benchmark/cmake_install.cmake")
  include("/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/worlds/bitmaps/cmake_install.cmake")
  include("/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/worlds/wifi/cmake_install.cmake")

endif()

