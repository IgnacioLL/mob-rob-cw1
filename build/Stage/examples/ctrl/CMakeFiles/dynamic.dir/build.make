# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage

# Include any dependencies generated for this target.
include examples/ctrl/CMakeFiles/dynamic.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/ctrl/CMakeFiles/dynamic.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/ctrl/CMakeFiles/dynamic.dir/progress.make

# Include the compile flags for this target's objects.
include examples/ctrl/CMakeFiles/dynamic.dir/flags.make

examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o: examples/ctrl/CMakeFiles/dynamic.dir/flags.make
examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o: /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/examples/ctrl/dynamic.cc
examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o: examples/ctrl/CMakeFiles/dynamic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o"
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o -MF CMakeFiles/dynamic.dir/dynamic.cc.o.d -o CMakeFiles/dynamic.dir/dynamic.cc.o -c /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/examples/ctrl/dynamic.cc

examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dynamic.dir/dynamic.cc.i"
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/examples/ctrl/dynamic.cc > CMakeFiles/dynamic.dir/dynamic.cc.i

examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dynamic.dir/dynamic.cc.s"
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/examples/ctrl/dynamic.cc -o CMakeFiles/dynamic.dir/dynamic.cc.s

# Object files for target dynamic
dynamic_OBJECTS = \
"CMakeFiles/dynamic.dir/dynamic.cc.o"

# External object files for target dynamic
dynamic_EXTERNAL_OBJECTS =

examples/ctrl/dynamic.so: examples/ctrl/CMakeFiles/dynamic.dir/dynamic.cc.o
examples/ctrl/dynamic.so: examples/ctrl/CMakeFiles/dynamic.dir/build.make
examples/ctrl/dynamic.so: libstage/libstage.so.4.3.0
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libGL.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libGLU.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libltdl.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libpng.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libz.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libGL.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libGLU.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libfltk_images.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libfltk_forms.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libfltk_gl.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libfltk.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libSM.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libICE.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libX11.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libXext.so
examples/ctrl/dynamic.so: /usr/lib/x86_64-linux-gnu/libm.so
examples/ctrl/dynamic.so: examples/ctrl/CMakeFiles/dynamic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module dynamic.so"
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dynamic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/ctrl/CMakeFiles/dynamic.dir/build: examples/ctrl/dynamic.so
.PHONY : examples/ctrl/CMakeFiles/dynamic.dir/build

examples/ctrl/CMakeFiles/dynamic.dir/clean:
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl && $(CMAKE_COMMAND) -P CMakeFiles/dynamic.dir/cmake_clean.cmake
.PHONY : examples/ctrl/CMakeFiles/dynamic.dir/clean

examples/ctrl/CMakeFiles/dynamic.dir/depend:
	cd /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/src/Stage/examples/ctrl /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl /afs/inf.ed.ac.uk/user/s27/s2761803/Desktop/MOB/mob-rob-cw1/build/Stage/examples/ctrl/CMakeFiles/dynamic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/ctrl/CMakeFiles/dynamic.dir/depend

