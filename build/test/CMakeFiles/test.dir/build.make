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
CMAKE_SOURCE_DIR = /home/bdosremedios/dev/gmres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bdosremedios/dev/gmres/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test.dir/flags.make

test/CMakeFiles/test.dir/test.cpp.o: test/CMakeFiles/test.dir/flags.make
test/CMakeFiles/test.dir/test.cpp.o: ../test/test.cpp
test/CMakeFiles/test.dir/test.cpp.o: test/CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bdosremedios/dev/gmres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test.dir/test.cpp.o"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test.dir/test.cpp.o -MF CMakeFiles/test.dir/test.cpp.o.d -o CMakeFiles/test.dir/test.cpp.o -c /home/bdosremedios/dev/gmres/test/test.cpp

test/CMakeFiles/test.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/test.cpp.i"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bdosremedios/dev/gmres/test/test.cpp > CMakeFiles/test.dir/test.cpp.i

test/CMakeFiles/test.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/test.cpp.s"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bdosremedios/dev/gmres/test/test.cpp -o CMakeFiles/test.dir/test.cpp.s

test/CMakeFiles/test.dir/test_MatrixReader.cpp.o: test/CMakeFiles/test.dir/flags.make
test/CMakeFiles/test.dir/test_MatrixReader.cpp.o: ../test/test_MatrixReader.cpp
test/CMakeFiles/test.dir/test_MatrixReader.cpp.o: test/CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bdosremedios/dev/gmres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test.dir/test_MatrixReader.cpp.o"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test.dir/test_MatrixReader.cpp.o -MF CMakeFiles/test.dir/test_MatrixReader.cpp.o.d -o CMakeFiles/test.dir/test_MatrixReader.cpp.o -c /home/bdosremedios/dev/gmres/test/test_MatrixReader.cpp

test/CMakeFiles/test.dir/test_MatrixReader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/test_MatrixReader.cpp.i"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bdosremedios/dev/gmres/test/test_MatrixReader.cpp > CMakeFiles/test.dir/test_MatrixReader.cpp.i

test/CMakeFiles/test.dir/test_MatrixReader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/test_MatrixReader.cpp.s"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bdosremedios/dev/gmres/test/test_MatrixReader.cpp -o CMakeFiles/test.dir/test_MatrixReader.cpp.s

test/CMakeFiles/test.dir/test_GMRES.cpp.o: test/CMakeFiles/test.dir/flags.make
test/CMakeFiles/test.dir/test_GMRES.cpp.o: ../test/test_GMRES.cpp
test/CMakeFiles/test.dir/test_GMRES.cpp.o: test/CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bdosremedios/dev/gmres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object test/CMakeFiles/test.dir/test_GMRES.cpp.o"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test.dir/test_GMRES.cpp.o -MF CMakeFiles/test.dir/test_GMRES.cpp.o.d -o CMakeFiles/test.dir/test_GMRES.cpp.o -c /home/bdosremedios/dev/gmres/test/test_GMRES.cpp

test/CMakeFiles/test.dir/test_GMRES.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/test_GMRES.cpp.i"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bdosremedios/dev/gmres/test/test_GMRES.cpp > CMakeFiles/test.dir/test_GMRES.cpp.i

test/CMakeFiles/test.dir/test_GMRES.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/test_GMRES.cpp.s"
	cd /home/bdosremedios/dev/gmres/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bdosremedios/dev/gmres/test/test_GMRES.cpp -o CMakeFiles/test.dir/test_GMRES.cpp.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/test.cpp.o" \
"CMakeFiles/test.dir/test_MatrixReader.cpp.o" \
"CMakeFiles/test.dir/test_GMRES.cpp.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test/test: test/CMakeFiles/test.dir/test.cpp.o
test/test: test/CMakeFiles/test.dir/test_MatrixReader.cpp.o
test/test: test/CMakeFiles/test.dir/test_GMRES.cpp.o
test/test: test/CMakeFiles/test.dir/build.make
test/test: lib/libgtest_main.a
test/test: lib/libgtest.a
test/test: test/CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bdosremedios/dev/gmres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable test"
	cd /home/bdosremedios/dev/gmres/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test.dir/build: test/test
.PHONY : test/CMakeFiles/test.dir/build

test/CMakeFiles/test.dir/clean:
	cd /home/bdosremedios/dev/gmres/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test.dir/clean

test/CMakeFiles/test.dir/depend:
	cd /home/bdosremedios/dev/gmres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bdosremedios/dev/gmres /home/bdosremedios/dev/gmres/test /home/bdosremedios/dev/gmres/build /home/bdosremedios/dev/gmres/build/test /home/bdosremedios/dev/gmres/build/test/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test.dir/depend

