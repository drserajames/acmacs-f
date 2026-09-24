# Build entry point for scikit-build-core.
#
# Until the C++ map optimiser lands under cpp/ (workstream 7) there is nothing to
# compile: LANGUAGES NONE means no compiler is needed and the package installs as
# pure Python. When cpp/CMakeLists.txt exists it is picked up without changing
# anything here; it installs its extension module into ${SKBUILD_PLATLIB_DIR}/af.
cmake_minimum_required(VERSION 3.26)
project(acmacs_f LANGUAGES NONE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
  enable_language(CXX)
  add_subdirectory(cpp)
endif()
