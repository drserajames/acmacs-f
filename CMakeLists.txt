# Build entry point for scikit-build-core.
#
# LANGUAGES NONE: the top level needs no compiler. cpp/CMakeLists.txt enables C++
# only when the optimiser has sources, so until then `pip install` is pure Python.
cmake_minimum_required(VERSION 3.26)
project(acmacs_f LANGUAGES NONE)

add_subdirectory(cpp)
