# af.map._core: the map optimiser (workstream 6). Built through cpp/CMakeLists.txt,
# which provides af::alglib and pybind11.

pybind11_add_module(_core
  bindings.cc
  grid_test.cc
  minimise.cc
  problem.cc
  relax.cc
  stress.cc)
target_link_libraries(_core PRIVATE af::alglib)
target_compile_features(_core PRIVATE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(_core PRIVATE -Wall -Wextra)
endif()

# OpenMP runs the random starts in parallel. It is optional: without it the results are
# identical (every start has its own seed), only slower. Apple clang has no OpenMP of its
# own; Homebrew's libomp provides it.
if(APPLE AND NOT OpenMP_CXX_FOUND)
  foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
    if(EXISTS "${prefix}/include/omp.h")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${prefix}/include")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${prefix}/lib/libomp.dylib")
      break()
    endif()
  endforeach()
endif()
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "af optimiser: OpenMP ${OpenMP_CXX_VERSION}")
else()
  message(WARNING "af optimiser: no OpenMP; random starts will run on one thread")
endif()

install(TARGETS _core DESTINATION af/map)
