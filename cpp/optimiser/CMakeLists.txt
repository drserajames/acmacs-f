# af.map._core and af.map._core_original: the map optimiser (workstream 6), once per arithmetic
# (numerics.hh). Built through cpp/CMakeLists.txt, which provides the two alglibs and pybind11.
#   _core           the contract: no contraction, portable exp and log2; the same bits everywhere
#   _core_original  AF_ORIGINAL_ARITHMETIC: the compiler's defaults and the system libm, as before
#                   the contract; the same bits as earlier releases on the same machine only

set(AF_CORE_SOURCES
  bindings.cc
  grid_test.cc
  minimise.cc
  numerics.cc
  problem.cc
  reactivity.cc
  relax.cc
  stress.cc)

pybind11_add_module(_core ${AF_CORE_SOURCES})
target_compile_definitions(_core PRIVATE AF_CORE_MODULE=_core)
target_link_libraries(_core PRIVATE af::alglib ${CMAKE_DL_LIBS})   # dl: dladdr, to name the library in messages
# The same bits on every machine: cpp/CMakeLists.txt checked these flags.
target_compile_options(_core PRIVATE ${AF_STRICT_FP_OPTIONS})

pybind11_add_module(_core_original ${AF_CORE_SOURCES})
target_compile_definitions(_core_original PRIVATE AF_CORE_MODULE=_core_original AF_ORIGINAL_ARITHMETIC)
target_link_libraries(_core_original PRIVATE af::alglib_original ${CMAKE_DL_LIBS})

foreach(af_core_target _core _core_original)
  target_compile_features(${af_core_target} PRIVATE cxx_std_20)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${af_core_target} PRIVATE -Wall -Wextra)
  endif()
endforeach()

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
  target_link_libraries(_core_original PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "af optimiser: OpenMP ${OpenMP_CXX_VERSION}")
else()
  message(WARNING "af optimiser: no OpenMP; random starts will run on one thread")
endif()

install(TARGETS _core _core_original DESTINATION af/map)
