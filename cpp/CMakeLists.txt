# The C++ hook: vendored alglib, then the optimiser extension (workstream 6).
#
# The optimiser lives in cpp/optimiser/ with its own CMakeLists.txt. It links
# `af::alglib` and pybind11, handles OpenMP itself (optional), and installs the
# extension `af.map._core` with install(TARGETS _core DESTINATION af/map).
# While cpp/optimiser/ does not exist this file does nothing and needs no compiler.
#
# alglib is fetched at a pinned version and checked against its SHA-256, the same
# tarball ae uses (ae/subprojects/alglib.wrap). Offline builds (e.g. an HPC node
# with no internet) point CMake at a local copy, and the hash is still checked:
#   pip install . -Ccmake.define.AF_ALGLIB_URL=/path/to/alglib-3.19.0.cpp.gpl.tgz
# or at an already unpacked tree (no hash check):
#   pip install . -Ccmake.define.FETCHCONTENT_SOURCE_DIR_ALGLIB=/path/to/alglib-cpp

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/optimiser/CMakeLists.txt")
  message(STATUS "af: no cpp/optimiser yet; pure-Python build")
  return()
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- alglib (GPL-2.0-or-later) -------------------------------------------------
set(AF_ALGLIB_URL "https://www.alglib.net/translator/re/alglib-3.19.0.cpp.gpl.tgz"
    CACHE STRING "alglib source tarball: URL or local path")
set(AF_ALGLIB_SHA256 "fbe7855f35a6a7f84538485ef7768c5442faf65e066edf85563797d65185f40f")

include(FetchContent)
FetchContent_Declare(alglib
  URL "${AF_ALGLIB_URL}"
  URL_HASH "SHA256=${AF_ALGLIB_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(alglib)   # alglib ships no CMakeLists; we only need its sources

# Only the units the minimisers (L-BFGS, CG) and PCA need, plus their #include
# closure; interpolation, integration, FFT and ODE units are left out.
set(AF_ALGLIB_UNITS ap alglibinternal alglibmisc linalg solvers optimization
    dataanalysis statistics specialfunctions kernels_sse2 kernels_avx2 kernels_fma)
set(AF_ALGLIB_SOURCES "")
foreach(unit IN LISTS AF_ALGLIB_UNITS)
  set(source "${alglib_SOURCE_DIR}/src/${unit}.cpp")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "af: alglib source missing: ${source}")
  endif()
  list(APPEND AF_ALGLIB_SOURCES "${source}")
endforeach()
add_library(af_alglib STATIC ${AF_ALGLIB_SOURCES})
add_library(af::alglib ALIAS af_alglib)
# The public include is marked SYSTEM so alglib's own warnings don't drown ours.
target_include_directories(af_alglib SYSTEM PUBLIC "${alglib_SOURCE_DIR}/src")
target_compile_definitions(af_alglib PUBLIC AE_COMPILE_MINLBFGS AE_COMPILE_MINCG AE_COMPILE_PCA)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(af_alglib PRIVATE -w)
endif()

# --- Python bindings: pybind11 --------------------------------------------------
# pybind11 (workstream 6's choice): mature, good numpy array support. The pip
# package is a build requirement, and scikit-build-core puts it on the prefix path.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- the optimiser (workstream 6) ----------------------------------------------
add_subdirectory(optimiser)
