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

# --- floating point: the same bits on every machine ----------------------------
# A map must not depend on the machine that built it (cpp/optimiser/numerics.hh). The compiler
# must not fuse a*b+c into one rounding (arm64 does by default, baseline x86_64 cannot), and must
# not be allowed to reorder or approximate arithmetic. The flag goes on both targets below, after
# any CXXFLAGS from the environment, so it wins; the checks here fail the build otherwise.
include(CheckCXXCompilerFlag)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  message(FATAL_ERROR "af: the optimiser needs clang or gcc (it relies on -ffp-contract=off); "
                      "this compiler is ${CMAKE_CXX_COMPILER_ID}")
endif()
check_cxx_compiler_flag(-ffp-contract=off AF_HAS_FP_CONTRACT_OFF)
if(NOT AF_HAS_FP_CONTRACT_OFF)
  message(FATAL_ERROR "af: the compiler does not accept -ffp-contract=off")
endif()
set(AF_STRICT_FP_OPTIONS -ffp-contract=off)

string(TOUPPER "${CMAKE_BUILD_TYPE}" af_build_type)
set(af_given_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${af_build_type}}")
foreach(af_unsafe -ffast-math -Ofast -funsafe-math-optimizations -fassociative-math
                  -freciprocal-math -ffinite-math-only -fno-signed-zeros -fcx-limited-range)
  if(" ${af_given_flags} " MATCHES " ${af_unsafe} ")
    message(FATAL_ERROR "af: ${af_unsafe} lets the compiler change floating-point results, so maps "
                        "would differ between machines. Remove it from CXXFLAGS / CMAKE_CXX_FLAGS "
                        "(given: ${af_given_flags})")
  endif()
endforeach()

# Compile and run a*b+c under the build's own flags: it must not be fused.
if(NOT CMAKE_CROSSCOMPILING)
  unset(AF_FP_CHECK_RAN CACHE)        # never a cached verdict: the flags may have changed
  unset(AF_FP_CHECK_COMPILED CACHE)
  try_run(AF_FP_CHECK_RAN AF_FP_CHECK_COMPILED
          SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/optimiser/check_fp_contract.cc"
          COMPILE_DEFINITIONS ${AF_STRICT_FP_OPTIONS}
          CMAKE_FLAGS "-DCMAKE_CXX_FLAGS:STRING=${af_given_flags}"
          COMPILE_OUTPUT_VARIABLE af_fp_check_output)
  if(NOT AF_FP_CHECK_COMPILED)
    message(FATAL_ERROR "af: the floating-point check did not compile:\n${af_fp_check_output}")
  endif()
  if(NOT AF_FP_CHECK_RAN EQUAL 0)
    message(FATAL_ERROR "af: a*b+c is still fused with ${AF_STRICT_FP_OPTIONS} "
                        "(flags given: ${af_given_flags}); maps would differ between machines")
  endif()
  message(STATUS "af: floating point not contracted (${AF_STRICT_FP_OPTIONS}), checked by running")
endif()

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
# alglib's minimisers are part of the arithmetic, so there is one alglib per arithmetic (see
# cpp/optimiser/numerics.hh): af_alglib without contraction for the contract, af_alglib_original
# with the compiler's defaults, as every release before the contract built it. Their symbols are
# hidden, so the two extension modules cannot pick up each other's copy when both are loaded.
foreach(af_alglib_target af_alglib af_alglib_original)
  add_library(${af_alglib_target} STATIC ${AF_ALGLIB_SOURCES})
  # The public include is marked SYSTEM so alglib's own warnings don't drown ours.
  target_include_directories(${af_alglib_target} SYSTEM PUBLIC "${alglib_SOURCE_DIR}/src")
  target_compile_definitions(${af_alglib_target} PUBLIC AE_COMPILE_MINLBFGS AE_COMPILE_MINCG AE_COMPILE_PCA)
  set_target_properties(${af_alglib_target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${af_alglib_target} PRIVATE -w)
  endif()
endforeach()
add_library(af::alglib ALIAS af_alglib)
add_library(af::alglib_original ALIAS af_alglib_original)
target_compile_options(af_alglib PRIVATE ${AF_STRICT_FP_OPTIONS})

# --- Python bindings: pybind11 --------------------------------------------------
# pybind11 (workstream 6's choice): mature, good numpy array support. The pip
# package is a build requirement, and scikit-build-core puts it on the prefix path.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- OpenMP from a given prefix (macOS) ----------------------------------------
# A process must hold ONE OpenMP runtime: a second libomp aborts it ("OMP: Error #15").
# In a conda env, numpy's OpenBLAS already uses the env's libomp (llvm-openmp), so the
# optimiser must link that same one, not Homebrew's. AF_OPENMP_PREFIX (set by
# tools/make-release.py to the release's env) is searched first; when it is set and
# OpenMP isn't there, that is an error, not a silent fallback to another copy.
set(AF_OPENMP_PREFIX "" CACHE PATH "prefix holding include/omp.h and lib/libomp (macOS)")
if(APPLE AND AF_OPENMP_PREFIX)
  if(NOT EXISTS "${AF_OPENMP_PREFIX}/include/omp.h")
    message(FATAL_ERROR "AF_OPENMP_PREFIX=${AF_OPENMP_PREFIX} has no include/omp.h")
  endif()
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${AF_OPENMP_PREFIX}/include")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${AF_OPENMP_PREFIX}/lib/libomp.dylib")
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  message(STATUS "af: OpenMP from AF_OPENMP_PREFIX (${OpenMP_omp_LIBRARY})")
endif()

# --- the optimiser (workstream 6) ----------------------------------------------
add_subdirectory(optimiser)
