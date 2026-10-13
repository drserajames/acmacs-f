// The optimiser's arithmetic contract: the same input gives the same bits on every machine.
//
// Two things broke that (measured 13 Oct 2026, notes in the PR): the compiler fusing a*b+c into
// one rounding where the CPU has the instruction (arm64 by default, baseline x86_64 never), and the
// system exp and log2 differing in the last bit between libms. The minimiser amplifies one bit of
// difference into a different local minimum. So this build must not contract (-ffp-contract=off,
// cpp/CMakeLists.txt) and must use portable_math.hh, and it checks both when the module is loaded.

//
// The ORIGINAL arithmetic is kept as a second compiled module, af.map._core_original (Sarah, 13 Oct
// 2026: "please keep the original optimiser (non-reproducible on a different machine) as an
// option"). It is these same sources compiled with AF_ORIGINAL_ARITHMETIC: the compiler's default
// contraction, the system exp and log2, and the code paths as they were before the contract, so it
// gives the bits earlier releases gave on the same machine. It has no name: its results belong to
// one machine and compiler. Everything that differs between the two is behind that one macro.

#pragma once

#include <cmath>

#include "portable_math.hh"

namespace af::map::numerics
{
#ifdef AF_ORIGINAL_ARITHMETIC
    inline double exp(double x) { return std::exp(x); }
    inline double log2(double x) { return std::log2(x); }
#else
    inline double exp(double x) { return portable::exp(x); }
    inline double log2(double x) { return portable::log2(x); }
#endif

    // Names the arithmetic. The digit goes up whenever results can change bit for bit: the
    // compiler flags, the portable functions, the order of the sums, the alglib version (pinned by
    // hash in cpp/CMakeLists.txt) or how alglib is built. Anything that reuses or compares
    // optimiser output sees only this name, so it must tell two arithmetics apart.
    // Exposed to Python as af.map._core.NUMERICS.
#ifdef AF_ORIGINAL_ARITHMETIC
    inline constexpr const char* id = nullptr;  // the original: no name
#else
    inline constexpr const char* id = "strict-fp-1";
#endif

    // True when this build fuses a*b+c: the flag was not honoured.
    bool contracts_multiply_add();

    // True when exp and log2 return the recorded bits for a few fixed arguments.
    bool portable_functions_as_recorded();

    // Throws std::runtime_error naming what is wrong if the build does not keep the contract.
    void require();
} // namespace af::map::numerics
