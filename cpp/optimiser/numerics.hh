// The optimiser's arithmetic contract: the same input gives the same bits on every machine.
//
// Two things broke that (measured 13 Oct 2026, notes in the PR): the compiler fusing a*b+c into
// one rounding where the CPU has the instruction (arm64 by default, baseline x86_64 never), and the
// system exp and log2 differing in the last bit between libms. The minimiser amplifies one bit of
// difference into a different local minimum. So this build must not contract (-ffp-contract=off,
// cpp/CMakeLists.txt) and must use portable_math.hh, and it checks both when the module is loaded.

#pragma once

namespace af::map::numerics
{
    // Names the arithmetic. The digit goes up whenever results can change bit for bit: the
    // compiler flags, the portable functions, the order of the sums, the alglib version (pinned by
    // hash in cpp/CMakeLists.txt) or how alglib is built. Anything that reuses or compares
    // optimiser output sees only this name, so it must tell two arithmetics apart.
    // Exposed to Python as af.map._core.NUMERICS.
    inline constexpr const char* id = "strict-fp-1";

    // True when this build fuses a*b+c: the flag was not honoured.
    bool contracts_multiply_add();

    // True when exp and log2 return the recorded bits for a few fixed arguments.
    bool portable_functions_as_recorded();

    // Throws std::runtime_error naming what is wrong if the build does not keep the contract.
    void require();
} // namespace af::map::numerics
