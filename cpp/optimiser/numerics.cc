#include "numerics.hh"

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "portable_math.hh"

namespace af::map::numerics
{
    namespace
    {
        std::uint64_t bits(double x)
        {
            std::uint64_t result;
            std::memcpy(&result, &x, sizeof result);
            return result;
        }

        struct Recorded
        {
            double argument;
            std::uint64_t result;
        };
    } // namespace

    bool contracts_multiply_add()
    {
        // volatile, so the compiler cannot fold this at compile time; see check_fp_contract.cc
        volatile double a = 1.0 + 0x1p-30, b = 1.0 - 0x1p-30, c = -1.0;
        const double x = a, y = b, z = c;
        const double sum = x * y + z;
        return sum != 0.0;
    }

    bool portable_functions_as_recorded()
    {
        // A few fixed arguments and the bits they must give (the same list, and more, is in
        // tests/map/test_numerics.py). A build that returns anything else is not this arithmetic.
        static const Recorded exp_values[] = {{0.0, 0x3ff0000000000000ULL},  {1.0, 0x4005bf0a8b14576aULL},  {-1.0, 0x3fd78b56362cef38ULL},
                                              {0.5, 0x3ffa61298e1e069cULL},  {-37.25, 0x3c93278bcd70e981ULL}, {100.0, 0x48f3494a9b171bf5ULL}};
        static const Recorded log2_values[] = {{1.0, 0x0000000000000000ULL}, {2.0, 0x3ff0000000000000ULL},  {3.0, 0x3ff95c01a39fbd69ULL},
                                               {10.0, 0x400a934f0979a371ULL}, {0.7071, 0xbfe0001d03eb60d8ULL}};
        for (const auto& value : exp_values) {
            volatile double argument = value.argument;
            if (bits(portable::exp(argument)) != value.result)
                return false;
        }
        for (const auto& value : log2_values) {
            volatile double argument = value.argument;
            if (bits(portable::log2(argument)) != value.result)
                return false;
        }
        return true;
    }

    void require()
    {
        if (contracts_multiply_add())
            throw std::runtime_error{"af optimiser: this build fuses a*b+c (-ffp-contract=off was not honoured), so its maps would differ from other machines'"};
        if (!portable_functions_as_recorded())
            throw std::runtime_error{"af optimiser: portable exp or log2 does not return its recorded bits on this build, so its maps would differ from other machines'"};
    }
} // namespace af::map::numerics
