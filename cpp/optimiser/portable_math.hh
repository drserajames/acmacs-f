// exp and log2 that give the same bits on every machine.
//
// The system libm's exp and log2 are not correctly rounded, and two libms (glibc, Apple's arm64 and
// x86_64 ones) disagree in the last bit for some arguments. The minimiser amplifies one such bit into
// a different local minimum, so a map built on two machines came out different (Sarah, 13 Oct 2026:
// "the algorithm should be reproducible regardless of machine"). These two use only IEEE-754
// + - * / on doubles and integer operations on their bits, so with fused multiply-add off
// (-ffp-contract=off, set in cpp/CMakeLists.txt) every conforming platform computes the same result.
//
// They are within 1 ulp (exp) and 2 ulp (log2) of the system functions over the ranges the optimiser
// uses (tests/map/test_numerics.py); log2 is exact for powers of two.
//
// The method and constants of exp and of the logarithm's polynomial are fdlibm's (e_exp.c, e_log.c):
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace af::map::portable
{
    namespace detail
    {
        inline std::uint64_t bits_of(double x)
        {
            std::uint64_t bits;
            std::memcpy(&bits, &x, sizeof bits);
            return bits;
        }

        inline double from_bits(std::uint64_t bits)
        {
            double x;
            std::memcpy(&x, &bits, sizeof x);
            return x;
        }
    } // namespace detail

    inline double exp(double x)
    {
        constexpr double half[2] = {0.5, -0.5};
        constexpr double huge = 1.0e+300, twom1000 = 9.33263618503218878990e-302;
        constexpr double overflow = 7.09782712893383973096e+02, underflow = -7.45133219101941108420e+02;
        constexpr double ln2_hi[2] = {6.93147180369123816490e-01, -6.93147180369123816490e-01};
        constexpr double ln2_lo[2] = {1.90821492927058770002e-10, -1.90821492927058770002e-10};
        constexpr double inv_ln2 = 1.44269504088896338700e+00;
        constexpr double p1 = 1.66666666666666019037e-01, p2 = -2.77777777770155933842e-03, p3 = 6.61375632143793436117e-05,
                         p4 = -1.65339022054652515390e-06, p5 = 4.13813679705723846039e-08;

        const std::uint64_t bits = detail::bits_of(x);
        const int sign = static_cast<int>(bits >> 63);
        const std::uint32_t high = static_cast<std::uint32_t>(bits >> 32) & 0x7fffffffU;
        const std::uint32_t low = static_cast<std::uint32_t>(bits);

        if (high >= 0x40862E42U) {  // |x| >= 709.78: overflow, underflow, inf or NaN
            if (high >= 0x7ff00000U) {
                if (((high & 0xfffffU) | low) != 0)
                    return x + x;  // NaN
                return sign == 0 ? x : 0.0;  // exp(+inf) = inf, exp(-inf) = 0
            }
            if (x > overflow)
                return huge * huge;
            if (x < underflow)
                return twom1000 * twom1000;
        }

        double hi = 0.0, lo = 0.0;
        int k = 0;
        if (high > 0x3fd62e42U) {  // |x| > 0.5 ln2: x = k ln2 + r, |r| <= 0.5 ln2
            if (high < 0x3FF0A2B2U) {  // |x| < 1.5 ln2
                hi = x - ln2_hi[sign];
                lo = ln2_lo[sign];
                k = 1 - sign - sign;
            }
            else {
                k = static_cast<int>(inv_ln2 * x + half[sign]);
                const double t = k;
                hi = x - t * ln2_hi[0];
                lo = t * ln2_lo[0];
            }
            x = hi - lo;
        }
        else if (high < 0x3e300000U) {  // |x| < 2^-28
            return 1.0 + x;
        }

        const double t = x * x;
        const double c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))));
        if (k == 0)
            return 1.0 - ((x * c) / (c - 2.0) - x);
        const double y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
        // multiply by 2^k through the exponent bits (k is within [-1075, 1024] here)
        if (k >= -1021)
            return detail::from_bits(detail::bits_of(y) + (static_cast<std::uint64_t>(static_cast<std::int64_t>(k)) << 52));
        return detail::from_bits(detail::bits_of(y) + (static_cast<std::uint64_t>(static_cast<std::int64_t>(k + 1000)) << 52)) * twom1000;
    }

    // log2 of a positive finite number. Zero, negatives, infinities and NaN go to the system log2,
    // whose results for them (-inf, NaN, +inf) are the same everywhere.
    inline double log2(double x)
    {
        constexpr double two54 = 1.80143985094819840000e+16;
        constexpr double sqrt2 = 1.41421356237309514547e+00;
        constexpr double inv_ln2 = 1.44269504088896338700e+00;
        constexpr double lg1 = 6.666666666666735130e-01, lg2 = 3.999999999940941908e-01, lg3 = 2.857142874366239149e-01,
                         lg4 = 2.222219843214978396e-01, lg5 = 1.818357216161805012e-01, lg6 = 1.531383769920937332e-01,
                         lg7 = 1.479819860511658591e-01;

        if (!(x > 0.0) || !std::isfinite(x))
            return std::log2(x);

        int k = 0;
        std::uint64_t bits = detail::bits_of(x);
        if ((bits >> 52) == 0) {  // subnormal: scale up exactly
            bits = detail::bits_of(x * two54);
            k = -54;
        }
        k += static_cast<int>(bits >> 52) - 1023;
        // the mantissa as a double in [1, 2)
        double m = detail::from_bits((bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL);
        if (m == 1.0)
            return static_cast<double>(k);  // a power of two: exact
        if (m >= sqrt2) {  // bring m into [sqrt(2)/2, sqrt(2)) so f = m - 1 is small
            m *= 0.5;
            k += 1;
        }
        const double f = m - 1.0;
        const double s = f / (2.0 + f);
        const double z = s * s;
        const double w = z * z;
        const double t1 = w * (lg2 + w * (lg4 + w * lg6));
        const double t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)));
        const double half_f_squared = 0.5 * f * f;
        const double natural = f - (half_f_squared - s * (half_f_squared + (t2 + t1)));  // log(m)
        return static_cast<double>(k) + natural * inv_ln2;
    }
} // namespace af::map::portable
