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
// exp is on the minimiser's hot path (one per "<" titre per evaluation), so it has a fast path: a
// table of 2^(j/128) and a short polynomial, no division. The table and its constants were computed
// with 90-digit decimal arithmetic and rounded to nearest. exp_wide, used outside |x| <= 700, and
// the logarithm's polynomial use fdlibm's method and constants (e_exp.c, e_log.c):
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

    // exp over the whole domain, by fdlibm's method (one division). exp() below is the fast path.
    inline double exp_wide(double x)
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

    // exp(x) for |x| <= 700 without a division: x = k ln2/128 + r, exp(x) = 2^(k/128) e^r, with
    // 2^(j/128) from a table of correctly rounded doubles and e^r - 1 from a degree-5 polynomial
    // (|r| <= ln2/256, so the next term is below 6e-19 of the result). Outside that range, and for
    // NaN, exp_wide. Within 1 ulp of the system exp (tests/map/test_numerics.py).
    inline double exp(double x)
    {
        constexpr int table_size = 128;
        constexpr double inv_step = 0x1.71547652b82fep+7;  // 128 / ln2
        // ln2 / 128 in two parts; the high part has 32 significant bits, so k * step_hi is exact
        constexpr double step_hi = 0x1.62e42fee00000p-8, step_lo = 0x1.a39ef35793c76p-40;
        constexpr double c3 = 0x1.5555555555555p-3, c4 = 0x1.5555555555555p-5, c5 = 0x1.1111111111111p-7;  // 1/6, 1/24, 1/120
        static constexpr double two_to[table_size] = {
            0x1.0000000000000p+0, 0x1.0163da9fb3335p+0, 0x1.02c9a3e778061p+0, 0x1.04315e86e7f85p+0,
            0x1.059b0d3158574p+0, 0x1.0706b29ddf6dep+0, 0x1.0874518759bc8p+0, 0x1.09e3ecac6f383p+0,
            0x1.0b5586cf9890fp+0, 0x1.0cc922b7247f7p+0, 0x1.0e3ec32d3d1a2p+0, 0x1.0fb66affed31bp+0,
            0x1.11301d0125b51p+0, 0x1.12abdc06c31ccp+0, 0x1.1429aaea92de0p+0, 0x1.15a98c8a58e51p+0,
            0x1.172b83c7d517bp+0, 0x1.18af9388c8deap+0, 0x1.1a35beb6fcb75p+0, 0x1.1bbe084045cd4p+0,
            0x1.1d4873168b9aap+0, 0x1.1ed5022fcd91dp+0, 0x1.2063b88628cd6p+0, 0x1.21f49917ddc96p+0,
            0x1.2387a6e756238p+0, 0x1.251ce4fb2a63fp+0, 0x1.26b4565e27cddp+0, 0x1.284dfe1f56381p+0,
            0x1.29e9df51fdee1p+0, 0x1.2b87fd0dad990p+0, 0x1.2d285a6e4030bp+0, 0x1.2ecafa93e2f56p+0,
            0x1.306fe0a31b715p+0, 0x1.32170fc4cd831p+0, 0x1.33c08b26416ffp+0, 0x1.356c55f929ff1p+0,
            0x1.371a7373aa9cbp+0, 0x1.38cae6d05d866p+0, 0x1.3a7db34e59ff7p+0, 0x1.3c32dc313a8e5p+0,
            0x1.3dea64c123422p+0, 0x1.3fa4504ac801cp+0, 0x1.4160a21f72e2ap+0, 0x1.431f5d950a897p+0,
            0x1.44e086061892dp+0, 0x1.46a41ed1d0057p+0, 0x1.486a2b5c13cd0p+0, 0x1.4a32af0d7d3dep+0,
            0x1.4bfdad5362a27p+0, 0x1.4dcb299fddd0dp+0, 0x1.4f9b2769d2ca7p+0, 0x1.516daa2cf6642p+0,
            0x1.5342b569d4f82p+0, 0x1.551a4ca5d920fp+0, 0x1.56f4736b527dap+0, 0x1.58d12d497c7fdp+0,
            0x1.5ab07dd485429p+0, 0x1.5c9268a5946b7p+0, 0x1.5e76f15ad2148p+0, 0x1.605e1b976dc09p+0,
            0x1.6247eb03a5585p+0, 0x1.6434634ccc320p+0, 0x1.6623882552225p+0, 0x1.68155d44ca973p+0,
            0x1.6a09e667f3bcdp+0, 0x1.6c012750bdabfp+0, 0x1.6dfb23c651a2fp+0, 0x1.6ff7df9519484p+0,
            0x1.71f75e8ec5f74p+0, 0x1.73f9a48a58174p+0, 0x1.75feb564267c9p+0, 0x1.780694fde5d3fp+0,
            0x1.7a11473eb0187p+0, 0x1.7c1ed0130c132p+0, 0x1.7e2f336cf4e62p+0, 0x1.80427543e1a12p+0,
            0x1.82589994cce13p+0, 0x1.8471a4623c7adp+0, 0x1.868d99b4492edp+0, 0x1.88ac7d98a6699p+0,
            0x1.8ace5422aa0dbp+0, 0x1.8cf3216b5448cp+0, 0x1.8f1ae99157736p+0, 0x1.9145b0b91ffc6p+0,
            0x1.93737b0cdc5e5p+0, 0x1.95a44cbc8520fp+0, 0x1.97d829fde4e50p+0, 0x1.9a0f170ca07bap+0,
            0x1.9c49182a3f090p+0, 0x1.9e86319e32323p+0, 0x1.a0c667b5de565p+0, 0x1.a309bec4a2d33p+0,
            0x1.a5503b23e255dp+0, 0x1.a799e1330b358p+0, 0x1.a9e6b5579fdbfp+0, 0x1.ac36bbfd3f37ap+0,
            0x1.ae89f995ad3adp+0, 0x1.b0e07298db666p+0, 0x1.b33a2b84f15fbp+0, 0x1.b59728de5593ap+0,
            0x1.b7f76f2fb5e47p+0, 0x1.ba5b030a1064ap+0, 0x1.bcc1e904bc1d2p+0, 0x1.bf2c25bd71e09p+0,
            0x1.c199bdd85529cp+0, 0x1.c40ab5fffd07ap+0, 0x1.c67f12e57d14bp+0, 0x1.c8f6d9406e7b5p+0,
            0x1.cb720dcef9069p+0, 0x1.cdf0b555dc3fap+0, 0x1.d072d4a07897cp+0, 0x1.d2f87080d89f2p+0,
            0x1.d5818dcfba487p+0, 0x1.d80e316c98398p+0, 0x1.da9e603db3285p+0, 0x1.dd321f301b460p+0,
            0x1.dfc97337b9b5fp+0, 0x1.e264614f5a129p+0, 0x1.e502ee78b3ff6p+0, 0x1.e7a51fbc74c83p+0,
            0x1.ea4afa2a490dap+0, 0x1.ecf482d8e67f1p+0, 0x1.efa1bee615a27p+0, 0x1.f252b376bba97p+0,
            0x1.f50765b6e4540p+0, 0x1.f7bfdad9cbe14p+0, 0x1.fa7c1819e90d8p+0, 0x1.fd3c22b8f71f1p+0
        };

        if (!(x >= -700.0 && x <= 700.0))
            return exp_wide(x);
        const double scaled = x * inv_step;
        const std::int64_t k = static_cast<std::int64_t>(scaled + (scaled < 0.0 ? -0.5 : 0.5));  // nearest, ties away
        const double kd = static_cast<double>(k);
        const double r = (x - kd * step_hi) - kd * step_lo;
        const std::int64_t j = ((k % table_size) + table_size) % table_size;
        const std::int64_t power = (k - j) / table_size;  // exact: k - j is a multiple of 128
        const double t = two_to[j];
        const double p = r + r * r * (0.5 + r * (c3 + r * (c4 + r * c5)));  // e^r - 1
        const double y = t + t * p;  // in [1, 2) up to rounding
        // times 2^power through the exponent bits: |power| <= 1010, so the result stays normal
        return detail::from_bits(detail::bits_of(y) + (static_cast<std::uint64_t>(power) << 52));
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
