// Random numbers for layout randomisation.
//
// ae shares one std::mt19937 between threads, so a seeded run depends on thread timing
// and std::uniform_real_distribution differs between standard libraries. Here each start
// has its own generator, seeded from (run seed, start index) alone, and the uniform draw
// is written out. So a seeded run gives identical results for any number of threads, and
// identical *starting* layouts on any machine. Minimised maps from different compilers or
// maths libraries can still differ in the last bits (exp, sqrt, fused multiply-add).

#pragma once

#include <cstdint>

namespace af::map
{
    // SplitMix64 (Steele, Lea & Flood 2014; the reference code by Sebastiano Vigna).
    class SplitMix64
    {
      public:
        static constexpr std::uint64_t gamma = 0x9e3779b97f4a7c15ULL;

        explicit SplitMix64(std::uint64_t state) : state_{state} {}

        static constexpr std::uint64_t mix(std::uint64_t z)
        {
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        }

        std::uint64_t next()
        {
            state_ += gamma;
            return mix(state_);
        }

        // Uniform in [low, high): the top 53 bits of one draw.
        double uniform(double low, double high)
        {
            const double unit = static_cast<double>(next() >> 11) * 0x1.0p-53;
            return low + (high - low) * unit;
        }

      private:
        std::uint64_t state_;
    };

    // Seed of start `index` of a run seeded with `seed`: the (index + 2)-th output of
    // SplitMix64(seed). The first output is kept for the sample optimisation that sizes the
    // random layouts, so no start shares its stream.
    constexpr std::uint64_t start_seed(std::uint64_t seed, std::uint64_t index) { return SplitMix64::mix(seed + (index + 2) * SplitMix64::gamma); }
    constexpr std::uint64_t sample_seed(std::uint64_t seed) { return SplitMix64::mix(seed + SplitMix64::gamma); }

} // namespace af::map
