#pragma once

// Random command sequences for the parity test. Both simulators generate the same
// sequences from the same seed.

#include <cmath>
#include <cstdint>
#include <vector>

struct ParityPlan {
    uint32_t sequences = 1024, ticks = 120, hold_ticks = 10, seed = 20261023;
    float max_speed = 1.0f, max_climb = .5f, max_yaw_rate = .5f;
};

inline uint32_t parity_next(uint32_t &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
inline float parity_uniform(uint32_t &s) { return float(parity_next(s) >> 8) * (1.0f / 16777216.0f); }

// Commands are world velocity x, y, z and yaw rate, held for hold_ticks ticks at a time.
inline std::vector<float> parity_commands(const ParityPlan &plan, uint32_t sequence) {
    uint32_t s = plan.seed ^ ((sequence + 1) * 0x9e3779b9u);
    if (!s)
        s = 1;
    for (int i = 0; i < 4; i++)
        parity_next(s);
    std::vector<float> commands(size_t(plan.ticks) * 4);
    float c[4] = {0, 0, 0, 0};
    for (uint32_t t = 0; t < plan.ticks; t++) {
        if (t % plan.hold_ticks == 0) {
            const float heading = 6.28318530718f * parity_uniform(s);
            const float speed = plan.max_speed * parity_uniform(s);
            c[0] = speed * std::cos(heading);
            c[1] = speed * std::sin(heading);
            c[2] = plan.max_climb * (2 * parity_uniform(s) - 1);
            c[3] = plan.max_yaw_rate * (2 * parity_uniform(s) - 1);
        }
        for (int j = 0; j < 4; j++)
            commands[size_t(t) * 4 + j] = c[j];
    }
    return commands;
}

// One record per tick: state after the tick, the depth frame and goal visibility seen at the
// start of the tick, and whether the tick ended in contact.
struct ParityRecord {
    float position[3], velocity[3], orientation[4];
    uint32_t alive, contact, contact_substep, visible;
    float depth[320];
};
