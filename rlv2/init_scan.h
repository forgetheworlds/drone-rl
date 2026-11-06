#pragma once

// Scripted start (RLV2_INIT_SCAN=1 in grade, --rlv2-init-scan in the planner adapter): climb
// straight up, hold there and turn slowly through 1.25 turns so the stereo camera sees every
// direction with overlap, settle, then hand over to the normal controller. Works on the
// estimated position and velocity only. One tick is 0.05 s.
//
// Camera: 66 deg across, frames every 0.1-0.25 s. At 1 rad/s a frame moves 6-14 deg, so
// each direction is in view for 4 to 10 frames. Climb about 1 s, turn 8 s, settle 0.5 s.

#include <cmath>

#define IS_YAW_RATE 1.0f
inline float is_climb = .5f, is_turns = 1.25f, is_settle = .5f;  // m, turns, s

struct InitScan {
    int phase;  // 0 climb, 1 turn, 2 settle, 3 done
    int ticks, near;
    float target_z;
    float last_yaw, turned;
};

inline void is_reset(InitScan &s, const float *start, float yaw) {
    s.phase = 0;
    s.ticks = 0;
    s.near = 0;
    s.target_z = start[2] + is_climb;
    s.last_yaw = yaw;
    s.turned = 0;
}

// Writes a world-frame velocity and yaw rate and returns true while the scan is running.
// Only the climb is a velocity command; otherwise it commands zero and lets the flight
// controller hold its reference, so the noisy estimate is not fed back twice.
inline bool is_step(InitScan &s, const float *position, float yaw, float *command) {
    if (s.phase == 3)
        return false;
    command[0] = command[1] = command[2] = command[3] = 0;
    float d = yaw - s.last_yaw;
    d = std::atan2(std::sin(d), std::cos(d));
    s.last_yaw = yaw;
    ++s.ticks;
    if (s.phase == 0) {
        const float gap = s.target_z - position[2];
        command[2] = std::fmin(.4f, std::fmax(-.4f, 1.5f * gap));
        s.near = std::fabs(gap) < .08f ? s.near + 1 : 0;
        if (s.near >= 4 || s.ticks > 60)
            s.phase = 1, s.ticks = 0;
    } else if (s.phase == 1) {
        s.turned += d;
        command[3] = IS_YAW_RATE * std::fmin(1.0f, s.ticks / 6.0f);
        if (s.turned >= is_turns * 6.2831853f)
            s.phase = 2, s.ticks = 0;
    } else if (s.ticks >= int(std::lround(is_settle / .05f))) {
        s.phase = 3;
        return false;
    }
    return true;
}
