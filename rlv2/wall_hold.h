#pragma once
// Wall-relative holding. While the commanded speed is small (hover, yield, turn in place) the
// drone keeps its distance to the surfaces it sees instead of its estimated position: the
// stereo frame taken when the hold starts is the keyframe, each later frame is aligned to it
// (point to plane, 20 x 16 rays) and the displacement found is fed back as a small velocity.
// Host side only; uses the same stereo rays the odometry sees.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "nav_task.h"

#define WH_SPEED .1f      // hold when the command is slower than this, m/s
#define WH_NEAR 1.5f      // and something is closer than this, m
#define WH_RANGE 3.0f     // use rays closer than this, m
#define WH_GAIN 1.0f      // 1/s
#define WH_MAX .3f        // largest correction, m/s
#define WH_MIN_MATCHES 20
#define WH_SIGMA .08f     // assumed range noise, m at 1 m (grows with range squared)
#define WH_PRIOR .1f      // trust in the odometry motion since the keyframe, m

struct WallFrame {
    float point[VX_RAYS][3];  // ray end points, world axes, relative to the camera
    float range[VX_RAYS];
    uint8_t valid[VX_RAYS];
    float rotation[9];
    float estimate[3];        // estimated position when taken
};

struct WallHold {
    WallFrame key, pending;
    float latest[VX_RAYS];    // last stereo ranges taken
    float base[3];            // keyframe position from where the hold started (true, as measured)
    float offset[3];          // displacement from the keyframe at the last aligned frame
    float estimate_at_offset[3];
    float target[3];          // where the slow command wants to be, from the hold start
    uint pending_ticks, failures;
    bool holding, has_key;
};

inline void wh_reset(WallHold &h) {
    h.pending_ticks = h.failures = 0;
    for (uint k = 0; k < VX_RAYS; ++k)
        h.latest[k] = VX_MAX_RANGE;
    h.holding = h.has_key = false;
}

inline VxVec wh_ray(uint k) {
    return vx_unit(vx_vec(1, 1 - (k % 20 + .5f) / 10, .75f * (1 - (k / 20 + .5f) / 8)));
}

inline void wh_frame(WallFrame &f, const float *stereo, const float *rotation, const float *estimate) {
    std::copy(rotation, rotation + 9, f.rotation);
    for (uint k = 0; k < VX_RAYS; ++k) {
        const VxVec p = vx_rotate(rotation, vx_scale(wh_ray(k), stereo[k]));
        f.point[k][0] = p.x, f.point[k][1] = p.y, f.point[k][2] = p.z;
        f.range[k] = stereo[k];
        f.valid[k] = stereo[k] < WH_RANGE && stereo[k] > VX_VO_MIN_RANGE;
    }
    for (uint i = 0; i < 3; ++i)
        f.estimate[i] = estimate[i];
}

// Displacement of frame f from the keyframe. `delta` holds the guess, which is also the
// prior; returns false when too few rays match.
inline bool wh_align(const WallHold &h, const WallFrame &f, float *delta) {
    const float prior[3] = {delta[0], delta[1], delta[2]};
    for (int iteration = 0; iteration < 4; ++iteration) {
        float a[9] = {0}, b[3] = {0};
        uint matches = 0;
        for (uint k = 0; k < VX_RAYS; ++k) {
            if (!f.valid[k])
                continue;
            const VxVec q = vx_vec(f.point[k][0] + delta[0], f.point[k][1] + delta[1], f.point[k][2] + delta[2]);
            const VxVec c = vx_inverse_rotate(h.key.rotation, q);
            if (c.x < .1f)
                continue;
            // Nearest keyframe ray: column (1 - y/x) * 10 - .5, row (1 - z/.75x) * 8 - .5.
            const int col = int(std::floor((1 - c.y / c.x) * 10));
            const int row = int(std::floor((1 - c.z / (.75f * c.x)) * 8));
            if (col < 1 || col > 18 || row < 1 || row > 14)
                continue;
            const uint m = uint(row * 20 + col);
            if (!h.key.valid[m] || !h.key.valid[m + 1] || !h.key.valid[m - 1] || !h.key.valid[m + 20] || !h.key.valid[m - 20])
                continue;
            const float *p0 = h.key.point[m], *px = h.key.point[m + 1], *pw = h.key.point[m - 1];
            const float *py = h.key.point[m + 20], *pv = h.key.point[m - 20];
            const VxVec u = vx_vec(px[0] - pw[0], px[1] - pw[1], px[2] - pw[2]);
            const VxVec v = vx_vec(py[0] - pv[0], py[1] - pv[1], py[2] - pv[2]);
            const VxVec n = vx_unit(vx_vec(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x));
            const float r = n.x * (q.x - p0[0]) + n.y * (q.y - p0[1]) + n.z * (q.z - p0[2]);
            if (std::fabs(r) > .3f || vx_norm(u) > 1.5f || vx_norm(v) > 1.5f)
                continue;
            // Stereo range noise grows with the square of the range, in both frames.
            const float d0 = vx_norm(vx_vec(p0[0], p0[1], p0[2])), d1 = f.range[k];
            const float w = 1 / (WH_SIGMA * WH_SIGMA * (d0 * d0 * d0 * d0 + d1 * d1 * d1 * d1) + 1e-4f);
            const float nn[3] = {n.x, n.y, n.z};
            for (uint i = 0; i < 3; ++i) {
                b[i] -= w * nn[i] * r;
                for (uint j = 0; j < 3; ++j)
                    a[i * 3 + j] += w * nn[i] * nn[j];
            }
            ++matches;
        }
        if (matches < WH_MIN_MATCHES)
            return false;
        // Directions the walls do not pin down stay near the prior (the odometry's motion).
        const float lambda = 1 / (WH_PRIOR * WH_PRIOR);
        for (uint i = 0; i < 3; ++i) {
            a[i * 4] += lambda;
            b[i] += lambda * (prior[i] - delta[i]);
        }
        const float det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                          a[2] * (a[3] * a[7] - a[4] * a[6]);
        if (std::fabs(det) < 1e-9f)
            return false;
        const float s[3] = {
            (b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2]) + a[2] * (b[1] * a[7] - a[4] * b[2])) / det,
            (a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * b[2] - b[1] * a[6])) / det,
            (a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6]) + b[0] * (a[3] * a[7] - a[4] * a[6])) / det};
        for (uint i = 0; i < 3; ++i)
            delta[i] += s[i];
    }
    return true;
}

inline void wh_set_key(WallHold &h, const WallFrame &f) {
    h.key = f;
    h.has_key = true;
    h.failures = 0;
    for (uint i = 0; i < 3; ++i) {
        h.offset[i] = 0;
        h.estimate_at_offset[i] = f.estimate[i];
    }
}

// Call each tick after nt_observe and nt_command. `rays` are the true depth rays of this tick,
// `captured` is true when nt_observe took a stereo frame. Adds the correction to `command`.
inline void wh_update(WallHold &h, const TrainEnv &e, const float *rays, bool captured, float *command) {
    bool arrived = h.pending_ticks > 0 && --h.pending_ticks == 0;
    if (arrived && h.holding) {
        if (!h.has_key) {
            wh_set_key(h, h.pending);
        } else {
            // Start from (and fall back to) the odometry's motion since the keyframe, so the
            // directions the walls do not pin down follow the odometry rather than piling up.
            float delta[3];
            for (uint i = 0; i < 3; ++i)
                delta[i] = h.pending.estimate[i] - h.key.estimate[i];
            if (wh_align(h, h.pending, delta)) {
                for (uint i = 0; i < 3; ++i) {
                    h.offset[i] = delta[i];
                    h.estimate_at_offset[i] = h.pending.estimate[i];
                }
                h.failures = 0;
            } else if (++h.failures >= 3) {
                // Lost the keyframe (turned away): carry the displacement over to a new one.
                for (uint i = 0; i < 3; ++i)
                    h.base[i] += delta[i];
                wh_set_key(h, h.pending);
            }
        }
    }
    if (captured) {
        float stereo[VX_RAYS], rotation[9];
        for (uint k = 0; k < VX_RAYS; ++k)
            stereo[k] = vx_stereo_ray(rays[k], k, e.pending_seed, e.sensor);
        vx_rotation(nt_attitude(e), rotation);
        wh_frame(h.pending, stereo, rotation, e.estimate.position);
        std::copy(stereo, stereo + VX_RAYS, h.latest);
        h.pending_ticks = e.sensor.depth_latency;
    }
    const float speed = std::sqrt(command[0] * command[0] + command[1] * command[1] + command[2] * command[2]);
    float nearest = VX_MAX_RANGE;
    for (uint k = 0; k < VX_RAYS; ++k)
        nearest = std::fmin(nearest, h.latest[k]);
    if (speed >= WH_SPEED || nearest > WH_NEAR) {
        h.holding = h.has_key = false;
        return;
    }
    if (!h.holding) {
        h.holding = true;
        h.has_key = false;
        for (uint i = 0; i < 3; ++i)
            h.base[i] = h.target[i] = 0;
    }
    if (!h.has_key)
        return;
    float correction[3], size = 0;
    for (uint i = 0; i < 3; ++i) {
        h.target[i] += command[i] * .05f;
        const float now = h.base[i] + h.offset[i] + e.estimate.position[i] - h.estimate_at_offset[i];
        correction[i] = -WH_GAIN * (now - h.target[i]);
        size += correction[i] * correction[i];
    }
    const float scale = size > WH_MAX * WH_MAX ? WH_MAX / std::sqrt(size) : 1.0f;
    for (uint i = 0; i < 3; ++i)
        command[i] += correction[i] * scale;
}
