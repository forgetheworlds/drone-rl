#pragma once

// Sensor models shared by the Metal kernels and the C++ host: a stereo-like depth camera that
// is slow, late and noisy, and a state estimate whose velocity drifts. See docs/DEPTH.md.

#include "voxel_world.h"

inline uint vx_random(VX_THREAD uint &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
inline float vx_uniform(VX_THREAD uint &s) { return float(vx_random(s) >> 8) * (1.0f / 16777216.0f); }
// Sum of four uniforms, scaled to unit variance. Close enough to normal and exact on CPU and GPU.
inline float vx_normal(VX_THREAD uint &s) {
    return (vx_uniform(s) + vx_uniform(s) + vx_uniform(s) + vx_uniform(s) - 2.0f) * 1.7320508f;
}
inline uint vx_seed(uint a, uint b) {
    uint s = a * 0x9e3779b9u ^ (b + 0x7f4a7c15u) * 0x85ebca6bu;
    s ^= s >> 15;
    return s ? s : 1u;
}

// Drawn once per episode.
struct VxSensorParams {
    uint depth_period;   // ticks between stereo frames
    uint depth_latency;  // ticks from capture to arrival
    float depth_noise;   // sigma = depth_noise * z^2
    float depth_range;   // stereo gives nothing past this
    float hole_rate;     // chance that a 4x4 patch has no depth
    float near_rate;     // chance of a false near return per ray
    float bias_walk;     // velocity bias random walk, m/s per sqrt(s)
    float velocity_noise;// white noise on the velocity estimate, m/s
    float correction;    // fraction of the bias removed when a stereo frame arrives
    float height_noise;  // downward range finder noise, m
    float height_range;  // range finder max range, m
    float gyro_noise;    // rad/s
    // Stereo visual odometry fused with the IMU (VX_VIO), see docs/DEPTH.md.
    float vo_pixel;      // disparity noise, px
    float vo_features;   // corners tracked in a well textured view
    float vo_dropout;    // chance a frame gives no odometry (Pi stall, blur, lost track)
    float vo_outlier;    // chance a frame gives a wrong displacement
    float accel_bias;    // sigma of the accelerometer bias left after calibration, m/s^2
    float accel_noise;   // accelerometer noise per 10 ms sample (vibration), m/s^2
    float drag_noise;    // rotor-drag velocity from the accelerometer, m/s per 50 ms sample
    float drag_scale;    // error of the drag coefficient, fraction of the speed
    float baro_noise;    // barometer height noise per 50 ms sample, m
};

// 1 (default): velocity from an IMU filter corrected by stereo odometry every frame.
// 0: the older model, a velocity bias that random-walks and is cut by a fixed fraction.
#ifndef VX_VIO
#define VX_VIO 1
#endif

// OV2640 pair at 320 x 240: focal length about 246 px (66 deg across), 6 cm baseline.
#define VX_VO_FOCAL_BASELINE 14.8f
#define VX_VO_MIN_RANGE .4f      // closer than this the disparity leaves the search window
#define VX_VO_MAX_RATE 2.5f      // rad/s; faster turns blur the frame
#define VX_VO_ACCEL_WALK .02f    // accelerometer bias walk, m/s^2 per sqrt(s)
#define VX_VO_DRIFT .02f         // odometry drift, fraction of distance travelled
#define VX_BARO_WALK .05f        // barometer drift, m per sqrt(s)

inline VxSensorParams vx_sensor_params(VX_THREAD uint &s, float level) {
    VxSensorParams p;
    p.depth_period = 2 + uint(vx_uniform(s) * 4 * level);
    p.depth_latency = 1 + uint(vx_uniform(s) * 5 * level);
    p.depth_noise = level * (.01f + .02f * vx_uniform(s));
    p.depth_range = 12 - level * (6 + 2 * vx_uniform(s));
    p.hole_rate = level * .08f * vx_uniform(s);
    p.near_rate = level * .004f * vx_uniform(s);
    p.bias_walk = level * (.05f + .15f * vx_uniform(s));
    p.velocity_noise = level * (.02f + .06f * vx_uniform(s));
    p.correction = 1 - level * (.4f + .5f * vx_uniform(s));
    p.height_noise = level * .02f;
    p.height_range = 2.0f;
    p.gyro_noise = level * .01f;
    p.vo_pixel = level * (.5f + .5f * vx_uniform(s));
    p.vo_features = 40 + 80 * vx_uniform(s);
    p.vo_dropout = level * (.05f + .1f * vx_uniform(s));
    p.vo_outlier = level * (.02f + .04f * vx_uniform(s));
    p.accel_bias = level * (.05f + .25f * vx_uniform(s));
    p.accel_noise = level * (.3f + .5f * vx_uniform(s));
    p.drag_noise = level * (.3f + .3f * vx_uniform(s));
    p.drag_scale = level * .15f * vx_normal(s);
    p.baro_noise = level * (.2f + .3f * vx_uniform(s));
    return p;
}

// Velocity and position estimate. Call vx_estimate_step once per 10 ms substep, vx_vo_capture
// when a stereo frame is taken and vx_estimate_correct when it arrives.
//
// VX_VIO 0: v_est = v_true + bias + noise, the bias random-walks and each frame removes a
// fixed fraction of it. Position is the integral of v_est.
//
// VX_VIO 1: the IMU integrates acceleration (true + bias + vibration) minus its estimated
// bias. Stereo odometry chains each good frame's displacement onto the last; its position
// error is the current frame's error against its keyframe plus a drift that grows with
// distance travelled. A per-axis Kalman filter over position, velocity and accelerometer bias
// fuses that position when the frame arrives (it was taken `latency` ago), gating outliers.
struct VxEstimate {
    float position[3], velocity[3], bias[3];
    float accel_bias[3], accel_bias_est[3], last_velocity[3];
    float cov[3][9];                       // per axis, over position, velocity, bias
    float drift[3], frame_error[3];        // odometry error: accumulated drift and this frame's
    float last_truth[3];                   // true position at the last good frame
    float since_capture[3];                // estimated motion since the frame in flight was taken
    float pending_meas[3], pending_sigma, interval, lag;
    float baro_offset;
    float accel_error[3];                  // from attitude error, set by the caller (attitude.h)
    float extra_walk;                      // extra bias walk the filter allows for that error, m/s^2/sqrt(s)
    uint started, pending, rejects;
};

inline void vx_estimate_reset(VX_THREAD VxEstimate &e, VX_THREAD const float *position) {
    for (uint i = 0; i < 3; ++i) {
        e.position[i] = e.last_truth[i] = position[i];
        e.drift[i] = 0;
        e.velocity[i] = 0;
        e.bias[i] = 0;
        e.accel_bias[i] = e.accel_bias_est[i] = e.last_velocity[i] = 0;
        e.frame_error[i] = e.since_capture[i] = e.pending_meas[i] = 0;
        e.accel_error[i] = 0;
        for (uint j = 0; j < 9; ++j)
            e.cov[i][j] = 0;
        e.cov[i][0] = 1e-6f;
        e.cov[i][4] = 1e-4f;
    }
    e.pending_sigma = e.interval = e.lag = e.baro_offset = e.extra_walk = 0;
    e.started = e.pending = e.rejects = 0;
}

inline void vx_estimate_step(VX_THREAD VxEstimate &e, VX_THREAD const float *true_velocity,
                             VX_THREAD const VxSensorParams &p, float dt, VX_THREAD uint &s) {
#if VX_VIO
    if (!e.started) {
        for (uint i = 0; i < 3; ++i) {
            e.accel_bias[i] = p.accel_bias * vx_normal(s);
            e.last_velocity[i] = true_velocity[i];
            e.cov[i][8] = p.accel_bias * p.accel_bias + 1e-4f;
        }
        e.started = 1;
    }
    const float walk = VX_VO_ACCEL_WALK * (p.accel_bias > 0 ? 1.0f : 0.0f);
    for (uint i = 0; i < 3; ++i) {
        e.accel_bias[i] += walk * sqrt(dt) * vx_normal(s);
        const float measured = (true_velocity[i] - e.last_velocity[i]) / dt + e.accel_bias[i] + e.accel_error[i] + p.accel_noise * vx_normal(s);
        e.last_velocity[i] = true_velocity[i];
        e.velocity[i] += (measured - e.accel_bias_est[i]) * dt;
        e.position[i] += e.velocity[i] * dt;
        e.since_capture[i] += e.velocity[i] * dt;
        // P = F P F' + Q with F = [1 dt 0; 0 1 -dt; 0 0 1].
        VX_THREAD float *c = e.cov[i];
        float f[9] = {c[0] + dt * c[3], c[1] + dt * c[4], c[2] + dt * c[5],
                      c[3] - dt * c[6], c[4] - dt * c[7], c[5] - dt * c[8],
                      c[6], c[7], c[8]};
        c[0] = f[0] + dt * f[1];
        c[1] = f[1] - dt * f[2];
        c[2] = f[2];
        c[3] = f[3] + dt * f[4];
        c[4] = f[4] - dt * f[5] + (p.accel_noise * p.accel_noise + 1e-4f) * dt * dt;
        c[5] = f[5];
        c[6] = f[6] + dt * f[7];
        c[7] = f[7] - dt * f[8];
        c[8] = f[8] + (walk * walk + e.extra_walk * e.extra_walk) * dt;
    }
    e.interval += dt;
    e.lag += dt;
#else
    const float walk = p.bias_walk * sqrt(dt);
    for (uint i = 0; i < 3; ++i) {
        // Height comes from the range finder and barometer, so vertical drift is smaller.
        const float axis = i == 2 ? .3f : 1.0f;
        e.bias[i] += walk * axis * vx_normal(s);
        e.velocity[i] = true_velocity[i] + e.bias[i] + p.velocity_noise * axis * vx_normal(s);
        e.position[i] += e.velocity[i] * dt;
    }
#endif
}

// Stereo-like depth for one ray k of a 20 x 16 frame. `true_depth` is the exact range.
// Returns VX_MAX_RANGE when there is no depth. `frame_seed` is the same for a whole frame.
inline float vx_stereo_ray(float true_depth, uint k, uint frame_seed, VX_THREAD const VxSensorParams &p) {
    uint patch = vx_seed(frame_seed, (k / 80) * 5 + (k % 20) / 4);
    if (vx_uniform(patch) < p.hole_rate)
        return VX_MAX_RANGE;
    uint s = vx_seed(frame_seed, k + 1000);
    if (vx_uniform(s) < p.near_rate)
        return .3f + 2 * vx_uniform(s);
    const float z = true_depth + p.depth_noise * true_depth * true_depth * vx_normal(s);
    return z > p.depth_range || z <= 0 ? VX_MAX_RANGE : z;
}

// How well a stereo frame supports odometry: the displacement noise in metres, or a negative
// value when odometry fails on it. `depth` are stereo ranges (VX_MAX_RANGE for none).
template <typename D>
inline float vx_vo_sigma(D depth, uint count, float turn_rate, VX_THREAD const VxSensorParams &p, VX_THREAD uint &s) {
    uint valid = 0, close = 0;
    float sum = 0, sum2 = 0, sum4 = 0;
    for (uint k = 0; k < count; ++k) {
        const float d = depth[k];
        if (d >= VX_MAX_RANGE)
            continue;
        ++valid;
        close += d < VX_VO_MIN_RANGE ? 1 : 0;
        sum += d;
        sum2 += d * d;
        sum4 += d * d * d * d;
    }
    const float random = vx_uniform(s);
    if (valid < count / 7 || close * 2 > valid || turn_rate > VX_VO_MAX_RATE || random < p.vo_dropout)
        return -1;
    const float fraction = float(valid) / count, mean = sum / valid;
    const float spread = sqrt(fmax(0.0f, sum2 / valid - mean * mean)) / mean;
    // A blank wall filling the view: few corners to track.
    float features = p.vo_features * fraction * (fraction > .6f && spread < .08f ? .25f : 1.0f);
    if (features < 15)
        return -1;
    // Range noise per corner grows with z^2; use the rms of z^2 over the frame.
    const float z2 = sqrt(sum4 / valid);
    return p.vo_pixel * (.004f + z2 / VX_VO_FOCAL_BASELINE / sqrt(features));
}

// A stereo frame is taken, at true position `truth`. `sigma` from vx_vo_sigma. The odometry
// position is held until the frame arrives.
inline void vx_vo_capture(VX_THREAD VxEstimate &e, float sigma, VX_THREAD const float *truth,
                          VX_THREAD const VxSensorParams &p, VX_THREAD uint &s) {
#if VX_VIO
    if (sigma < 0) {
        // Tracking lost for over a second: odometry restarts from the filter's position.
        if (e.interval > 1) {
            for (uint i = 0; i < 3; ++i) {
                e.drift[i] = e.position[i] - truth[i];
                e.frame_error[i] = 0;
                e.last_truth[i] = truth[i];
            }
            e.interval = 0;
        }
        return;
    }
    const bool outlier = vx_uniform(s) < p.vo_outlier;
    float travel = 0;
    for (uint i = 0; i < 3; ++i)
        travel += (truth[i] - e.last_truth[i]) * (truth[i] - e.last_truth[i]);
    travel = sqrt(travel);
    for (uint i = 0; i < 3; ++i) {
        // The frame's pose is found against a keyframe, so its error replaces the last frame's
        // instead of adding to it; moving on to new keyframes adds drift with distance.
        e.drift[i] += VX_VO_DRIFT * travel * vx_normal(s);
        e.frame_error[i] = sigma * vx_normal(s) + (outlier ? .3f * vx_normal(s) : 0.0f);
        e.pending_meas[i] = truth[i] + e.drift[i] + e.frame_error[i];
        e.last_truth[i] = truth[i];
        e.since_capture[i] = 0;
    }
    e.pending_sigma = sigma;
    e.pending = 1;
    e.lag = 0;
    e.interval = 0;
#endif
}

// One scalar measurement of axis i's position (state 0) or velocity (state 1), gated.
inline void vx_kf_update(VX_THREAD VxEstimate &e, uint i, uint state, float measured, float sigma) {
    VX_THREAD float *c = e.cov[i];
    const float value = state == 0 ? e.position[i] : e.velocity[i];
    const float sv = c[state * 4] + sigma * sigma + 1e-6f, z = measured - value;
    if (z * z > 16 * sv)
        return;
    const float n[3] = {c[state], c[3 + state], c[6 + state]};
    e.position[i] += n[0] / sv * z;
    e.velocity[i] += n[1] / sv * z;
    e.accel_bias_est[i] += n[2] / sv * z;
    for (uint a = 0; a < 3; ++a)
        for (uint b = 0; b < 3; ++b)
            c[a * 3 + b] -= n[a] * n[b] / sv;
}

// Height above the floor from the downward range finder, fused into the vertical axis.
inline void vx_height_correct(VX_THREAD VxEstimate &e, float height, float sigma) {
#if VX_VIO
    vx_kf_update(e, 2, 0, height, sigma);
#endif
}

// Once per 50 ms tick: what the flight controller has without the cameras. Rotor drag makes
// the accelerometer read a force against the horizontal airspeed (no wind here), which bounds
// the horizontal velocity error when odometry is lost; the barometer bounds height.
inline void vx_estimate_aid(VX_THREAD VxEstimate &e, VX_THREAD const float *true_velocity,
                            VX_THREAD const float *true_position, VX_THREAD const VxSensorParams &p,
                            VX_THREAD uint &s) {
#if VX_VIO
    for (uint i = 0; i < 2; ++i)
        vx_kf_update(e, i, 1, true_velocity[i] * (1 + p.drag_scale) + p.drag_noise * vx_normal(s), p.drag_noise + .02f);
    e.baro_offset += VX_BARO_WALK * (p.baro_noise > 0 ? 1.0f : 0.0f) * sqrt(.05f) * vx_normal(s);
    vx_kf_update(e, 2, 0, true_position[2] + e.baro_offset + p.baro_noise * vx_normal(s), p.baro_noise + .3f);
#endif
}

// The stereo frame arrived.
inline void vx_estimate_correct(VX_THREAD VxEstimate &e, VX_THREAD const VxSensorParams &p) {
#if VX_VIO
    (void)p;
    if (!e.pending)
        return;
    e.pending = 0;
    // The frame saw position `lag` ago: H = [1, -lag, -lag^2 / 2] over position, velocity, bias.
    const float h[3] = {1, -e.lag, -e.lag * e.lag / 2};
    const float r = e.pending_sigma * e.pending_sigma + 1e-6f;
    float z[3], n[3][3], sv[3], chi = 0;
    for (uint i = 0; i < 3; ++i) {
        VX_THREAD const float *c = e.cov[i];
        for (uint j = 0; j < 3; ++j)
            n[i][j] = c[j * 3] * h[0] + c[j * 3 + 1] * h[1] + c[j * 3 + 2] * h[2];
        sv[i] = h[0] * n[i][0] + h[1] * n[i][1] + h[2] * n[i][2] + r;
        z[i] = e.pending_meas[i] - (e.position[i] - e.since_capture[i]);
        chi += z[i] * z[i] / sv[i];
    }
    // Reject outliers, unless several frames in a row disagree (then the filter is off).
    if (chi > 16 && e.rejects < 3) {
        ++e.rejects;
        return;
    }
    e.rejects = 0;
    for (uint i = 0; i < 3; ++i) {
        const float k0 = n[i][0] / sv[i], k1 = n[i][1] / sv[i], k2 = n[i][2] / sv[i];
        e.position[i] += k0 * z[i];
        e.velocity[i] += k1 * z[i];
        e.accel_bias_est[i] += k2 * z[i];
        VX_THREAD float *c = e.cov[i];
        for (uint a = 0; a < 3; ++a)
            for (uint b = 0; b < 3; ++b)
                c[a * 3 + b] -= n[i][a] * n[i][b] / sv[i];
    }
#else
    for (uint i = 0; i < 3; ++i)
        e.bias[i] *= 1 - p.correction;
#endif
}
