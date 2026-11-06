#pragma once

// The navigation task shared by the Metal trainer and the C++ grader: per-env state, episode
// reset, sensors, observations, the command interface and rewards. The world queries and the
// vehicle (RAPTOR plus L2F) live with the caller, so the grader can use drone-rl's own code.

#include "sensors.h"
#include "attitude.h"

#define NT_BINS 40
#define NT_PROXIMITY 14
#define NT_PROXIMITY_RANGE 2.0f
#define NT_TEACHER_OBS 90
#define NT_PATH 16
#define NT_STUDENT_OBS 111
#define NT_ACTIONS 4
#define NT_GOAL_TICKS 800
#define NT_MARGIN .45f
#define NT_CONTACT_PENALTY 30

struct NavTaskConfig {
    float max_speed, max_climb, max_yaw_rate, sensor_level;
    uint env_count, mission_count, seed, shield;
    uint attitude;  // 1: the drone works with its own attitude estimate (attitude.h)
};

struct TrainEnv {
    RLPhysicsState physics;
    float hidden[16], previous_motor[4], reference[3], reference_yaw;
    VxEstimate estimate;
    VxSensorParams sensor;
    float frame[NT_BINS], previous_frame[NT_BINS], pending[NT_BINS];
    float goal_offset[3];
    float action[4];
    float goal_distance, min_depth, clearance, reward, episode_return;
    uint mission, goal, goal_count, steps;
    uint tick, goal_tick, settle, frame_age;
    float carrot[4];
    uint pending_ticks, rng, done, path_index;
    uint pending_seed;
    uint episodes, contacts, arrivals, completions;
    VxAttitude attitude;   // what the drone believes its attitude is (attitude.h)
    uint arrive_estimate;  // 1: the drone declares arrival from its own estimate
    float arrive_error;    // true distance to the waypoint at the last declared arrival
    float dyn[4];          // per-episode vehicle scales: mass, thrust, spin-up lag, spin-down lag (1 = nominal)
};

inline float nt_yaw(VX_THREAD const float *q) {
    return atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}
// The attitude the drone works with: its estimate when the estimator is on, else the truth.
inline VX_THREAD const float *nt_attitude(VX_THREAD const TrainEnv &e) {
    return e.attitude.on ? e.attitude.q : e.physics.orientation_wxyz;
}
inline VxVec nt_to_yaw_frame(VxVec v, float yaw) {
    const float c = cos(yaw), s = sin(yaw);
    return vx_vec(c * v.x + s * v.y, -s * v.x + c * v.y, v.z);
}
inline float nt_clip(float x, float limit) { return fmin(limit, fmax(-limit, x)); }

// Starts an episode at a mission start. hover_rpm comes from the vehicle parameters.
inline void nt_reset(VX_THREAD TrainEnv &e, VX_THREAD const float *start, float yaw, uint goal_count,
                     float hover_rpm, float sensor_level) {
    for (uint i = 0; i < 3; ++i) {
        e.physics.position[i] = start[i];
        e.physics.linear_velocity[i] = 0;
        e.physics.angular_velocity_body[i] = 0;
        e.reference[i] = start[i];
        e.goal_offset[i] = 0;
    }
    e.physics.orientation_wxyz[0] = cos(yaw / 2);
    e.physics.orientation_wxyz[1] = 0;
    e.physics.orientation_wxyz[2] = 0;
    e.physics.orientation_wxyz[3] = sin(yaw / 2);
    for (uint i = 0; i < 4; ++i) {
        e.physics.rpm[i] = hover_rpm;
        e.previous_motor[i] = 0;
        e.action[i] = 0;
    }
    e.reference_yaw = yaw;
    vx_estimate_reset(e.estimate, e.physics.position);
    va_reset(e.attitude, e.physics.orientation_wxyz);
    e.arrive_estimate = 0;
    e.arrive_error = 0;
    e.sensor = vx_sensor_params(e.rng, sensor_level);
#ifndef __METAL_VERSION__
    // RLV2_DEPTH_PERIOD / RLV2_DEPTH_LATENCY (ticks of 50 ms): fixed stereo rate and latency for a
    // target host (round 8: 2 / 2 = 10 Hz depth, 100 ms capture-to-use on an ESP32-P4-class board).
    if (const char *v = std::getenv("RLV2_DEPTH_PERIOD"))
        e.sensor.depth_period = uint(std::atoi(v));
    if (const char *v = std::getenv("RLV2_DEPTH_LATENCY"))
        e.sensor.depth_latency = uint(std::atoi(v));
#endif
    if (e.sensor.depth_latency > e.sensor.depth_period)
        e.sensor.depth_latency = e.sensor.depth_period;
    for (uint i = 0; i < NT_BINS; ++i)
        e.frame[i] = e.previous_frame[i] = e.pending[i] = VX_MAX_RANGE;
    e.goal = 0;
    e.goal_count = goal_count;
    e.steps = e.tick = e.goal_tick = e.settle = 0;
    e.frame_age = 20;
    e.pending_ticks = 0;
    e.goal_distance = -1;
    e.path_index = 0;
    e.min_depth = VX_MAX_RANGE;
    e.clearance = NT_PROXIMITY_RANGE;
    e.episode_return = 0;
    e.done = 0;
}

// The VLM hands over a waypoint in the drone's own (drifting) frame.
inline void nt_refresh_goal(VX_THREAD TrainEnv &e) {
    for (uint i = 0; i < 3; ++i)
        e.goal_offset[i] = e.estimate.position[i] - e.physics.position[i];
}

// Min-pools 320 rays into 10 x 4 bins of 2 x 4 rays.
// Stereo range for ray k of a frame. grade.cpp overrides it for the measured preset
// (depth_preset.h); everything else, the Metal trainer included, uses vx_stereo_ray.
#ifndef NT_STEREO_RAY
#define NT_STEREO_RAY(rays, k, seed, p) vx_stereo_ray(rays[k], k, seed, p)
#endif

template <typename R> inline void nt_pool(R rays, VX_THREAD float *bins, bool stereo, uint frame_seed,
                                          VX_THREAD const VxSensorParams &p) {
    for (uint b = 0; b < NT_BINS; ++b) {
        const uint col0 = (b % 10) * 2, row0 = (b / 10) * 4;
        float m = VX_MAX_RANGE;
        for (uint r = 0; r < 4; ++r)
            for (uint c = 0; c < 2; ++c) {
                const uint k = (row0 + r) * VX_RAY_COLS + col0 + c;
                const float d = stereo ? NT_STEREO_RAY(rays, k, frame_seed, p) : rays[k];
                m = fmin(m, d);
            }
        bins[b] = m;
    }
}

inline float nt_depth_feature(float d) { return fmin(d, 6.0f) / 6.0f; }

// Student depth feature for a pooled bin. grade.cpp / cpu_env.hpp can override it so that a bin
// whose 8 rays all returned nothing (VX_MAX_RANGE: stereo hole) is not encoded like far free space.
#ifndef NT_STUDENT_DEPTH
#define NT_STUDENT_DEPTH(d) nt_depth_feature(d)
#endif

// Privileged for the teacher: distance to the static world along 14 directions around the
// body (the six axes and eight diagonals, in the yaw frame), capped at NT_PROXIMITY_RANGE.
inline void nt_proximity(VX_DEV const VxWorld &w, VX_DEV const uint *bits, VxVec p, float yaw,
                         VX_THREAD float *out) {
    const float c = cos(yaw), s = sin(yaw), k = .57735027f;
    for (uint i = 0; i < NT_PROXIMITY; ++i) {
        VxVec d;
        if (i < 6) {
            const float sign = i % 2 ? -1 : 1;
            d = vx_vec(i / 2 == 0 ? sign : 0, i / 2 == 1 ? sign : 0, i / 2 == 2 ? sign : 0);
        } else {
            const uint j = i - 6;
            d = vx_vec(j & 1 ? -k : k, j & 2 ? -k : k, j & 4 ? -k : k);
        }
        const VxVec world = vx_vec(c * d.x - s * d.y, s * d.x + c * d.y, d.z);
        out[i] = vx_ray_static(w, bits, p, world, NT_PROXIMITY_RANGE);
    }
}

// Call once per tick, before the policy acts. `rays` are the true ranges at this instant,
// `goal` is the current waypoint in world coordinates, `mover` the nearest mover's position
// and velocity, `proximity` the output of nt_proximity and `path` this leg's reference path
// from paths.hpp (all privileged, teacher only). Advances the stereo pipeline.
// How close the drone must get to a path corner before it aims at the next one.
#ifndef NT_CORNER
#define NT_CORNER .35f
#endif

// The VLM can point at the waypoint when it is in the camera view and not hidden.
// How far ahead the student's route target sits, m.
#define NT_ROUTE_REACH 2.0f
#ifndef NT_ROUTE_SPEED
#define NT_ROUTE_SPEED 1.0f
#endif

// The student's prior: fly at the route target, facing it, braking as it gets close.
// Slots 100..103 of the student observation hold the target in the yaw frame and a found flag.
// Without a route the target is only a small push off nearby walls, and the drone turns.
inline void nt_student_prior(VX_THREAD const float *s, VX_THREAD float *prior) {
    const VxVec t = vx_vec(s[100] * NT_ROUTE_REACH, s[101] * NT_ROUTE_REACH, s[102] * NT_ROUTE_REACH);
    const float distance = vx_norm(t), flat = sqrt(t.x * t.x + t.y * t.y);
    const float error = flat > .15f ? atan2(t.y, t.x) : 0.0f;
    float speed = fmin(NT_ROUTE_SPEED, sqrt(2 * fmax(0.0f, distance - .05f))) * fmax(0.0f, cos(error));
    if (s[103] < .5f)
        speed = fmin(.4f, distance);
    const VxVec u = vx_scale(t, speed / fmax(distance, 1e-3f));
    prior[0] = u.x / 2;
    prior[1] = u.y / 2;
    prior[2] = u.z / .75f;
    prior[3] = (s[103] < .5f ? 1.0f : fmin(1.5f, fmax(-1.5f, 2 * error))) / 1.5f;
}

// The student network only nudges its prior: it cannot see what the teacher sees, and copying
// the teacher's full actions made it fly faster than its own view allows.
#ifndef NT_STUDENT_RESIDUAL
#define NT_STUDENT_RESIDUAL .1f
#endif
inline float nt_student_residual(float x) { return fmin(NT_STUDENT_RESIDUAL, fmax(-NT_STUDENT_RESIDUAL, x)); }

// The residual moves the drone only as much as the prior already does, so it cannot add
// drift while the prior holds still. The yaw residual is not gated.
#ifndef NT_RESIDUAL_GATE
#define NT_RESIDUAL_GATE .15f
#endif
inline void nt_student_combine(VX_THREAD const float *prior, VX_THREAD const float *residual, VX_THREAD float *action) {
    const float moving = sqrt(prior[0] * prior[0] + prior[1] * prior[1] + prior[2] * prior[2]);
    const float gate = fmin(1.0f, moving / NT_RESIDUAL_GATE);
    for (uint j = 0; j < NT_ACTIONS; ++j)
        action[j] = prior[j] + nt_student_residual(residual[j]) * (j < 3 ? gate : 1.0f);
#if NT_RESIDUAL_NO_FASTER
    const float speed = sqrt(action[0] * action[0] + action[1] * action[1] + action[2] * action[2]);
    if (speed > moving)
        for (uint j = 0; j < 3; ++j)
            action[j] *= moving / speed;
#endif
}

template <typename R> inline bool nt_goal_visible(VX_THREAD TrainEnv &e, R rays, VxVec goal) {
    float r[9];
    vx_rotation(e.physics.orientation_wxyz, r);
    const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
    const VxVec b = vx_inverse_rotate(r, vx_sub(goal, p));
    const float distance = vx_norm(b);
    if (b.x < .3f || distance > 8)
        return false;
    float u, v;
    if (!vx_cam_uv(b, .75f, u, v))
        return false;
    if (u < 0 || v < 0 || u > 19 || v > 15)
        return false;
    return rays[uint(round(v)) * 20 + uint(round(u))] > distance - .3f;
}

template <typename R, typename P>
inline void nt_observe(VX_THREAD TrainEnv &e, R rays, VxVec goal, VX_THREAD const float *mover, bool has_mover,
                       VX_THREAD const float *proximity, P path, VX_THREAD float *teacher, VX_THREAD float *student) {
    // Stereo: a frame arrives `depth_latency` ticks after capture.
    if (e.pending_ticks > 0 && --e.pending_ticks == 0) {
        for (uint i = 0; i < NT_BINS; ++i) {
            e.previous_frame[i] = e.frame[i];
            e.frame[i] = e.pending[i];
        }
        e.frame_age = e.sensor.depth_latency;
        vx_estimate_correct(e.estimate, e.sensor);
        if (nt_goal_visible(e, rays, goal))
            nt_refresh_goal(e);
    } else {
        e.frame_age = e.frame_age < 20 ? e.frame_age + 1 : 20;
    }
    if (e.tick % e.sensor.depth_period == 0) {
        e.pending_seed = vx_seed(e.rng, e.steps);
        nt_pool(rays, e.pending, true, e.pending_seed, e.sensor);
        e.pending_ticks = e.sensor.depth_latency;
#if VX_VIO
        float stereo[VX_RAYS];
        for (uint k = 0; k < VX_RAYS; ++k)
            stereo[k] = vx_stereo_ray(rays[k], k, e.pending_seed, e.sensor);
        const float turn = vx_norm(vx_vec(e.physics.angular_velocity_body[0], e.physics.angular_velocity_body[1],
                                          e.physics.angular_velocity_body[2]));
        const float sigma = vx_vo_sigma(stereo, VX_RAYS, turn, e.sensor, e.rng);
        vx_vo_capture(e.estimate, sigma, e.physics.position, e.sensor, e.rng);
        if (sigma >= 0)
            va_vo_frame(e.attitude, e.physics.orientation_wxyz, e.physics.position);
#endif
    }
    float bins[NT_BINS];
    VxSensorParams none = e.sensor;
    nt_pool(rays, bins, false, 0, none);
    e.min_depth = VX_MAX_RANGE;
    for (uint k = 0; k < VX_RAYS; ++k)
        e.min_depth = fmin(e.min_depth, rays[k]);

    float r[9];
    vx_rotation(e.physics.orientation_wxyz, r);
    const float yaw = nt_yaw(e.physics.orientation_wxyz);
    const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
    const VxVec v = vx_vec(e.physics.linear_velocity[0], e.physics.linear_velocity[1], e.physics.linear_velocity[2]);
    const VxVec to_goal = nt_to_yaw_frame(vx_sub(goal, p), yaw);
    const float distance = vx_norm(to_goal);
    const VxVec gravity = vx_inverse_rotate(r, vx_vec(0, 0, 1));

    for (uint i = 0; i < NT_BINS; ++i)
        teacher[i] = nt_depth_feature(bins[i]);
    teacher[40] = nt_clip(to_goal.x / 4, 2);
    teacher[41] = nt_clip(to_goal.y / 4, 2);
    teacher[42] = nt_clip(to_goal.z / 4, 2);
    teacher[43] = fmin(distance / 4, 3.0f);
    const VxVec vy = nt_to_yaw_frame(v, yaw);
    teacher[44] = vy.x / 2;
    teacher[45] = vy.y / 2;
    teacher[46] = vy.z / 2;
    teacher[47] = gravity.x;
    teacher[48] = gravity.y;
    teacher[49] = gravity.z;
    for (uint i = 0; i < 3; ++i)
        teacher[50 + i] = nt_clip(e.physics.angular_velocity_body[i] / 3, 3);
    for (uint i = 0; i < 4; ++i)
        teacher[53 + i] = e.action[i];
    const float heading = atan2(to_goal.y, to_goal.x);
    teacher[57] = sin(heading);
    teacher[58] = cos(heading);
    teacher[59] = e.settle / 6.0f;
    teacher[60] = float(e.goal_tick) / NT_GOAL_TICKS;
    if (has_mover) {
        const VxVec mp = nt_to_yaw_frame(vx_sub(vx_vec(mover[0], mover[1], mover[2]), p), yaw);
        const VxVec mv = nt_to_yaw_frame(vx_vec(mover[3], mover[4], mover[5]), yaw);
        teacher[61] = nt_clip(mp.x / 4, 2);
        teacher[62] = nt_clip(mp.y / 4, 2);
        teacher[63] = nt_clip(mp.z / 4, 2);
        teacher[64] = mv.x;
        teacher[65] = mv.y;
        teacher[66] = mv.z;
        teacher[67] = 1;
    } else {
        for (uint i = 61; i < 68; ++i)
            teacher[i] = 0;
    }
    const VxVec bias = nt_to_yaw_frame(vx_vec(e.estimate.velocity[0] - v.x, e.estimate.velocity[1] - v.y,
                                              e.estimate.velocity[2] - v.z), yaw);
    teacher[68] = nt_clip(bias.x, 2);
    teacher[69] = nt_clip(bias.y, 2);
    teacher[70] = nt_clip(bias.z, 2);
    teacher[71] = e.goal + 1 >= e.goal_count ? 1 : 0;
    e.clearance = NT_PROXIMITY_RANGE;
    for (uint i = 0; i < NT_PROXIMITY; ++i) {
        teacher[72 + i] = proximity[i] / NT_PROXIMITY_RANGE;
        e.clearance = fmin(e.clearance, proximity[i]);
    }
    // Follow the reference path: move on to the next corner once within NT_CORNER of this one.
    bool moved = false;
    while (e.path_index + 1 < NT_PATH && path[(e.path_index + 1) * 4 + 3] < path[e.path_index * 4 + 3]) {
        const uint k = e.path_index * 4;
        if (vx_norm(vx_sub(p, vx_vec(path[k], path[k + 1], path[k + 2]))) > NT_CORNER)
            break;
        ++e.path_index;
        moved = true;
    }
    for (uint j = 0; j < 4; ++j)
        e.carrot[j] = path[e.path_index * 4 + j];
    const VxVec carrot = vx_vec(e.carrot[0], e.carrot[1], e.carrot[2]);
    const VxVec to_carrot = nt_to_yaw_frame(vx_sub(carrot, p), yaw);
    const float remaining = vx_norm(to_carrot) + e.carrot[3];
    teacher[86] = nt_clip(to_carrot.x / 4, 2);
    teacher[87] = nt_clip(to_carrot.y / 4, 2);
    teacher[88] = nt_clip(to_carrot.z / 4, 2);
    teacher[89] = fmin(remaining / 4, 3.0f);

    // Student: only what the drone has, in its own estimated attitude.
    float re[9];
    vx_rotation(nt_attitude(e), re);
    const float yaw_est = nt_yaw(nt_attitude(e));
    const VxVec gravity_est = vx_inverse_rotate(re, vx_vec(0, 0, 1));
    for (uint i = 0; i < NT_BINS; ++i) {
        student[i] = NT_STUDENT_DEPTH(e.frame[i]);
        student[40 + i] = NT_STUDENT_DEPTH(e.previous_frame[i]);
    }
    student[80] = e.frame_age / 10.0f;
    const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
    const VxVec ge = vx_vec(goal.x + e.goal_offset[0], goal.y + e.goal_offset[1], goal.z + e.goal_offset[2]);
    const VxVec to_goal_est = nt_to_yaw_frame(vx_sub(ge, pe), yaw_est);
    student[81] = nt_clip(to_goal_est.x / 4, 2);
    student[82] = nt_clip(to_goal_est.y / 4, 2);
    student[83] = nt_clip(to_goal_est.z / 4, 2);
    student[84] = fmin(vx_norm(to_goal_est) / 4, 3.0f);
    const VxVec ve = nt_to_yaw_frame(vx_vec(e.estimate.velocity[0], e.estimate.velocity[1], e.estimate.velocity[2]), yaw_est);
    student[85] = ve.x / 2;
    student[86] = ve.y / 2;
    student[87] = ve.z / 2;
    student[88] = gravity_est.x;
    student[89] = gravity_est.y;
    student[90] = gravity_est.z;
    const float gyro_bias[2] = {e.attitude.on ? e.attitude.gyro_bias[0] : 0.0f, e.attitude.on ? e.attitude.gyro_bias[1] : 0.0f};
    student[91] = nt_clip((e.physics.angular_velocity_body[0] + gyro_bias[0] + e.sensor.gyro_noise * vx_normal(e.rng)) / 3, 3);
    student[92] = nt_clip((e.physics.angular_velocity_body[1] + gyro_bias[1] + e.sensor.gyro_noise * vx_normal(e.rng)) / 3, 3);
    // Downward range finder: height above whatever is below, here the floor of the room.
    const float height = e.physics.position[2] + e.sensor.height_noise * vx_normal(e.rng);
    const bool height_valid = height < e.sensor.height_range && vx_uniform(e.rng) > .05f;
    if (height_valid)
        vx_height_correct(e.estimate, height, e.sensor.height_noise);
    vx_estimate_aid(e.estimate, e.physics.linear_velocity, e.physics.position, e.sensor, e.rng);
    student[93] = height_valid ? height / 2 : 0;
    student[94] = height_valid ? 1 : 0;
    for (uint i = 0; i < 4; ++i)
        student[95 + i] = e.action[i];
    student[99] = float(e.goal_tick) / NT_GOAL_TICKS;
    // Route target from the local map (local_map.h) when the host builds one; otherwise the goal.
    const float reach = fmin(1.0f, NT_ROUTE_REACH / fmax(vx_norm(to_goal_est), 1e-3f));
    student[100] = to_goal_est.x * reach / NT_ROUTE_REACH;
    student[101] = to_goal_est.y * reach / NT_ROUTE_REACH;
    student[102] = to_goal_est.z * reach / NT_ROUTE_REACH;
    student[103] = 1;
    for (uint i = 104; i < NT_STUDENT_OBS; ++i)
        student[i] = 0;
    if (e.goal_distance < 0 || moved)
        e.goal_distance = remaining;
}

#ifndef NT_YIELD_TIME
#define NT_YIELD_TIME 1.0f
#endif
#ifndef NT_YIELD_RADIUS
#define NT_YIELD_RADIUS .9f
#endif

// Giving way to a moving box, from a 7-slot mover block (teacher 61..67, student 104..110):
// relative position / 4 and velocity in the yaw frame, then a flag. The fastest fraction of the
// velocity (vx, vy, vz) in the yaw frame that keeps NT_YIELD_RADIUS from the box over the next
// NT_YIELD_TIME seconds, assuming both keep their velocity. A negative fraction backs away.
inline float nt_yield(VX_THREAD const float *mover, float vx, float vy, float vz) {
    if (mover[6] < .5f)
        return 1;
    const float mx = mover[0] * 4, my = mover[1] * 4, mz = mover[2] * 4;
    const float fractions[6] = {1, .7f, .4f, .15f, 0, -.4f};
    float k = 1, best = -1;
    for (uint c = 0; c < 6; ++c) {
        const float ux = mover[3] - vx * fractions[c], uy = mover[4] - vy * fractions[c],
                    uz = mover[5] - vz * fractions[c];
        float closest = 1e9f;
        for (uint t = 0; t <= 10; ++t) {
            const float s = NT_YIELD_TIME * t / 10;
            const float dx = mx + ux * s, dy = my + uy * s, dz = mz + uz * s;
            closest = fmin(closest, sqrt(dx * dx + dy * dy + dz * dz));
        }
        if (closest >= NT_YIELD_RADIUS)
            return fractions[c];
        if (closest > best) {
            best = closest;
            k = fractions[c];
        }
    }
    return k;
}

// The same check applied to a whole action, after the policy's correction: the teacher
// always gives way, whatever the policy asks for.
inline void nt_mover_certify(VX_THREAD const float *mover, VX_THREAD float *action) {
    float a[3];
    for (uint i = 0; i < 3; ++i)
        a[i] = fmin(1.0f, fmax(-1.0f, action[i]));
    const float k = nt_yield(mover, a[0] * 2, a[1] * 2, a[2] * .75f);
    if (k < 1)
        for (uint i = 0; i < 3; ++i)
            action[i] = a[i] * k;
}

inline void nt_teacher_certify(VX_THREAD const float *teacher, VX_THREAD float *action) {
    nt_mover_certify(teacher + 61, action);
}

// The student does the same with the box it tracks in its own stereo map (local_map.h).
inline void nt_student_certify(VX_THREAD const float *student, VX_THREAD float *action) {
    nt_mover_certify(student + 104, action);
}

// A hand-written pull along the reference path, from the teacher's observation before
// normalization. The teacher learns a correction on top of it, which avoids the early phase
// where random actions crash so often that the policy learns to hover in place.
inline void nt_teacher_prior(VX_THREAD const float *teacher, VX_THREAD float *prior) {
    const float gx = teacher[86] * 4, gy = teacher[87] * 4, gz = teacher[88] * 4;
    const float distance = sqrt(gx * gx + gy * gy + gz * gz), remaining = teacher[89] * 4;
    // Speed in m/s along the path, slowed so the climb stays within the climb limit.
    float scale = 2 * fmin(remaining * .5f, 1.0f) / fmax(distance, 1e-3f);
    scale = fmin(scale, .75f / fmax(fabs(gz), 1e-3f));
    scale *= nt_yield(teacher + 61, gx * scale, gy * scale, gz * scale);
    const float brake = remaining < 1 ? .3f * teacher[44] * 2 : 0;
    prior[0] = nt_clip(gx * scale / 2 - brake, 1);
    prior[1] = nt_clip(gy * scale / 2, 1);
    prior[2] = nt_clip(gz * scale / .75f, 1);
    prior[3] = nt_clip(atan2(gy, gx), 1);
}

// A speed limit from the last stereo frame, which the drone has on board. Any depth bin whose
// obstacle lies within NT_SHIELD_TUBE of the commanded direction limits the speed to what
// can still stop before it, allowing for the frame's age. Motion outside the camera's view is
// capped at NT_SHIELD_BLIND.
#define NT_SHIELD_TUBE .32f
#define NT_SHIELD_STOP .3f
#define NT_SHIELD_BRAKE 1.5f
#define NT_SHIELD_BLIND .5f
inline void nt_shield(VX_THREAD const TrainEnv &e, float yaw, VX_THREAD float *command) {
    const float cy = cos(yaw), sy = sin(yaw);
    // Commanded horizontal velocity and climb, in the yaw frame.
    VxVec v = vx_vec(cy * command[0] + sy * command[1], -sy * command[0] + cy * command[1], command[2]);
    const float speed = vx_norm(v);
    if (speed < 1e-3f)
        return;
    const VxVec u = vx_scale(v, 1 / speed);
    const float age = e.frame_age * .05f;
    float limit = speed;
    for (uint b = 0; b < NT_BINS; ++b) {
        const float col = float((b % 10) * 2 + 1), row = float((b / 10) * 4 + 2);
        const VxVec ray = vx_cam_dir(col, row, .75f);
        const float d = e.frame[b], cosine = fmax(-1.0f, fmin(1.0f, ray.x * u.x + ray.y * u.y + ray.z * u.z));
        const float angle = fmax(0.0f, acos(cosine) - .15f);
        if (angle > 1.5f || d * sin(angle) > NT_SHIELD_TUBE)
            continue;
        const float room = d * cos(angle) - NT_SHIELD_STOP - speed * age;
        limit = fmin(limit, room > 0 ? sqrt(2 * NT_SHIELD_BRAKE * room) : 0.0f);
    }
    // The camera sees about 45 degrees to each side; beyond that the drone is blind.
    const float level = sqrt(v.x * v.x + v.y * v.y);
    if (v.x < .7f * level)
        limit = fmin(limit, NT_SHIELD_BLIND);
    // Climbing is never braked, so the drone can still rise over what it sees.
    const float moving = command[2] > 0 ? level : speed;
    if (limit < moving) {
        const float k = limit / moving;
        command[0] *= k;
        command[1] *= k;
        if (command[2] < 0)
            command[2] *= k;
    }
}

// Turns a policy action in [-1, 1] into the controller command: velocity in the yaw frame,
// climb rate and yaw rate.
inline void nt_command(VX_THREAD TrainEnv &e, VX_THREAD const float *action, VX_THREAD const NavTaskConfig &c,
                       VX_THREAD float *command) {
    float a[4];
    float change = 0;
    for (uint i = 0; i < 4; ++i) {
        a[i] = fmin(1.0f, fmax(-1.0f, action[i]));
        change += (a[i] - e.action[i]) * (a[i] - e.action[i]);
        e.action[i] = a[i];
    }
    e.reward = -.02f * change;
    const float yaw = nt_yaw(nt_attitude(e)), cy = cos(yaw), sy = sin(yaw);
    const float vx = a[0] * c.max_speed, vy = a[1] * c.max_speed;
    command[0] = cy * vx - sy * vy;
    command[1] = sy * vx + cy * vy;
    command[2] = a[2] * c.max_climb;
    command[3] = a[3] * c.max_yaw_rate;
    if (c.shield)
        nt_shield(e, yaw, command);
}

// What the flight controller measures this substep: the estimate, not the truth.
inline RLPhysicsState nt_measure(VX_THREAD TrainEnv &e) {
    VX_THREAD VxAttitude &a = e.attitude;
    float gyro[3] = {0, 0, 0};
    float rt[9];
    vx_rotation(e.physics.orientation_wxyz, rt);
    const VxVec bias = vx_inverse_rotate(rt, vx_vec(e.estimate.accel_bias[0], e.estimate.accel_bias[1], e.estimate.accel_bias[2]));
    const VxVec accel = vx_vec((e.physics.linear_velocity[0] - a.last_velocity[0]) / .01f,
                               (e.physics.linear_velocity[1] - a.last_velocity[1]) / .01f,
                               (e.physics.linear_velocity[2] - a.last_velocity[2]) / .01f);
    va_step(a, e.physics.orientation_wxyz, e.physics.angular_velocity_body, e.physics.linear_velocity, bias, e.estimate.velocity, .01f, gyro);
    if (a.on) {
        const VxVec error = va_accel_error(a, e.physics.orientation_wxyz, accel, bias);
        e.estimate.accel_error[0] = error.x;
        e.estimate.accel_error[1] = error.y;
        e.estimate.accel_error[2] = error.z;
    }
    vx_estimate_step(e.estimate, e.physics.linear_velocity, e.sensor, .01f, e.rng);
    RLPhysicsState m = e.physics;
    for (uint i = 0; i < 4; ++i)
        m.orientation_wxyz[i] = a.q[i];
    for (uint i = 0; i < 3; ++i) {
        m.position[i] = e.estimate.position[i];
        m.linear_velocity[i] = e.estimate.velocity[i];
        const float noise = e.sensor.gyro_noise * vx_normal(e.rng);
        m.angular_velocity_body[i] = a.on ? gyro[i] : m.angular_velocity_body[i] + noise;
    }
    return m;
}

// End of a tick. Scores progress, waypoint arrival and contact. Sets e.done to 1 when the
// episode ends (success or contact) and 2 when it runs out of time. Returns true if the
// waypoint changed, so the caller passes the new goal next tick.
inline bool nt_finish_tick(VX_THREAD TrainEnv &e, bool contact, VxVec goal) {
    ++e.tick;
    ++e.goal_tick;
    const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
    const float speed = vx_norm(vx_vec(e.physics.linear_velocity[0], e.physics.linear_velocity[1], e.physics.linear_velocity[2]));
    const float distance = vx_norm(vx_sub(p, goal));
    // Arrival is judged on the truth, or (arrive_estimate) on what the drone believes: its
    // estimated position against the waypoint in its own frame, and its estimated speed.
    const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
    const VxVec ge = vx_vec(goal.x + e.goal_offset[0], goal.y + e.goal_offset[1], goal.z + e.goal_offset[2]);
    const float judged = e.arrive_estimate ? vx_norm(vx_sub(pe, ge)) : distance;
    const float judged_speed = e.arrive_estimate
        ? vx_norm(vx_vec(e.estimate.velocity[0], e.estimate.velocity[1], e.estimate.velocity[2])) : speed;
    // Progress is measured along the reference path.
    const float remaining = vx_norm(vx_sub(p, vx_vec(e.carrot[0], e.carrot[1], e.carrot[2]))) + e.carrot[3];
    float r = e.reward + 2 * (e.goal_distance - remaining) - .01f;
    e.goal_distance = remaining;
    if (distance < .5f)
        r -= .1f * speed;
    r -= .1f * fmax(0.0f, .6f - e.min_depth);
    // Keep a margin around the body, and slow down inside it.
    const float tight = fmax(0.0f, NT_MARGIN - e.clearance);
    r -= tight + speed * tight;
    e.settle = judged < .25f && judged_speed < .2f ? e.settle + 1 : 0;
    bool advanced = false;
    e.done = 0;
    if (contact) {
        r -= NT_CONTACT_PENALTY;
        ++e.contacts;
        e.done = 1;
    } else if (e.settle >= 6) {
        e.arrive_error = distance;
        r += 5;
        ++e.arrivals;
        e.settle = 0;
        e.goal_tick = 0;
        e.goal_distance = -1;
        e.path_index = 0;
        if (++e.goal >= e.goal_count) {
            r += 5;
            ++e.completions;
            e.done = 1;
        } else {
            advanced = true;
            nt_refresh_goal(e);
        }
    } else if (e.goal_tick >= NT_GOAL_TICKS) {
        e.done = 2;
    }
    if (e.done)
        ++e.episodes;
    e.reward = r;
    e.episode_return += r;
    return advanced;
}
