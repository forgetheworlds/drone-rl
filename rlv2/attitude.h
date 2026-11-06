#pragma once

// Attitude from the IMU alone, as on the ESP32-S3 kit: gyro plus accelerometer through the
// complementary (Mahony) filter of the Crazyflie / ESP-Drone firmware (sensfusion6, Kp 0.4,
// Ki 0.001). There is no magnetometer, so yaw drifts with the gyro bias unless stereo
// odometry corrects it. Off by default; the grader turns it on (RLV2_ATT_EST).
//
// The accelerometer reads specific force in the body frame: the true acceleration plus
// gravity, plus a bias and vibration. On a quadrotor that points along the thrust while it
// accelerates, so the filter is pulled toward level then, as on the real drone.

#include "voxel_world.h"

#define VA_TWO_KP .8f
#define VA_TWO_KI .002f
#define VA_GYRO_WALK .0003f   // gyro bias walk, rad/s per sqrt(s)
#define VA_VO_GAIN .1f        // share of the innovation taken from each good frame
#define VA_VO_GATE .35f       // rad; larger innovations are ignored
#define VA_VO_TURN_DRIFT .01f // odometry rotation drift per radian turned
#define VA_VO_MOVE_DRIFT .003f// odometry rotation drift per sqrt(metre) travelled, rad

struct VxAttitude {
    float q[4];                 // estimated body-to-world rotation, wxyz
    float integral[3];          // filter's integral term (its gyro bias estimate), rad/s
    float gyro_bias[3];         // true gyro bias left after the start-up calibration, rad/s
    float last_velocity[3];
    float vo_drift[3], vo_yaw, vo_last_truth[3];
    float last_estimate[3], accel_estimate[3];  // for compensating the accelerometer
    float gyro_noise, accel_noise, vo_pixel;
    float two_kp;               // proportional gain, VA_TWO_KP unless the caller changes it
    uint on, vo_yaw_on, compensate, started, rng;
};

inline float va_yaw(VX_THREAD const float *q) {
    return atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}
inline float va_wrap(float a) {
    while (a > 3.14159265f)
        a -= 6.28318531f;
    while (a < -3.14159265f)
        a += 6.28318531f;
    return a;
}

// Copies the true attitude; the estimator stays off until va_begin.
inline void va_reset(VX_THREAD VxAttitude &a, VX_THREAD const float *truth) {
    for (uint i = 0; i < 4; ++i)
        a.q[i] = truth[i];
    a.on = a.vo_yaw_on = a.started = 0;
}

inline float va_uniform(VX_THREAD uint &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return float(s >> 8) * (1.0f / 16777216.0f);
}
inline float va_normal(VX_THREAD uint &s) {
    return (va_uniform(s) + va_uniform(s) + va_uniform(s) + va_uniform(s) - 2.0f) * 1.7320508f;
}

// Turns the estimator on. `seed` gives its own random stream, so the other sensor draws stay
// the same as with it off. Residual gyro bias 0.1-0.5 deg/s per axis at level 1.
inline void va_begin(VX_THREAD VxAttitude &a, uint seed, float level, float gyro_noise, float accel_noise,
                     float vo_pixel, uint vo_mode, VX_THREAD const float *position,
                     VX_THREAD const float *velocity) {
    a.rng = seed ? seed : 1u;
    a.on = 1;
    a.vo_yaw_on = vo_mode;
    a.compensate = 0;
    a.two_kp = VA_TWO_KP;
    a.started = 0;
    a.gyro_noise = gyro_noise;
    a.accel_noise = accel_noise;
    a.vo_pixel = vo_pixel;
    for (uint i = 0; i < 3; ++i) {
        const float sign = va_uniform(a.rng) < .5f ? -1.0f : 1.0f;
        a.gyro_bias[i] = sign * level * (.1f + .4f * va_uniform(a.rng)) * .0174533f;
        a.integral[i] = 0;
        a.last_velocity[i] = velocity[i];
        a.vo_last_truth[i] = position[i];
        a.last_estimate[i] = a.accel_estimate[i] = 0;
    }
    a.vo_drift[0] = a.vo_drift[1] = a.vo_drift[2] = 0;
    a.vo_yaw = va_yaw(a.q);
}

// One 10 ms IMU sample. `truth` is the true attitude, `rate` the true body rate, `bias` the
// accelerometer bias in the body frame (the same one the velocity filter has, sensors.h).
// Returns the gyro reading (with bias and noise) in `gyro`, which the rate controller gets.
//
// compensate: the filter compares the accelerometer with gravity plus the acceleration the
// velocity filter sees (`estimate`, its velocity), not with gravity alone, so a quadrotor
// accelerating is not taken for a tilt. Fixed-wing autopilots do the same with GPS.
inline void va_step(VX_THREAD VxAttitude &a, VX_THREAD const float *truth, VX_THREAD const float *rate,
                    VX_THREAD const float *velocity, VxVec bias, VX_THREAD const float *estimate, float dt,
                    VX_THREAD float *gyro) {
    if (!a.on) {
        for (uint i = 0; i < 4; ++i)
            a.q[i] = truth[i];
        return;
    }
    float r[9];
    vx_rotation(truth, r);
    VxVec accel = vx_vec((velocity[0] - a.last_velocity[0]) / dt, (velocity[1] - a.last_velocity[1]) / dt,
                         (velocity[2] - a.last_velocity[2]) / dt + 9.81f);
    if (!a.started)
        accel = vx_vec(0, 0, 9.81f);
    a.started = a.started ? a.started : 1;
    for (uint i = 0; i < 3; ++i)
        a.last_velocity[i] = velocity[i];
    const VxVec f = vx_inverse_rotate(r, accel);
    const float fm[3] = {f.x + bias.x + a.accel_noise * va_normal(a.rng), f.y + bias.y + a.accel_noise * va_normal(a.rng),
                         f.z + bias.z + a.accel_noise * va_normal(a.rng)};
    const float walk = VA_GYRO_WALK * sqrt(dt);
    float g[3];
    for (uint i = 0; i < 3; ++i) {
        a.gyro_bias[i] += walk * va_normal(a.rng);
        g[i] = rate[i] + a.gyro_bias[i] + a.gyro_noise * va_normal(a.rng);
        gyro[i] = g[i];
    }
    VxVec up = vx_vec(0, 0, 1);
    if (a.compensate) {
        for (uint i = 0; i < 3; ++i) {
            const float raw = a.started > 1 ? (estimate[i] - a.last_estimate[i]) / dt : 0.0f;
            a.accel_estimate[i] += (raw - a.accel_estimate[i]) * dt / .1f;
            a.last_estimate[i] = estimate[i];
        }
        up = vx_unit(vx_vec(a.accel_estimate[0], a.accel_estimate[1], a.accel_estimate[2] + 9.81f));
    }
    a.started = 2;
    VX_THREAD float *q = a.q;
    float re[9];
    vx_rotation(q, re);
    // Where the estimate expects the accelerometer to point, halved as in sensfusion6.
    const VxVec h2 = vx_scale(vx_inverse_rotate(re, up), .5f);
    const float norm = sqrt(fm[0] * fm[0] + fm[1] * fm[1] + fm[2] * fm[2]);
    if (norm > 1e-3f) {
        const float ax = fm[0] / norm, ay = fm[1] / norm, az = fm[2] / norm;
        const float hx = h2.x, hy = h2.y, hz = h2.z;
        const float ex = ay * hz - az * hy, ey = az * hx - ax * hz, ez = ax * hy - ay * hx;
        a.integral[0] += VA_TWO_KI * ex * dt;
        a.integral[1] += VA_TWO_KI * ey * dt;
        a.integral[2] += VA_TWO_KI * ez * dt;
        g[0] += a.integral[0] + a.two_kp * ex;
        g[1] += a.integral[1] + a.two_kp * ey;
        g[2] += a.integral[2] + a.two_kp * ez;
    }
    const float h = .5f * dt;
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    q[0] += h * (-x * g[0] - y * g[1] - z * g[2]);
    q[1] += h * (w * g[0] + y * g[2] - z * g[1]);
    q[2] += h * (w * g[1] - x * g[2] + z * g[0]);
    q[3] += h * (w * g[2] + x * g[1] - y * g[0]);
    const float n = 1 / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (uint i = 0; i < 4; ++i)
        q[i] *= n;
}

// The extra world-frame acceleration error the velocity filter sees because it rotates the
// accelerometer reading (specific force plus body bias) with the estimated attitude instead of
// the true one. A tilt that a horizontal bias causes cancels that bias, as on a real drone.
inline VxVec va_accel_error(VX_THREAD const VxAttitude &a, VX_THREAD const float *truth, VxVec accel_world, VxVec bias) {
    float rt[9], re[9];
    vx_rotation(truth, rt);
    vx_rotation(a.q, re);
    const VxVec body = vx_add(vx_inverse_rotate(rt, vx_vec(accel_world.x, accel_world.y, accel_world.z + 9.81f)), bias);
    return vx_sub(vx_rotate(re, body), vx_rotate(rt, body));
}

// A good stereo frame. Odometry gives the rotation since its keyframes, with a slow drift of
// its own; pull the estimate toward it. vo_yaw_on 1 takes only its yaw, 2 the whole attitude
// (what a visual-inertial filter that also estimates tilt would give the flight controller).
inline void va_vo_frame(VX_THREAD VxAttitude &a, VX_THREAD const float *truth, VX_THREAD const float *position) {
    if (!a.on || !a.vo_yaw_on)
        return;
    const float yaw = va_yaw(truth);
    float travel = 0;
    for (uint i = 0; i < 3; ++i) {
        travel += (position[i] - a.vo_last_truth[i]) * (position[i] - a.vo_last_truth[i]);
        a.vo_last_truth[i] = position[i];
    }
    const float turned = fabs(va_wrap(yaw - a.vo_yaw));
    a.vo_yaw = yaw;
    const float step = VA_VO_TURN_DRIFT * turned + VA_VO_MOVE_DRIFT * sqrt(travel);
    float error[3];
    for (uint i = 0; i < 3; ++i) {
        a.vo_drift[i] += step * va_normal(a.rng);
        error[i] = a.vo_drift[i] + .005f * a.vo_pixel * va_normal(a.rng);
    }
    // Rotation from the estimate to the odometry attitude, as a small world-frame vector:
    // truth carries the odometry error `error`, so q_vo = exp(error) * truth.
    VX_THREAD const float *e = a.q;
    // d = truth * conj(estimate)
    float d[4];
    const float c[4] = {e[0], -e[1], -e[2], -e[3]};
    d[0] = truth[0] * c[0] - truth[1] * c[1] - truth[2] * c[2] - truth[3] * c[3];
    d[1] = truth[0] * c[1] + truth[1] * c[0] + truth[2] * c[3] - truth[3] * c[2];
    d[2] = truth[0] * c[2] - truth[1] * c[3] + truth[2] * c[0] + truth[3] * c[1];
    d[3] = truth[0] * c[3] + truth[1] * c[2] - truth[2] * c[1] + truth[3] * c[0];
    const float sign = d[0] < 0 ? -2.0f : 2.0f;
    float phi[3] = {sign * d[1] + error[0], sign * d[2] + error[1], sign * d[3] + error[2]};
    if (a.vo_yaw_on == 1)
        phi[0] = phi[1] = 0;
    if (sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]) > VA_VO_GATE)
        return;
    // estimate = exp(gain * phi) * estimate
    const float h[4] = {1, .5f * VA_VO_GAIN * phi[0], .5f * VA_VO_GAIN * phi[1], .5f * VA_VO_GAIN * phi[2]};
    const float w = e[0], x = e[1], y = e[2], z = e[3];
    VX_THREAD float *q = a.q;
    q[0] = h[0] * w - h[1] * x - h[2] * y - h[3] * z;
    q[1] = h[0] * x + h[1] * w + h[2] * z - h[3] * y;
    q[2] = h[0] * y - h[1] * z + h[2] * w + h[3] * x;
    q[3] = h[0] * z + h[1] * y - h[2] * x + h[3] * w;
    const float n = 1 / sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (uint i = 0; i < 4; ++i)
        q[i] *= n;
}

// Tilt error (angle between the true and estimated body z axes) and yaw error, rad.
inline void va_error(VX_THREAD const VxAttitude &a, VX_THREAD const float *truth, VX_THREAD float &tilt,
                     VX_THREAD float &yaw) {
    float rt[9], re[9];
    vx_rotation(truth, rt);
    vx_rotation(a.q, re);
    const VxVec zt = vx_rotate(rt, vx_vec(0, 0, 1)), ze = vx_rotate(re, vx_vec(0, 0, 1));
    tilt = acos(fmin(1.0f, zt.x * ze.x + zt.y * ze.y + zt.z * ze.z));
    yaw = va_wrap(va_yaw(a.q) - va_yaw(truth));
}
