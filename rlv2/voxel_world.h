#pragma once

// Voxel world shared by the Metal kernels and the C++ host.
// The ray, contact, camera and visibility rules follow drone-rl exactly
// (include/navigation/voxel_world.hpp, visible_world.hpp and src/benchmark.cpp).

#ifdef __METAL_VERSION__
#define VX_DEV device
#define VX_THREAD thread
#else
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstdint>
using uint = uint32_t;
#define VX_DEV
#define VX_THREAD
using std::atan2;
using std::ceil;
using std::cos;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::sin;
using std::sqrt;
#endif

#define VX_MAX_MOVERS 16
#define VX_RAYS 320
#define VX_RAY_COLS 20
#define VX_RAY_ROWS 16
#define VX_MAX_RANGE 12.0f
#define VX_BODY_RADIUS 0.18f

struct VxVec {
    float x, y, z;
};

// A box with half-size `extent` that slides along one world axis:
// center + amplitude * cos(2 pi t / period + phase).
struct VxMover {
    float center[3], extent[3];
    uint axis;
    float amplitude, period, phase;
};

// Dense occupancy grid stored as a bitset. bits_offset counts 32-bit words into the shared
// bit buffer, so many worlds of different sizes can live in one GPU buffer.
struct VxWorld {
    uint nx, ny, nz, bits_offset;
    float scale;
    uint mover_count;
    VxMover movers[VX_MAX_MOVERS];
};

inline VxVec vx_vec(float x, float y, float z) {
    VxVec v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
inline VxVec vx_add(VxVec a, VxVec b) { return vx_vec(a.x + b.x, a.y + b.y, a.z + b.z); }
inline VxVec vx_sub(VxVec a, VxVec b) { return vx_vec(a.x - b.x, a.y - b.y, a.z - b.z); }
inline VxVec vx_scale(VxVec a, float s) { return vx_vec(a.x * s, a.y * s, a.z * s); }
inline float vx_norm(VxVec p) { return sqrt(p.x * p.x + p.y * p.y + p.z * p.z); }
inline VxVec vx_unit(VxVec p) { return vx_scale(p, 1 / fmax(vx_norm(p), 1e-6f)); }

// Row-major body-to-world rotation from a w, x, y, z quaternion.
inline void vx_rotation(VX_THREAD const float *q, VX_THREAD float *r) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1 - 2 * y * y - 2 * z * z;
    r[1] = 2 * x * y - 2 * w * z;
    r[2] = 2 * x * z + 2 * w * y;
    r[3] = 2 * x * y + 2 * w * z;
    r[4] = 1 - 2 * x * x - 2 * z * z;
    r[5] = 2 * y * z - 2 * w * x;
    r[6] = 2 * x * z - 2 * w * y;
    r[7] = 2 * y * z + 2 * w * x;
    r[8] = 1 - 2 * x * x - 2 * y * y;
}
inline VxVec vx_rotate(VX_THREAD const float *r, VxVec v) {
    return vx_vec(r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z,
                  r[6] * v.x + r[7] * v.y + r[8] * v.z);
}
inline VxVec vx_inverse_rotate(VX_THREAD const float *r, VxVec v) {
    return vx_vec(r[0] * v.x + r[3] * v.y + r[6] * v.z, r[1] * v.x + r[4] * v.y + r[7] * v.z,
                  r[2] * v.x + r[5] * v.y + r[8] * v.z);
}

// Body-frame direction of depth ray k: 20 columns by 16 rows, 90 degree horizontal field of
// view, vertical half-tangent ray_slope.
// Camera model. Default: the 90 x 74 deg pinhole the policies were first trained with, rays on
// a tangent grid. RLV2_HFOV / RLV2_VFOV (degrees, CPU builds only) switch to a wide lens with an
// equiangular (fisheye-like) grid; VFOV defaults to 0.75 x HFOV (4:3). Same 20 x 16 rays either
// way, so wider means coarser per ray.
#ifndef __METAL_VERSION__
struct VxCam {
    bool wide;
    float half_h, half_v;  // rad
};
inline VxCam vx_cam_from_env() {
    const char *h = std::getenv("RLV2_HFOV"), *v = std::getenv("RLV2_VFOV");
    if (!h)
        return VxCam{false, .785398f, .643501f};
    const float hd = std::stof(h), vd = v ? std::stof(v) : .75f * hd;
    return VxCam{true, hd * .00872665f, vd * .00872665f};
}
inline const VxCam g_vx_cam = vx_cam_from_env();
#endif

// Direction in the body frame for fractional column cf (0..20) and row rf (0..16).
inline VxVec vx_cam_dir(float cf, float rf, float ray_slope) {
#ifndef __METAL_VERSION__
    if (g_vx_cam.wide) {
        const float az = (1 - cf / 10) * g_vx_cam.half_h, el = (1 - rf / 8) * g_vx_cam.half_v;
        return vx_vec(cos(el) * cos(az), cos(el) * sin(az), sin(el));
    }
#endif
    return vx_unit(vx_vec(1, 1 - cf / 10, ray_slope * (1 - rf / 8)));
}

// Body-frame point to ray-grid coordinates (column u, row v, centred on ray indices). False
// when outside the field of view.
inline VxVec vx_camera_ray(uint k, float ray_slope) {
    return vx_cam_dir(float(k % VX_RAY_COLS) + .5f, float(k / VX_RAY_COLS) + .5f, ray_slope);
}

// The point's column and row on the ray grid (ray k sits at (k % 20, k / 20)); false outside the
// field of view.
inline bool vx_cam_uv(VxVec b, float ray_slope, VX_THREAD float &u, VX_THREAD float &v) {
#ifndef __METAL_VERSION__
    if (g_vx_cam.wide) {
        const float az = atan2(b.y, b.x), el = atan2(b.z, sqrt(b.x * b.x + b.y * b.y));
        u = (1 - az / g_vx_cam.half_h) * 10 - .5f;
        v = (1 - el / g_vx_cam.half_v) * 8 - .5f;
        return fabs(az) <= g_vx_cam.half_h && fabs(el) <= g_vx_cam.half_v;
    }
#endif
    if (b.x <= 0)
        return false;
    u = (1 - b.y / b.x) * 10 - .5f;
    v = (1 - b.z / (ray_slope * b.x)) * 8 - .5f;
    return fabs(b.y) <= b.x && fabs(b.z) <= ray_slope * b.x;
}

inline bool vx_solid(VX_DEV const VxWorld &w, VX_DEV const uint *bits, int x, int y, int z) {
    if (x < 0 || x >= int(w.nx) || y < 0 || y >= int(w.ny) || z < 0 || z >= int(w.nz))
        return true;
    const uint index = (uint(z) * w.ny + uint(y)) * w.nx + uint(x);
    return (bits[w.bits_offset + (index >> 5)] >> (index & 31u)) & 1u;
}

// Amanatides-Woo traversal through the static grid. Returns the distance at which the ray
// enters the first solid cell, or maximum.
inline float vx_ray_static(VX_DEV const VxWorld &w, VX_DEV const uint *bits, VxVec p, VxVec d,
                           float maximum) {
    const float s = w.scale;
    int cell[3] = {int(floor(p.x / s)), int(floor(p.y / s)), int(floor(p.z / s))};
    float point[3] = {p.x, p.y, p.z}, dir[3] = {d.x, d.y, d.z}, tmax[3], delta[3];
    int step[3];
    for (int a = 0; a < 3; a++) {
        step[a] = dir[a] >= 0 ? 1 : -1;
        tmax[a] = fabs(dir[a]) < 1e-8f
                      ? 1e30f
                      : ((float(cell[a] + (step[a] > 0 ? 1 : 0))) * s - point[a]) / dir[a];
        delta[a] = fabs(dir[a]) < 1e-8f ? 1e30f : s / fabs(dir[a]);
    }
    float distance = 0;
    while (distance < maximum) {
        if (vx_solid(w, bits, cell[0], cell[1], cell[2]))
            return distance;
        int a = tmax[0] < tmax[1] ? (tmax[0] < tmax[2] ? 0 : 2) : (tmax[1] < tmax[2] ? 1 : 2);
        distance = tmax[a];
        tmax[a] += delta[a];
        cell[a] += step[a];
    }
    return maximum;
}

inline VxVec vx_mover_center(VX_DEV const VxMover &m, float time) {
    VxVec c = vx_vec(m.center[0], m.center[1], m.center[2]);
    const float offset = m.amplitude * cos(6.28318530718f * time / m.period + m.phase);
    if (m.axis == 0)
        c.x += offset;
    else if (m.axis == 1)
        c.y += offset;
    else
        c.z += offset;
    return c;
}

inline float vx_ray(VX_DEV const VxWorld &w, VX_DEV const uint *bits, VxVec p, VxVec d,
                    float time) {
    float nearest = vx_ray_static(w, bits, p, d, VX_MAX_RANGE);
    for (uint i = 0; i < w.mover_count; i++) {
        VX_DEV const VxMover &m = w.movers[i];
        const VxVec c = vx_mover_center(m, time);
        float low[3] = {c.x - m.extent[0], c.y - m.extent[1], c.z - m.extent[2]};
        float high[3] = {c.x + m.extent[0], c.y + m.extent[1], c.z + m.extent[2]};
        float point[3] = {p.x, p.y, p.z}, direction[3] = {d.x, d.y, d.z};
        float entry = 0, exit = nearest;
        for (int a = 0; a < 3; a++) {
            if (fabs(direction[a]) < 1e-8f) {
                if (point[a] < low[a] || point[a] > high[a])
                    exit = -1;
            } else {
                float t0 = (low[a] - point[a]) / direction[a],
                      t1 = (high[a] - point[a]) / direction[a];
                entry = fmax(entry, fmin(t0, t1));
                exit = fmin(exit, fmax(t0, t1));
            }
        }
        if (exit >= entry)
            nearest = entry;
    }
    return nearest;
}

// Sphere-versus-world contact for the airframe at point p.
inline bool vx_contact(VX_DEV const VxWorld &w, VX_DEV const uint *bits, VxVec p, float time) {
    const float s = w.scale, radius = VX_BODY_RADIUS;
    const int x = int(floor(p.x / s)), y = int(floor(p.y / s)), z = int(floor(p.z / s));
    const int n = int(ceil(radius / s));
    for (int dz = -n; dz <= n; dz++)
        for (int dy = -n; dy <= n; dy++)
            for (int dx = -n; dx <= n; dx++) {
                if (!vx_solid(w, bits, x + dx, y + dy, z + dz))
                    continue;
                float lo[3] = {float(x + dx) * s, float(y + dy) * s, float(z + dz) * s};
                float hi[3] = {float(x + dx + 1) * s, float(y + dy + 1) * s, float(z + dz + 1) * s};
                float point[3] = {p.x, p.y, p.z}, d2 = 0;
                for (int a = 0; a < 3; a++) {
                    float d = point[a] - fmin(fmax(point[a], lo[a]), hi[a]);
                    d2 += d * d;
                }
                if (d2 <= radius * radius)
                    return true;
            }
    for (uint i = 0; i < w.mover_count; i++) {
        VX_DEV const VxMover &m = w.movers[i];
        const VxVec c = vx_mover_center(m, time);
        float dx = fmax(0.f, fabs(p.x - c.x) - m.extent[0]), dy = fmax(0.f, fabs(p.y - c.y) - m.extent[1]),
              dz = fmax(0.f, fabs(p.z - c.z) - m.extent[2]);
        if (dx * dx + dy * dy + dz * dz <= radius * radius)
            return true;
    }
    return false;
}

// A waypoint may be issued only when it is inside the camera frustum, between 0.25 m and
// 11.9 m away, and the straight ray to it is not blocked.
inline bool vx_goal_visible(VX_DEV const VxWorld &w, VX_DEV const uint *bits,
                            VX_THREAD const float *rotation, VxVec position, VxVec goal,
                            float time) {
    const VxVec d = vx_sub(goal, position);
    const float distance = vx_norm(d);
    const VxVec body = vx_inverse_rotate(rotation, d);
    float u, v;
    const bool frustum = vx_cam_uv(body, .75f, u, v);
    const bool range = distance > .25f && distance < 11.9f;
    const float ray_range = vx_ray(w, bits, position, vx_unit(d), time);
    return frustum && range && ray_range + .001f >= distance;
}

