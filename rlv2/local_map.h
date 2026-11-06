#pragma once

// A small rolling occupancy grid built only from stereo frames and the drifting state estimate,
// plus a route search that turns it into a nearby target the student can actually see.
// Everything lives in the estimated frame, so drift moves the map with the drone.

#include "nav_task.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#define LM_NX 40
#define LM_NY 40
#define LM_NZ 16
#define LM_CELLS (LM_NX * LM_NY * LM_NZ)
#define LM_RES .2f
#define LM_HIT_RANGE 3.5f   // stereo noise grows with z^2, so far hits are not trusted
#define LM_FREE_RANGE 4.0f
#define LM_OCCUPIED 2
#define LM_INFLATE_XY 2     // cells; 0.4 m around every obstacle
#define LM_INFLATE_Z 1
#define LM_CARROT 2.0f
#ifndef LM_WAS_FREE
#define LM_WAS_FREE -3        // odds at or below this: seen free three times
#endif
#ifndef LM_MOVER_HITS
#define LM_MOVER_HITS 5
#endif
#define LM_FLOOR .3f          // hits lower than this are the floor
#define LM_MOVER_MEMORY .6f   // s
#ifndef LM_SWEPT_MEMORY
#define LM_SWEPT_MEMORY 20.0f // s
#endif
#ifndef LM_SWEPT_Z
#define LM_SWEPT_Z 16         // cells above and below
#endif
#define LM_SWEPT_SPEED .2f    // m/s
#ifndef LM_MOVER_SURE
#define LM_MOVER_SURE 12
#endif
#ifndef LM_SWEPT_AHEAD
#define LM_SWEPT_AHEAD 1.0f   // s
#endif
#ifndef LM_MAX_AGE
#define LM_MAX_AGE 255
#endif

struct LocalMap {
    int8_t odds[LM_CELLS];
    uint8_t age[LM_CELLS];   // frames since the last hit
    uint8_t blocked[LM_CELLS];
    uint16_t field[LM_CELLS];
    uint16_t queue[LM_CELLS];
    int origin[3];
    bool ready;
    // A moving box: depth landing in space seen free several times before.
    VxVec mover, mover_velocity;
    float mover_time;
    bool mover_valid;
    bool mover_moving;   // seen in two frames in a row, or clearly in one
    // Where a moving box has been seen, and when: the route keeps clear of its whole path.
    float swept[LM_CELLS];
    float now;
};

inline int lm_index(const LocalMap &m, VxVec p) {
    const int x = int(std::floor(p.x / LM_RES)) - m.origin[0], y = int(std::floor(p.y / LM_RES)) - m.origin[1],
              z = int(std::floor(p.z / LM_RES)) - m.origin[2];
    if (x < 0 || y < 0 || z < 0 || x >= LM_NX || y >= LM_NY || z >= LM_NZ)
        return -1;
    return (z * LM_NY + y) * LM_NX + x;
}

inline VxVec lm_center(const LocalMap &m, int i) {
    const int x = i % LM_NX, y = (i / LM_NX) % LM_NY, z = i / (LM_NX * LM_NY);
    return vx_vec((x + m.origin[0] + .5f) * LM_RES, (y + m.origin[1] + .5f) * LM_RES, (z + m.origin[2] + .5f) * LM_RES);
}

inline void lm_reset(LocalMap &m) {
    std::memset(m.odds, 0, sizeof(m.odds));
    std::memset(m.age, 255, sizeof(m.age));
    m.ready = false;
    m.mover_valid = false;
    m.mover_moving = false;
    m.mover_time = -1e9f;
    std::fill(m.swept, m.swept + LM_CELLS, -1e9f);
    m.now = 0;
}

// Keep the drone near the middle of the grid; cells that scroll in start unknown.
inline void lm_recenter(LocalMap &m, VxVec p) {
    const int want[3] = {int(std::floor(p.x / LM_RES)) - LM_NX / 2, int(std::floor(p.y / LM_RES)) - LM_NY / 2,
                         int(std::floor(p.z / LM_RES)) - LM_NZ / 2};
    if (!m.ready) {
        std::copy(want, want + 3, m.origin);
        m.ready = true;
        return;
    }
    const int d[3] = {want[0] - m.origin[0], want[1] - m.origin[1], want[2] - m.origin[2]};
    if (std::abs(d[0]) < 4 && std::abs(d[1]) < 4 && std::abs(d[2]) < 3)
        return;
    static thread_local int8_t old[LM_CELLS];
    static thread_local uint8_t old_age[LM_CELLS];
    static thread_local float old_swept[LM_CELLS];
    std::memcpy(old, m.odds, sizeof(old));
    std::memcpy(old_age, m.age, sizeof(old_age));
    std::memcpy(old_swept, m.swept, sizeof(old_swept));
    for (int z = 0; z < LM_NZ; z++)
        for (int y = 0; y < LM_NY; y++)
            for (int x = 0; x < LM_NX; x++) {
                const int ox = x + d[0], oy = y + d[1], oz = z + d[2];
                const bool in = ox >= 0 && oy >= 0 && oz >= 0 && ox < LM_NX && oy < LM_NY && oz < LM_NZ;
                const int j = (oz * LM_NY + oy) * LM_NX + ox, i = (z * LM_NY + y) * LM_NX + x;
                m.odds[i] = in ? old[j] : 0;
                m.age[i] = in ? old_age[j] : 255;
                m.swept[i] = in ? old_swept[j] : -1e9f;
            }
    std::copy(want, want + 3, m.origin);
}

// depth: 20 x 16 stereo ranges, VX_MAX_RANGE where stereo gave nothing (unknown, not free).
// position: estimated position at capture; rotation: attitude at capture.
inline void lm_observe(LocalMap &m, const float *depth, VxVec position, const float *rotation, float time) {
    lm_recenter(m, position);
    m.now = time;
    for (int i = 0; i < LM_CELLS; i++)
        if (m.age[i] < 255)
            m.age[i]++;
    float clean[320];
    for (int k = 0; k < 320; k++) {
        // Median of the valid 3x3 neighbours drops lone false returns and most of the noise.
        const int c = k % 20, r = k / 20;
        float v[9];
        int n = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                const int rr = r + dr, cc = c + dc;
                if (rr < 0 || cc < 0 || rr > 15 || cc > 19)
                    continue;
                const float d = depth[rr * 20 + cc];
#ifdef VX_FAR_FREE
                if (d >= VX_FAR_FREE)
                    continue;
#endif
                if (d < VX_MAX_RANGE)
                    v[n++] = d;
            }
#ifdef VX_FAR_FREE
        // Matched past the range cap (depth_preset.h): free up to the cap, kept as it is.
        if (depth[k] >= VX_FAR_FREE && depth[k] < VX_MAX_RANGE) {
            clean[k] = VX_FAR_FREE;
            continue;
        }
#endif
        if (depth[k] >= VX_MAX_RANGE || n < 4) {
            clean[k] = VX_MAX_RANGE;
            continue;
        }
        std::nth_element(v, v + n / 2, v + n);
        clean[k] = v[n / 2];
    }
    VxVec moved_hits[320];
    int moved_count = 0;
    for (int k = 0; k < 320; k++) {
        const float d = clean[k];
        if (d >= VX_MAX_RANGE)
            continue;
        const VxVec ray = vx_camera_ray(uint(k), .75f);
        const VxVec w = vx_rotate(rotation, ray);
        float free_to = std::min(d - .25f, LM_FREE_RANGE);
#ifdef VX_FAR_FREE
        if (d >= VX_FAR_FREE)
            free_to = LM_FAR_FREE_TO;
#endif
        int last = -1;
        for (float s = .1f; s < free_to; s += LM_RES * .5f) {
            const int i = lm_index(m, vx_add(position, vx_scale(w, s)));
            if (i < 0)
                break;
            if (i != last && m.odds[i] > -4)
                m.odds[i]--;
            last = i;
        }
        if (d < LM_HIT_RANGE) {
            const int i = lm_index(m, vx_add(position, vx_scale(w, d + .02f)));
            if (i >= 0) {
                const VxVec hit = vx_add(position, vx_scale(w, d));
                if (m.odds[i] <= LM_WAS_FREE && hit.z > LM_FLOOR && moved_count < 320)
                    moved_hits[moved_count++] = hit;
                m.odds[i] = int8_t(std::min(6, m.odds[i] + 3));
                m.age[i] = 0;
            }
        }
    }
    // Keep the cluster around the closest such hit.
    int closest = -1;
    for (int k = 0; k < moved_count; k++)
        if (closest < 0 || vx_norm(vx_sub(moved_hits[k], position)) < vx_norm(vx_sub(moved_hits[closest], position)))
            closest = k;
    VxVec moved = vx_vec(0, 0, 0);
    int cluster = 0;
    for (int k = 0; k < moved_count; k++)
        if (vx_norm(vx_sub(moved_hits[k], moved_hits[closest])) < .5f) {
            moved = vx_add(moved, moved_hits[k]);
            ++cluster;
        }
    if (cluster >= LM_MOVER_HITS) {
        const VxVec c = vx_scale(moved, 1.0f / cluster);
        const float dt = time - m.mover_time;
        const bool tracked = m.mover_valid && dt > 1e-3f && dt < .5f && vx_norm(vx_sub(c, m.mover)) < 1.0f;
        if (tracked) {
            const VxVec v = vx_scale(vx_sub(c, m.mover), 1 / dt);
            m.mover_velocity = vx_add(vx_scale(m.mover_velocity, .5f), vx_scale(v, .5f));
        } else {
            m.mover_velocity = vx_vec(0, 0, 0);
        }
        // Only a box seen moving over two frames marks its path, with extra height so the
        // route does not hop over it.
        // It may go on along its line, or turn back, for LM_SWEPT_AHEAD seconds either way.
        const float speed = vx_norm(m.mover_velocity);
        m.mover_moving = tracked || cluster >= LM_MOVER_SURE;
        if (tracked && speed > LM_SWEPT_SPEED) {
            const VxVec along = vx_scale(m.mover_velocity, 1 / speed);
            const float reach = speed * LM_SWEPT_AHEAD;
            for (int k = 0; k < moved_count; k++)
                if (vx_norm(vx_sub(moved_hits[k], moved_hits[closest])) < .5f)
                    for (float a = -reach; a <= reach; a += LM_RES * .5f)
                        for (int dz = -LM_SWEPT_Z; dz <= LM_SWEPT_Z; dz++) {
                            const VxVec q = vx_add(moved_hits[k], vx_add(vx_scale(along, a), vx_vec(0, 0, dz * LM_RES)));
                            const int i = lm_index(m, q);
                            if (i >= 0)
                                m.swept[i] = time;
                        }
        }
        m.mover = c;
        m.mover_time = time;
        m.mover_valid = true;
    } else if (time - m.mover_time > LM_MOVER_MEMORY) {
        m.mover_valid = false;
    }
}

inline bool lm_segment_clear(const LocalMap &m, VxVec a, VxVec b) {
    const VxVec d = vx_sub(b, a);
    const float length = vx_norm(d);
    for (float s = 0; s <= length; s += LM_RES * .5f) {
        const int i = lm_index(m, vx_add(a, vx_scale(d, s / std::max(length, 1e-6f))));
        if (i < 0)
            return true;
        if (m.blocked[i])
            return false;
    }
    return true;
}

// Writes a target in the world (estimated) frame. Returns false when no route exists.
// Unknown space counts as free, so the drone keeps exploring toward the goal; the shield and
// the policy handle what the frame shows right now.
// RLV2_NARROW=1: when no route exists with the full 0.4 m margin (a window or doorway closes
// up once both sides are inflated), search again with a 0.2 m margin before giving up.
static const bool g_lm_narrow = std::getenv("RLV2_NARROW") && std::string(std::getenv("RLV2_NARROW")) == "1";

inline bool lm_route(LocalMap &m, VxVec position, VxVec goal, VxVec &target, float &route_length, bool swept = true,
                     int inflate_xy = LM_INFLATE_XY) {
    target = goal;
    route_length = vx_norm(vx_sub(goal, position));
    if (!m.ready)
        return true;
    // Inflate obstacles with a separable box dilation.
    static thread_local uint8_t a[LM_CELLS], b[LM_CELLS];
    for (int i = 0; i < LM_CELLS; i++)
        a[i] = (m.odds[i] >= LM_OCCUPIED && m.age[i] < LM_MAX_AGE) || (swept && m.now - m.swept[i] < LM_SWEPT_MEMORY);
    auto dilate = [](const uint8_t *in, uint8_t *out, int stride, int extent, int radius, int axis_size) {
        for (int i = 0; i < LM_CELLS; i++) {
            const int c = (i / stride) % axis_size;
            uint8_t v = 0;
            for (int o = -radius; o <= radius && !v; o++)
                if (c + o >= 0 && c + o < axis_size)
                    v = in[i + o * stride];
            out[i] = v;
        }
        (void)extent;
    };
    dilate(a, b, 1, 0, inflate_xy, LM_NX);
    dilate(b, a, LM_NX, 0, inflate_xy, LM_NY);
    dilate(a, m.blocked, LM_NX * LM_NY, 0, LM_INFLATE_Z, LM_NZ);
    // The floor and ceiling of the grid are walls for the search.
    int start = lm_index(m, position);
    if (start < 0)
        return false;
    m.blocked[start] = 0;
    // Goal: the cell itself, or the nearest grid cell toward it when it is outside.
    VxVec g = goal;
    g.x = std::clamp(g.x, (m.origin[0] + 1.5f) * LM_RES, (m.origin[0] + LM_NX - 1.5f) * LM_RES);
    g.y = std::clamp(g.y, (m.origin[1] + 1.5f) * LM_RES, (m.origin[1] + LM_NY - 1.5f) * LM_RES);
    g.z = std::clamp(g.z, (m.origin[2] + 1.5f) * LM_RES, (m.origin[2] + LM_NZ - 1.5f) * LM_RES);
    const int goal_cell = lm_index(m, g);
    std::fill(m.field, m.field + LM_CELLS, uint16_t(0xffff));
    int head = 0, tail = 0;
    if (goal_cell >= 0 && !m.blocked[goal_cell]) {
        m.field[goal_cell] = 0;
        m.queue[tail++] = uint16_t(goal_cell);
    } else {
        // Goal cell is inside an obstacle margin: seed from the free cells nearest to it.
        for (int i = 0; i < LM_CELLS; i++)
            if (!m.blocked[i] && vx_norm(vx_sub(lm_center(m, i), g)) < .6f) {
                m.field[i] = 0;
                m.queue[tail++] = uint16_t(i);
            }
    }
    while (head < tail) {
        const int i = m.queue[head++];
        const int x = i % LM_NX, y = (i / LM_NX) % LM_NY, z = i / (LM_NX * LM_NY);
        const int nb[6] = {x > 0 ? i - 1 : -1, x < LM_NX - 1 ? i + 1 : -1, y > 0 ? i - LM_NX : -1,
                           y < LM_NY - 1 ? i + LM_NX : -1, z > 0 ? i - LM_NX * LM_NY : -1,
                           z < LM_NZ - 1 ? i + LM_NX * LM_NY : -1};
        for (int n : nb)
            if (n >= 0 && !m.blocked[n] && m.field[n] == 0xffff) {
                m.field[n] = uint16_t(m.field[i] + 1);
                m.queue[tail++] = uint16_t(n);
            }
    }
    if (m.field[start] == 0xffff) {
        // Boxed in by margins: head for the nearest cell that does reach the goal.
        int best = -1;
        float closest = 1.2f;
        for (int i = 0; i < LM_CELLS; i++)
            if (m.field[i] != 0xffff) {
                const float d = vx_norm(vx_sub(lm_center(m, i), position));
                if (d < closest) {
                    closest = d;
                    best = i;
                }
            }
        if (best < 0)
            return swept ? lm_route(m, position, goal, target, route_length, false, inflate_xy)
                 : g_lm_narrow && inflate_xy > 1 ? lm_route(m, position, goal, target, route_length, false, 1) : false;
        target = lm_center(m, best);
        route_length = closest + m.field[best] * LM_RES;
        return true;
    }
    // Walk down the distance field; aim at the farthest cell still in straight sight.
    int cell = start;
    target = lm_center(m, start);
    for (int step = 0; step < 40 && m.field[cell] > 0; step++) {
        const int x = cell % LM_NX, y = (cell / LM_NX) % LM_NY, z = cell / (LM_NX * LM_NY);
        int best = cell;
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    const int xx = x + dx, yy = y + dy, zz = z + dz;
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= LM_NX || yy >= LM_NY || zz >= LM_NZ)
                        continue;
                    const int n = (zz * LM_NY + yy) * LM_NX + xx;
                    if (m.field[n] < m.field[best])
                        best = n;
                }
        if (best == cell)
            break;
        cell = best;
        const VxVec c = lm_center(m, cell);
        // The first steps are taken even without a clear line: they lead out of an obstacle margin.
        if (step >= 2 && (vx_norm(vx_sub(c, position)) > LM_CARROT || !lm_segment_clear(m, position, c)))
            break;
        target = c;
    }
    if (m.field[cell] == 0 && lm_segment_clear(m, position, goal) && vx_norm(vx_sub(goal, position)) < LM_CARROT)
        target = goal;
    route_length = m.field[start] * LM_RES + vx_norm(vx_sub(goal, g));
    return true;
}

#ifndef LM_PUSH
#define LM_PUSH .5f
#endif
#ifndef LM_PUSH_RANGE
#define LM_PUSH_RANGE .55f
#endif
#ifndef LM_PUSH_AGE
#define LM_PUSH_AGE 12
#endif

// Away from recently seen obstacles closer than LM_PUSH_RANGE, up to LM_PUSH metres.
inline VxVec lm_push(const LocalMap &m, VxVec p) {
    VxVec push = vx_vec(0, 0, 0);
    if (!m.ready)
        return push;
    const int x0 = int(std::floor(p.x / LM_RES)) - m.origin[0], y0 = int(std::floor(p.y / LM_RES)) - m.origin[1],
              z0 = int(std::floor(p.z / LM_RES)) - m.origin[2];
    const int r = int(std::ceil(LM_PUSH_RANGE / LM_RES));
    float strongest = 0;
    for (int z = z0 - r; z <= z0 + r; z++)
        for (int y = y0 - r; y <= y0 + r; y++)
            for (int x = x0 - r; x <= x0 + r; x++) {
                if (x < 0 || y < 0 || z < 0 || x >= LM_NX || y >= LM_NY || z >= LM_NZ)
                    continue;
                const int i = (z * LM_NY + y) * LM_NX + x;
                if (m.odds[i] < LM_OCCUPIED || m.age[i] > LM_PUSH_AGE)
                    continue;
                const VxVec d = vx_sub(p, lm_center(m, i));
                const float n = vx_norm(d);
                if (n >= LM_PUSH_RANGE || n < 1e-3f)
                    continue;
                const float w = (LM_PUSH_RANGE - n) / LM_PUSH_RANGE;
                push = vx_add(push, vx_scale(d, w / n));
                strongest = std::max(strongest, w);
            }
    const float n = vx_norm(push);
    return n > 1e-3f ? vx_scale(push, LM_PUSH * strongest / n) : push;
}

#ifndef LM_NEAR_GOAL
#define LM_NEAR_GOAL .3f
#endif
#ifndef LM_NEAR_TICKS
#define LM_NEAR_TICKS 20
#endif

// Per-drone state for the student's route target: the map plus the stereo frame in flight.
struct StudentRoute {
    LocalMap map;
    float pending[320];
    float rotation[9];
    VxVec position;
    float capture_time;
    int wait;
    int near_goal;
};

inline void sr_reset(StudentRoute &r) {
    lm_reset(r.map);
    r.wait = -1;
    r.near_goal = 0;
}

// Call right after nt_observe with the same true rays. Mirrors the stereo pipeline (same noise,
// same delay), updates the map when a frame lands and writes student[100..103].
inline void sr_update(StudentRoute &r, const TrainEnv &e, const float *rays, VxVec goal, float *student) {
    if (e.tick == 0)
        sr_reset(r);
    const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
    const float time = e.tick * .05f;
    if (r.wait > 0 && --r.wait == 0)
        lm_observe(r.map, r.pending, r.position, r.rotation, r.capture_time);
    if (e.tick % e.sensor.depth_period == 0) {
        for (uint k = 0; k < 320; k++)
            r.pending[k] = NT_STEREO_RAY(rays, k, e.pending_seed, e.sensor);
        r.position = pe;
        r.capture_time = time;
        vx_rotation(nt_attitude(e), r.rotation);
        r.wait = int(e.sensor.depth_latency);
    }
    const VxVec ge = vx_vec(goal.x + e.goal_offset[0], goal.y + e.goal_offset[1], goal.z + e.goal_offset[2]);
    VxVec target;
    float length;
    bool found = lm_route(r.map, pe, ge, target, length);
    // Hovering on the estimated goal without arriving means the estimate drifted: turn in
    // place until the waypoint comes into view again and refreshes.
    r.near_goal = vx_norm(vx_sub(ge, pe)) < LM_NEAR_GOAL ? r.near_goal + 1 : 0;
    if (r.near_goal > LM_NEAR_TICKS)
        found = false;
    // Without a route the drone turns in place and only keeps off nearby walls.
    const VxVec push = LM_PUSH > 0 ? lm_push(r.map, pe) : vx_vec(0, 0, 0);
    target = found ? vx_add(target, push) : vx_add(pe, push);
    const VxVec t = nt_to_yaw_frame(vx_sub(target, pe), nt_yaw(nt_attitude(e)));
    const float reach = std::min(1.0f, NT_ROUTE_REACH / std::max(vx_norm(t), 1e-3f));
    student[100] = t.x * reach / NT_ROUTE_REACH;
    student[101] = t.y * reach / NT_ROUTE_REACH;
    student[102] = t.z * reach / NT_ROUTE_REACH;
    student[103] = found ? 1 : 0;
    // The tracked box, carried forward to now at its estimated velocity.
    const float yaw = nt_yaw(nt_attitude(e));
    const LocalMap &m = r.map;
    const bool mover = m.mover_valid && m.mover_moving && time - m.mover_time < LM_MOVER_MEMORY;
    const VxVec now = vx_add(m.mover, vx_scale(m.mover_velocity, time - m.mover_time)), velocity = m.mover_velocity;
    const VxVec rel = nt_to_yaw_frame(vx_sub(now, pe), yaw), vel = nt_to_yaw_frame(velocity, yaw);
    student[104] = mover ? nt_clip(rel.x / 4, 2) : 0;
    student[105] = mover ? nt_clip(rel.y / 4, 2) : 0;
    student[106] = mover ? nt_clip(rel.z / 4, 2) : 0;
    student[107] = mover ? vel.x : 0;
    student[108] = mover ? vel.y : 0;
    student[109] = mover ? vel.z : 0;
    student[110] = mover ? 1 : 0;
}

