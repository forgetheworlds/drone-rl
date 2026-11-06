#pragma once

// Procedural training worlds in the drone-rl voxel format. They never read drone-rl's maps;
// those stay held out for evaluation.
//
// Two thirds of the worlds are obstacle courses (gen_course). The rest are closed rooms with
// furniture-like boxes, pillars, cross walls with gaps, overhanging slabs, thin bars and
// sometimes moving boxes. Waypoints follow the rule a VLM would give them: each one is in view
// (clear line of sight, within the camera's vertical field) from the pose before it, and
// reachable through free space wide enough for the drone.

#include "missions.hpp"
#include "paths.hpp"
#include <cstdint>
#include <queue>
#include <random>
#include <thread>

struct GenGrid {
    int nx, ny, nz;
    float scale;
    std::vector<uint8_t> solid;
    uint8_t &at(int x, int y, int z) { return solid[(size_t(z) * ny + y) * nx + x]; }
    bool inside(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz; }
    void box(float x0, float y0, float z0, float x1, float y1, float z1, uint8_t v = 1) {
        const int a = std::max(0, int(std::floor(x0 / scale))), b = std::min(nx - 1, int(std::ceil(x1 / scale)) - 1);
        const int c = std::max(0, int(std::floor(y0 / scale))), d = std::min(ny - 1, int(std::ceil(y1 / scale)) - 1);
        const int e = std::max(0, int(std::floor(z0 / scale))), f = std::min(nz - 1, int(std::ceil(z1 / scale)) - 1);
        for (int z = e; z <= f; z++)
            for (int y = c; y <= d; y++)
                for (int x = a; x <= b; x++)
                    at(x, y, z) = v;
    }
};

// Cells whose center is at least `radius` (box distance) from any solid cell or the outside.
inline std::vector<uint8_t> gen_free_space(const GenGrid &g, float radius) {
    const int r = int(std::ceil(radius / g.scale - .5f));
    std::vector<uint8_t> a = g.solid, b(a.size());
    auto pass = [&](int axis) {
        const int n[3] = {g.nx, g.ny, g.nz};
        for (int z = 0; z < g.nz; z++)
            for (int y = 0; y < g.ny; y++)
                for (int x = 0; x < g.nx; x++) {
                    int c[3] = {x, y, z};
                    uint8_t v = 0;
                    for (int k = -r; k <= r && !v; k++) {
                        int q[3] = {x, y, z};
                        q[axis] = c[axis] + k;
                        v = q[axis] < 0 || q[axis] >= n[axis] ? 1 : a[(size_t(q[2]) * g.ny + q[1]) * g.nx + q[0]];
                    }
                    b[(size_t(z) * g.ny + y) * g.nx + x] = v;
                }
        std::swap(a, b);
    };
    pass(0);
    pass(1);
    pass(2);
    for (auto &v : a)
        v = !v;
    return a;
}

// Breadth-first search over free cells (6-connected) from one cell. Returns hop counts, -1 if
// unreachable.
inline std::vector<int> gen_reach(const GenGrid &g, const std::vector<uint8_t> &free, int sx, int sy, int sz) {
    std::vector<int> hops(free.size(), -1);
    auto index = [&](int x, int y, int z) { return (size_t(z) * g.ny + y) * g.nx + x; };
    if (!g.inside(sx, sy, sz) || !free[index(sx, sy, sz)])
        return hops;
    std::queue<int> q;
    hops[index(sx, sy, sz)] = 0;
    q.push(int(index(sx, sy, sz)));
    const int dx[6] = {1, -1, 0, 0, 0, 0}, dy[6] = {0, 0, 1, -1, 0, 0}, dz[6] = {0, 0, 0, 0, 1, -1};
    while (!q.empty()) {
        const int c = q.front();
        q.pop();
        const int x = c % g.nx, y = (c / g.nx) % g.ny, z = c / (g.nx * g.ny);
        for (int k = 0; k < 6; k++) {
            const int a = x + dx[k], b = y + dy[k], d = z + dz[k];
            if (!g.inside(a, b, d))
                continue;
            const size_t n = index(a, b, d);
            if (free[n] && hops[n] < 0) {
                hops[n] = hops[c] + 1;
                q.push(int(n));
            }
        }
    }
    return hops;
}

// Static line of sight between two points, using the same DDA as the depth camera.
inline bool gen_clear(const VxWorld &w, const uint32_t *bits, VxVec a, VxVec b) {
    const VxVec d = vx_sub(b, a);
    const float distance = vx_norm(d);
    return vx_ray_static(w, bits, a, vx_unit(d), VX_MAX_RANGE) + .001f >= distance;
}

struct GenStats {
    uint64_t attempts = 0, rejected = 0;
};

// Builds one mission from a seed. Appends its bits to `set` and returns false if the layout
// left no valid waypoints (the caller retries with another seed).
inline bool gen_mission(uint32_t seed, Mission &m, std::vector<uint32_t> &bits_out) {
    std::mt19937 rng(seed);
    auto u = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto chance = [&](float p) { return u(0, 1) < p; };
    GenGrid g;
    const bool hall = chance(.25f);
    g.scale = hall ? (chance(.5f) ? .25f : .5f) : .125f;
    const float lx = hall ? u(12, 30) : u(8, 14), ly = hall ? u(8, 20) : u(6, 10), lz = hall ? u(3, 8) : u(3, 4.5f);
    g.nx = int(lx / g.scale);
    g.ny = int(ly / g.scale);
    g.nz = int(lz / g.scale);
    g.solid.assign(size_t(g.nx) * g.ny * g.nz, 0);
    for (int z = 0; z < g.nz; z++)
        for (int y = 0; y < g.ny; y++)
            for (int x = 0; x < g.nx; x++)
                if (x == 0 || y == 0 || z == 0 || x == g.nx - 1 || y == g.ny - 1 || z == g.nz - 1)
                    g.at(x, y, z) = 1;
    const float W = g.nx * g.scale, D = g.ny * g.scale, H = g.nz * g.scale;
    const float density = u(0, 1);

    const int boxes = int(density * (hall ? 25 : 10) + u(0, 3));
    for (int i = 0; i < boxes; i++) {
        const float sx = u(.3f, 1.5f), sy = u(.3f, 1.5f), h = u(.3f, std::min(2.2f, H - .5f));
        const float x = u(1.5f, W - sx), y = u(0, D - sy);
        g.box(x, y, 0, x + sx, y + sy, h);
    }
    const int pillars = int(density * (hall ? 12 : 6) * u(0, 1));
    for (int i = 0; i < pillars; i++) {
        const float s = u(.15f, .5f), x = u(1.5f, W - s), y = u(0, D - s);
        g.box(x, y, 0, x + s, y + s, H);
    }
    const int walls = chance(.6f) ? 1 + int(u(0, hall ? 3 : 2)) : 0;
    for (int i = 0; i < walls; i++) {
        const float x = u(2.5f, W - 1.5f), t = u(.15f, .35f);
        g.box(x, 0, 0, x + t, D, H);
        const int gaps = 1 + int(u(0, 2));
        for (int k = 0; k < gaps; k++) {
            const float wy = u(.5f, 1.3f), wz = u(.5f, 1.4f), y = u(.3f, D - wy - .3f), z = u(.3f, std::max(.31f, H - wz - .3f));
            g.box(x - .01f, y, z, x + t + .01f, y + wy, z + wz, 0);
        }
    }
    const int slabs = int(density * 4 * u(0, 1));
    for (int i = 0; i < slabs; i++) {
        const float sx = u(.8f, 3), sy = u(.8f, 3), z = u(.8f, std::max(.9f, H - .8f)), x = u(1.5f, W - sx), y = u(0, D - sy);
        g.box(x, y, z, x + sx, y + sy, z + u(.1f, .3f));
    }
    const int bars = int(density * 5 * u(0, 1));
    for (int i = 0; i < bars; i++) {
        const float z = u(.5f, H - .3f), s = std::max(g.scale, .1f);
        if (chance(.5f)) {
            const float x = u(1.5f, W - .2f);
            g.box(x, 0, z, x + s, D, z + s);
        } else {
            const float y = u(.2f, D - .2f);
            g.box(1.5f, y, z, W, y + s, z + s);
        }
    }

    VxWorld &w = m.world;
    w = VxWorld{};
    w.nx = g.nx;
    w.ny = g.ny;
    w.nz = g.nz;
    w.scale = g.scale;
    std::vector<uint32_t> bits((g.solid.size() + 31) / 32, 0);
    for (size_t i = 0; i < g.solid.size(); i++)
        if (g.solid[i])
            bits[i / 32] |= 1u << (i % 32);

    // Free space for planning keeps 0.3 m between the drone center and any voxel.
    const auto free = gen_free_space(g, .3f);
    auto cell_center = [&](int x, int y, int z) {
        return vx_vec((x + .5f) * g.scale, (y + .5f) * g.scale, (z + .5f) * g.scale);
    };
    auto cell_of = [&](VxVec p, int &x, int &y, int &z) {
        x = int(std::floor(p.x / g.scale));
        y = int(std::floor(p.y / g.scale));
        z = int(std::floor(p.z / g.scale));
        return g.inside(x, y, z) && free[(size_t(z) * g.ny + y) * g.nx + x];
    };
    // Start near the low-x end, at a usable height.
    VxVec start{};
    bool found = false;
    for (int t = 0; t < 200 && !found; t++) {
        start = vx_vec(u(.5f, std::min(2.5f, W * .25f)), u(.5f, D - .5f), u(.7f, std::min(2.2f, H - .5f)));
        int x, y, z;
        found = cell_of(start, x, y, z);
        if (found)
            start = cell_center(x, y, z);
    }
    if (!found)
        return false;
    m.launch_yaw = chance(.5f) ? 0.0f : u(-1.2f, 1.2f);

    const int goal_count = chance(.4f) ? 1 : (chance(.3f) ? 2 : 3);
    VxVec from = start;
    float from_yaw = m.launch_yaw;
    m.goals.clear();
    w.mover_count = 0;
    for (int k = 0; k < goal_count; k++) {
        int fx, fy, fz;
        cell_of(from, fx, fy, fz);
        const auto hops = gen_reach(g, free, fx, fy, fz);
        bool placed = false;
        for (int t = 0; t < 400 && !placed; t++) {
            const float distance = u(1.5f, hall ? 9.0f : 7.5f);
            // The first waypoint must be in the launch view. Later ones only need to be
            // reachable by yawing on the spot, so any heading works.
            const float heading = k == 0 ? from_yaw + u(-.7f, .7f) : u(-3.14159f, 3.14159f);
            const float horizontal = distance * std::cos(std::atan(u(-.5f, .5f)));
            const VxVec p = vx_vec(from.x + horizontal * std::cos(heading), from.y + horizontal * std::sin(heading),
                                   from.z + u(-.5f, .5f) * horizontal);
            int x, y, z;
            if (!cell_of(p, x, y, z) || hops[(size_t(z) * g.ny + y) * g.nx + x] < 0)
                continue;
            const VxVec c = cell_center(x, y, z);
            if (k == 0) {
                float q[4] = {std::cos(from_yaw / 2), 0, 0, std::sin(from_yaw / 2)}, r[9];
                vx_rotation(q, r);
                if (!vx_goal_visible(w, bits.data(), r, from, c, 0))
                    continue;
            } else {
                const VxVec d = vx_sub(c, from);
                const float flat = std::sqrt(d.x * d.x + d.y * d.y);
                if (std::fabs(d.z) > .7f * flat || vx_norm(d) < 1 || vx_norm(d) > 11.5f ||
                    !gen_clear(w, bits.data(), from, c))
                    continue;
            }
            m.goals.push_back({c.x, c.y, c.z});
            from_yaw = std::atan2(c.y - from.y, c.x - from.x);
            from = c;
            placed = true;
        }
        if (!placed)
            break;
    }
    if (m.goals.empty())
        return false;

    // Moving boxes cross the route between waypoints but never sit on a waypoint.
    if (chance(.35f)) {
        const int movers = 1 + int(u(0, 3));
        for (int i = 0; i < movers && w.mover_count < VX_MAX_MOVERS; i++) {
            const auto &a = i == 0 ? std::array<float, 3>{start.x, start.y, start.z} : m.goals[(i - 1) % m.goals.size()];
            const auto &b = m.goals[i % m.goals.size()];
            const float f = u(.35f, .65f);
            VxMover mv{};
            for (int j = 0; j < 3; j++) {
                mv.center[j] = a[j] + f * (b[j] - a[j]);
                mv.extent[j] = u(.15f, .45f);
            }
            const float dx = b[0] - a[0], dy = b[1] - a[1];
            mv.axis = std::fabs(dx) > std::fabs(dy) ? 1 : 0;
            if (chance(.2f))
                mv.axis = 2;
            mv.amplitude = u(.4f, 1.6f);
            mv.period = u(4, 14);
            mv.phase = u(0, 6.28318f);
            bool ok = true;
            for (const auto &q : m.goals) {
                float gap = 0;
                for (int j = 0; j < 3; j++) {
                    const float reach = mv.extent[j] + (int(mv.axis) == j ? mv.amplitude : 0) + .6f;
                    gap = std::max(gap, std::fabs(q[j] - mv.center[j]) - reach);
                }
                ok = ok && gap > 0;
            }
            float sgap = 0;
            const float sp[3] = {start.x, start.y, start.z};
            for (int j = 0; j < 3; j++)
                sgap = std::max(sgap, std::fabs(sp[j] - mv.center[j]) - (mv.extent[j] + (int(mv.axis) == j ? mv.amplitude : 0) + .8f));
            if (ok && sgap > 0)
                w.movers[w.mover_count++] = mv;
        }
    }
    m.start = {start.x, start.y, start.z};
    m.name = "gen-" + std::to_string(seed);
    m.portfolio = "generated";
    w.bits_offset = 0;
    bits_out = std::move(bits);
    return true;
}

inline std::vector<uint32_t> gen_pack(const GenGrid &g) {
    std::vector<uint32_t> bits((g.solid.size() + 31) / 32, 0);
    for (size_t i = 0; i < g.solid.size(); i++)
        if (g.solid[i])
            bits[i / 32] |= 1u << (i % 32);
    return bits;
}

// Obstacle courses: waypoints roughly in a line, with the obstacles placed on or right next to
// each leg, so flying straight at the waypoint is not enough. The obstacle families are thin
// walls with a gap the drone must thread, pillars close to the line, staggered pillars, slabs
// just above or below the line, low furniture and boxes that slide across the line.
inline bool gen_course(uint32_t seed, Mission &m, std::vector<uint32_t> &bits_out) {
    std::mt19937 rng(seed);
    auto u = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto chance = [&](float p) { return u(0, 1) < p; };
    GenGrid g;
    g.scale = .125f;
    g.nx = int(u(10, 14) / g.scale);
    g.ny = int(u(6, 9) / g.scale);
    g.nz = int(u(3.2f, 4.2f) / g.scale);
    g.solid.assign(size_t(g.nx) * g.ny * g.nz, 0);
    for (int z = 0; z < g.nz; z++)
        for (int y = 0; y < g.ny; y++)
            for (int x = 0; x < g.nx; x++)
                if (x == 0 || y == 0 || z == 0 || x == g.nx - 1 || y == g.ny - 1 || z == g.nz - 1)
                    g.at(x, y, z) = 1;
    const float W = g.nx * g.scale, D = g.ny * g.scale, H = g.nz * g.scale;
    auto snap = [&](float v) { return (std::floor(v / g.scale) + .5f) * g.scale; };

    std::vector<VxVec> points;
    points.push_back(vx_vec(snap(u(1, 2)), snap(u(D * .3f, D * .7f)), snap(u(1, 2.2f))));
    const int goal_count = chance(.25f) ? 1 : (chance(.25f) ? 2 : 3);
    for (int k = 0; k < goal_count; k++) {
        const VxVec a = points.back();
        const float length = u(1.5f, 4.0f), turn = chance(.6f) ? 0 : u(-.6f, .6f);
        VxVec b = vx_vec(a.x + length * std::cos(turn), a.y + length * std::sin(turn), a.z + (chance(.7f) ? 0 : u(-.6f, .6f)));
        b = vx_vec(snap(std::fmin(b.x, W - .8f)), snap(std::fmin(std::fmax(b.y, .8f), D - .8f)),
                   snap(std::fmin(std::fmax(b.z, .7f), H - .8f)));
        if (vx_norm(vx_sub(b, a)) < 1.2f)
            return false;
        points.push_back(b);
    }

    VxWorld &w = m.world;
    w = VxWorld{};
    std::vector<VxMover> movers;
    for (size_t k = 0; k + 1 < points.size(); k++) {
        const VxVec a = points[k], b = points[k + 1];
        const float f = u(.35f, .65f);
        const VxVec c = vx_add(a, vx_scale(vx_sub(b, a), f));
        const bool along_x = std::fabs(b.x - a.x) >= std::fabs(b.y - a.y);
        const int kind = int(u(0, 10));
        auto box_lateral = [&](float along0, float along1, float lat0, float lat1, float z0, float z1, uint8_t v = 1) {
            if (along_x)
                g.box(along0, lat0, z0, along1, lat1, z1, v);
            else
                g.box(lat0, along0, z0, lat1, along1, z1, v);
        };
        const float ca = along_x ? c.x : c.y, cl = along_x ? c.y : c.x;
        if (kind == 0 || kind == 1) {
            // Panel across the leg with a gap around the line.
            const float t = u(.15f, .3f), half = u(1, 2.5f), top = chance(.5f) ? H : c.z + u(.6f, 1.2f);
            box_lateral(ca - t / 2, ca + t / 2, cl - half, cl + half, 0, top);
            const float gw = u(.65f, 1.1f), gh = u(.65f, 1.1f), off = u(-.3f, .3f) * gw, offz = u(-.3f, .3f) * gh;
            box_lateral(ca - t, ca + t, cl + off - gw / 2, cl + off + gw / 2, c.z + offz - gh / 2, c.z + offz + gh / 2, 0);
        } else if (kind == 2) {
            // One to three pillars just off the line.
            const int n = 1 + int(u(0, 3));
            for (int i = 0; i < n; i++) {
                const float side = chance(.5f) ? 1 : -1, s = u(.2f, .4f), lat = cl + side * (u(.03f, .45f) + s / 2),
                            al = ca + u(-.6f, .6f);
                box_lateral(al - s / 2, al + s / 2, lat - s / 2, lat + s / 2, 0, H);
            }
        } else if (kind == 3) {
            // Staggered pillars on alternating sides.
            for (int i = 0; i < 3; i++) {
                const float side = i % 2 ? 1 : -1, s = u(.2f, .35f), lat = cl + side * (u(.03f, .3f) + s / 2),
                            al = ca + (i - 1) * u(.6f, 1.0f);
                box_lateral(al - s / 2, al + s / 2, lat - s / 2, lat + s / 2, 0, H);
            }
        } else if (kind == 4) {
            // Slab just above the line.
            const float bottom = c.z + u(.08f, .45f), len = u(.5f, 1.5f), half = u(1, 2.5f);
            box_lateral(ca - len / 2, ca + len / 2, cl - half, cl + half, bottom, bottom + u(.15f, .4f));
        } else if (kind == 5) {
            // Furniture under or beside the line.
            const int n = 1 + int(u(0, 3));
            for (int i = 0; i < n; i++) {
                const float sa = u(.4f, 1.2f), sl = u(.4f, 1.2f), al = ca + u(-.8f, .8f), lat = cl + u(-1, 1);
                const float top = std::fmin(c.z - u(.08f, .5f), 1.8f);
                if (top > .2f)
                    box_lateral(al - sa / 2, al + sa / 2, lat - sl / 2, lat + sl / 2, 0, top);
            }
        } else if (kind == 7 || kind == 8) {
            // A wall with a slot too narrow to fly through, so the drone has to go around or
            // over it. The waypoint stays visible through the slot.
            const float t = u(.15f, .3f), half = u(.8f, 1.8f), top = chance(.6f) ? H : c.z + u(.5f, 1.0f);
            const float slot = g.scale * (chance(.5f) ? 1 : 2);
            box_lateral(ca - t / 2, ca + t / 2, cl - half, cl + half, 0, top);
            box_lateral(ca - t, ca + t, cl - g.scale / 2, cl - g.scale / 2 + slot, 0, top, 0);
            if (kind == 8 && top >= H) {
                const float bottom = c.z + u(.15f, .45f);
                box_lateral(ca + t, ca + t + u(.5f, 1.2f), cl - half, cl + half, bottom, bottom + .25f);
            }
        } else if (kind == 9) {
            // Pillars that cut into the line, alternating sides, as in staggered_vertical.
            for (int i = 0; i < 2 + int(u(0, 2)); i++) {
                const float side = i % 2 ? 1 : -1, s = u(.2f, .3f), lat = cl + side * (s / 2 - u(.02f, .14f)),
                            al = ca + (i - .5f) * u(.4f, .8f);
                box_lateral(al - s / 2, al + s / 2, lat - s / 2, lat + s / 2, 0, H);
            }
        } else {
            // A box sliding across the line.
            VxMover mv{};
            mv.center[0] = c.x;
            mv.center[1] = c.y;
            mv.center[2] = c.z;
            for (int j = 0; j < 3; j++)
                mv.extent[j] = u(.15f, .35f);
            mv.axis = along_x ? 1 : 0;
            mv.amplitude = u(.6f, 1.4f);
            mv.period = u(4, 10);
            mv.phase = u(0, 6.28318f);
            movers.push_back(mv);
        }
    }
    // A little unrelated clutter.
    const int clutter = int(u(0, 4));
    for (int i = 0; i < clutter; i++) {
        const float sx = u(.3f, 1.2f), sy = u(.3f, 1.2f), x = u(2, W - sx), y = u(0, D - sy);
        g.box(x, y, 0, x + sx, y + sy, u(.3f, 1.5f));
    }

    const auto free = gen_free_space(g, .2f);
    auto cell = [&](VxVec p, int &x, int &y, int &z) {
        x = int(std::floor(p.x / g.scale));
        y = int(std::floor(p.y / g.scale));
        z = int(std::floor(p.z / g.scale));
        return g.inside(x, y, z) && free[(size_t(z) * g.ny + y) * g.nx + x];
    };
    w.nx = g.nx;
    w.ny = g.ny;
    w.nz = g.nz;
    w.scale = g.scale;
    std::vector<uint32_t> bits = gen_pack(g);
    m.launch_yaw = chance(.7f) ? 0.0f : u(-.4f, .4f);
    for (size_t k = 0; k + 1 < points.size(); k++) {
        int x, y, z, gx, gy, gz;
        if (!cell(points[k], x, y, z) || !cell(points[k + 1], gx, gy, gz))
            return false;
        const auto hops = gen_reach(g, free, x, y, z);
        if (hops[(size_t(gz) * g.ny + gy) * g.nx + gx] < 0)
            return false;
        if (k == 0) {
            float q[4] = {std::cos(m.launch_yaw / 2), 0, 0, std::sin(m.launch_yaw / 2)}, r[9];
            vx_rotation(q, r);
            if (!vx_goal_visible(w, bits.data(), r, points[0], points[1], 0))
                return false;
        } else if (!gen_clear(w, bits.data(), points[k], points[k + 1])) {
            return false;
        }
    }
    for (const VxMover &mv : movers) {
        bool ok = true;
        for (const VxVec &q : points) {
            const float p[3] = {q.x, q.y, q.z};
            float gap = 0;
            for (int j = 0; j < 3; j++)
                gap = std::max(gap, std::fabs(p[j] - mv.center[j]) -
                                        (mv.extent[j] + (int(mv.axis) == j ? mv.amplitude : 0) + .6f));
            ok = ok && gap > 0;
        }
        if (ok && w.mover_count < VX_MAX_MOVERS)
            w.movers[w.mover_count++] = mv;
    }
    m.start = {points[0].x, points[0].y, points[0].z};
    m.goals.clear();
    for (size_t k = 1; k < points.size(); k++)
        m.goals.push_back({points[k].x, points[k].y, points[k].z});
    m.name = "course-" + std::to_string(seed);
    m.portfolio = "generated";
    bits_out = std::move(bits);
    return true;
}

// Generates `count` missions on all cores. Seeds are base_seed + i, skipping failed layouts.
inline MissionSet gen_missions(uint32_t count, uint32_t base_seed) {
    std::vector<Mission> missions(count);
    std::vector<std::vector<uint32_t>> bits(count);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (uint32_t i = t; i < count; i += threads) {
                uint32_t seed = base_seed + i * 7919u;
                // A third rooms with free-form layouts, two thirds obstacle courses.
                const bool course = i % 3 != 0;
                while (!(course ? gen_course(seed, missions[i], bits[i]) : gen_mission(seed, missions[i], bits[i])))
                    seed += 104729u;
                VxWorld w = missions[i].world;
                w.bits_offset = 0;
                missions[i].paths = path_mission(w, bits[i].data(), missions[i].start, missions[i].goals);
            }
        });
    for (auto &p : pool)
        p.join();
    MissionSet set;
    for (uint32_t i = 0; i < count; i++) {
        missions[i].world.bits_offset = uint32_t(set.bits.size());
        set.bits.insert(set.bits.end(), bits[i].begin(), bits[i].end());
        set.missions.push_back(missions[i]);
    }
    return set;
}

// Writes a generated mission as .task and .3dmap files, for viewing or for drone-rl's planner.
inline void gen_write(const Mission &m, const std::vector<uint32_t> &bits, const std::string &stem) {
    std::ofstream map(stem + ".3dmap");
    map << "voxel " << m.world.nx << " " << m.world.ny << " " << m.world.nz << "\n";
    for (uint32_t z = 0; z < m.world.nz; z++)
        for (uint32_t y = 0; y < m.world.ny; y++)
            for (uint32_t x = 0; x < m.world.nx; x++)
                if (vx_solid(m.world, bits.data(), int(x), int(y), int(z)))
                    map << x << " " << y << " " << z << "\n";
    std::ofstream task(stem + ".task");
    const auto slash = stem.find_last_of('/');
    task << (slash == std::string::npos ? stem : stem.substr(slash + 1)) << ".3dmap " << m.world.scale << "\n"
         << m.start[0] << " " << m.start[1] << " " << m.start[2] << "\n" << m.goals.size() << "\n";
    for (const auto &g : m.goals)
        task << g[0] << " " << g[1] << " " << g[2] << "\n";
    task << m.world.mover_count << "\n";
    for (uint32_t i = 0; i < m.world.mover_count; i++) {
        const VxMover &v = m.world.movers[i];
        task << v.center[0] << " " << v.center[1] << " " << v.center[2] << " " << v.extent[0] << " " << v.extent[1]
             << " " << v.extent[2] << " " << v.axis << " " << v.amplitude << " " << v.period << " " << v.phase << "\n";
    }
    task << "yaw " << m.launch_yaw << "\n";
    if (m.has_blocker) {
        task << "blocker";
        for (float v : m.blocker)
            task << " " << v;
        task << "\n";
    }
    for (const auto &h : m.hidden)
        task << "hidden " << h[0] << " " << h[1] << " " << h[2] << " " << h[3] << " " << h[4] << " " << h[5] << "\n";
}
