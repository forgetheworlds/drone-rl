#pragma once

// 3D training and test missions: waypoints at other heights and waypoints behind openings,
// plus the "blocker" class (low drone, waypoint straight ahead at the same height, a kitchen
// counter or a standing or slowly walking person in the way, clutter hidden behind it).
//
// Kinds (Mission::portfolio):
//   counter  start 0.3-0.6 m; a 0.85-1.0 m counter (island, or wall to wall) 1.4-2.4 m ahead,
//            1-3 boxes hidden behind it, waypoint 1.2-2.5 m past it at start height.
//   person   start 0.3-0.6 m; a 1.6-1.8 m person-sized box 1.5-2.8 m ahead, half of them
//            walking slowly across the line (0.3-0.8 m sway, 10-20 s period, <= 0.5 m/s),
//            sometimes two side by side, clutter hidden behind; waypoint past at start height.
//   window   wall across the room with a 0.6-1.0 m window at 0.6-1.5 m; waypoint behind it.
//   door     wall with a 0.8-1.0 m doorway well off the line; waypoint behind it.
//   under    a table (underside 0.75-0.85 m) ahead; waypoint under its middle at 0.44 m.
//   overwall full-width 0.7-1.1 m wall; drone starts at 1.0-1.6 m, waypoint low behind it.
//   level    up a level (platform 1.0-1.5 m high, waypoint above it) or down low (start
//            1.8+ m, waypoint at 0.45-0.6 m), sometimes a second waypoint further on.
//
// Waypoints follow the VLM rule (in the camera frustum with a clear line of sight from the
// launch pose), with one stated relaxation for counter/person: a point 0.25-1.2 m straight above
// the waypoint may be the visible one instead (the VLM sees the doorway or shelf behind the
// counter and marks the spot at the drone's height below it), seen from the start or from 0.5 m
// above it (take-off). Behind a counter a waypoint at the drone's own low height cannot be in
// view from that height at all.
//
// Hidden clutter goes into Mission::hidden so the teacher's reference path is planned without
// it (mission_plan_bits): the teacher must discover it like the student does.

#include "worldgen.hpp"
#include <atomic>

inline const char *g3d_kinds[] = {"counter", "person", "window", "door", "under", "overwall", "level"};
constexpr int G3D_KINDS = 7;

struct G3dStats {
    std::atomic<uint64_t> tries{0}, frustum{0}, occluded{0}, placed{0};
};
inline G3dStats g3d_stats;

inline bool gen_mission3d(uint32_t seed, int kind, Mission &m, std::vector<uint32_t> &bits_out) {
    std::mt19937 rng(seed);
    auto u = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto chance = [&](float p) { return u(0, 1) < p; };
    GenGrid g;
    g.scale = .125f;
    const float lx = u(9, 12), ly = u(5, 8), lz = u(2.6f, 3.2f);
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
    // Furniture along the side walls, off the main line.
    for (int i = 0, n = int(u(1, 5)); i < n; i++) {
        const float sx = u(.4f, 1.2f), sy = u(.3f, .8f), h = u(.4f, 2.0f), x = u(1.8f, W - sx - .2f);
        const float y = chance(.5f) ? u(.1f, .4f) : D - sy - u(.1f, .4f);
        g.box(x, y, 0, x + sx, y + sy, h);
    }

    const bool low = kind == 0 || kind == 1;
    const float sx = u(1.0f, 1.5f), sy = D / 2 + u(-.6f, .6f);
    float sz;
    switch (kind) {
    case 0: case 1: sz = u(.3f, .6f); break;
    case 4: sz = u(.8f, 1.4f); break;
    case 5: sz = u(1.0f, 1.6f); break;
    default: sz = u(.4f, 1.8f);
    }
    m.has_blocker = false;
    m.hidden.clear();
    VxWorld &w = m.world;
    w = VxWorld{};
    w.mover_count = 0;
    auto add_hidden = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
        g.box(x0, y0, z0, x1, y1, z1);
        m.hidden.push_back({x0, y0, z0, x1, y1, z1});
    };
    // Goal region: x range, lateral spread around sy, z range.
    float gx0 = 0, gx1 = 0, gy = .4f, gz0 = sz - .1f, gz1 = sz + .1f;
    bool level_two = false;
    if (kind == 0) {
        const float x0 = sx + u(1.4f, 2.4f), x1 = x0 + u(.55f, .9f), h = u(.85f, 1.0f);
        float y0 = 0, y1 = D;
        if (chance(.5f)) {
            const float half = u(.6f, 1.2f), c = sy + u(-.4f, .4f);
            y0 = c - half;
            y1 = c + half;
        }
        g.box(x0, y0, 0, x1, y1, h);
        if (chance(.4f))  // wall cabinets above a wall-to-wall counter leave a band to fly through
            if (y0 == 0) g.box(x0, 0, h + u(.5f, .7f) + .3f, x1 - .2f, D, H);
        m.has_blocker = true;
        m.blocker = {x0, y0, 0, x1, y1, h};
        for (int i = 0, n = 1 + int(u(0, 3)); i < n; i++) {
            const float bx = x1 + u(.2f, 1.4f), by = sy + u(-1.2f, 1.0f), s = u(.3f, .7f);
            add_hidden(bx, by, 0, bx + s, by + u(.3f, .7f), u(.3f, .8f));
        }
        gx0 = x1 + 1.2f;
        gx1 = x1 + 2.5f;
    } else if (kind == 1) {
        const float px = sx + u(1.5f, 2.8f), py = sy + u(-.15f, .15f), h = u(1.6f, 1.8f);
        const int people = chance(.25f) ? 2 : 1;
        float ylo = 1e9f, yhi = -1e9f;
        for (int k = 0; k < people; k++) {
            const float cy = py + (people == 2 ? (k ? .35f : -.35f) : 0);
            if (chance(.5f) && w.mover_count < VX_MAX_MOVERS) {
                VxMover mv{};
                mv.center[0] = px + .15f;
                mv.center[1] = cy;
                mv.center[2] = h / 2 + .01f;
                mv.extent[0] = .15f;
                mv.extent[1] = .22f;
                mv.extent[2] = h / 2;
                mv.axis = 1;
                mv.amplitude = u(.3f, .8f);
                mv.period = u(10, 20);
                mv.phase = u(0, 6.28318f);
                w.movers[w.mover_count++] = mv;
                ylo = std::min(ylo, cy - .22f - mv.amplitude);
                yhi = std::max(yhi, cy + .22f + mv.amplitude);
            } else {
                g.box(px, cy - .22f, 0, px + .3f, cy + .22f, h);
                ylo = std::min(ylo, cy - .22f);
                yhi = std::max(yhi, cy + .22f);
            }
        }
        m.has_blocker = true;
        m.blocker = {px, ylo, 0, px + .3f, yhi, h};
        for (int i = 0, n = int(u(0, 3)); i < n; i++) {
            const float bx = px + u(.6f, 1.5f), by = sy + u(-1.2f, 1.0f), s = u(.3f, .6f);
            add_hidden(bx, by, 0, bx + s, by + u(.3f, .6f), u(.3f, 1.0f));
        }
        gx0 = px + 1.4f;
        gx1 = px + 2.8f;
    } else if (kind == 2 || kind == 3) {
        const float x = sx + u(2.0f, 3.0f), t = u(.15f, .25f);
        g.box(x, 0, 0, x + t, D, H);
        if (kind == 2) {
            const float ww = u(.7f, 1.0f), wh = u(.65f, .9f), wz = u(.6f, 1.5f), wy = sy + u(-1.2f, 1.2f) - ww / 2;
            g.box(x - .01f, wy, wz, x + t + .01f, wy + ww, wz + wh, 0);
            gy = .5f;
            gz0 = wz + .2f;
            gz1 = wz + wh - .2f;
        } else {
            const float dw = u(.85f, 1.0f), side = chance(.5f) ? 1.f : -1.f;
            const float c = std::clamp(sy + side * u(.8f, 1.8f), dw / 2 + .2f, D - dw / 2 - .2f);
            g.box(x - .01f, c - dw / 2, 0, x + t + .01f, c + dw / 2, u(1.9f, 2.1f), 0);
            gz0 = .6f;
            gz1 = 1.6f;
            gy = 1.0f;
        }
        gx0 = x + t + 1.0f;
        gx1 = x + t + 2.5f;
    } else if (kind == 4) {
        const float x0 = sx + u(1.6f, 2.6f), dx = u(1.2f, 1.8f), dy = u(.8f, 1.2f), y0 = sy + u(-.2f, .2f) - dy / 2;
        const float under = u(.75f, .85f), leg = .06f;
        g.box(x0, y0, under, x0 + dx, y0 + dy, under + .05f);
        for (float lx2 : {x0, x0 + dx - leg})
            for (float ly2 : {y0, y0 + dy - leg})
                g.box(lx2, ly2, 0, lx2 + leg, ly2 + leg, under);
        gx0 = x0 + dx / 2 - .1f;
        gx1 = x0 + dx / 2 + .1f;
        gy = .1f;
        gz0 = .43f;
        gz1 = .45f;
    } else if (kind == 5) {
        const float x = sx + u(1.6f, 2.6f), h = u(.7f, 1.1f);
        g.box(x, 0, 0, x + .2f, D, h);
        gx0 = x + 1.0f;
        gx1 = x + 2.0f;
        gz0 = .4f;
        gz1 = .7f;
    } else {
        if (chance(.5f)) {  // up a level
            sz = u(.4f, .8f);
            const float x = sx + u(2.0f, 3.0f), p = u(1.0f, 1.5f);
            g.box(x, 0, 0, W, D, p);
            gx0 = x + .8f;
            gx1 = x + 2.0f;
            gz0 = p + .4f;
            gz1 = p + .7f;
        } else {  // down low
            sz = u(1.8f, std::min(2.3f, H - .5f));
            gx0 = sx + 2.5f;
            gx1 = sx + 4.0f;
            gz0 = .45f;
            gz1 = .6f;
            // Something to duck under on the way down sometimes.
            if (chance(.5f)) {
                const float x = sx + u(1.0f, 1.8f);
                g.box(x, 0, u(1.1f, 1.3f), x + u(.4f, .8f), D, H);
            }
        }
        gy = .6f;
        level_two = chance(.4f);
    }

    w.nx = g.nx;
    w.ny = g.ny;
    w.nz = g.nz;
    w.scale = g.scale;
    std::vector<uint32_t> bits = gen_pack(g);
    const auto free = gen_free_space(g, .3f);
    auto cell_of = [&](VxVec p, int &x, int &y, int &z) {
        x = int(std::floor(p.x / g.scale));
        y = int(std::floor(p.y / g.scale));
        z = int(std::floor(p.z / g.scale));
        return g.inside(x, y, z) && free[(size_t(z) * g.ny + y) * g.nx + x];
    };
    auto center = [&](int x, int y, int z) { return vx_vec((x + .5f) * g.scale, (y + .5f) * g.scale, (z + .5f) * g.scale); };
    int x, y, z;
    VxVec start = vx_vec(sx, sy, sz);
    if (!cell_of(start, x, y, z))
        return false;
    start = center(x, y, z);
    m.launch_yaw = u(-.15f, .15f);
    const auto hops = gen_reach(g, free, x, y, z);
    float q[4] = {std::cos(m.launch_yaw / 2), 0, 0, std::sin(m.launch_yaw / 2)}, r[9];
    vx_rotation(q, r);
    m.goals.clear();
    bool placed = false;
    for (int t = 0; t < 300 && !placed; t++) {
        const VxVec p = vx_vec(u(gx0, gx1), sy + u(-gy, gy), u(gz0, gz1));
        if (!cell_of(p, x, y, z) || hops[(size_t(z) * g.ny + y) * g.nx + x] < 0)
            continue;
        const VxVec c = center(x, y, z);
        g3d_stats.tries++;
        bool seen = vx_goal_visible(w, bits.data(), r, start, c, 0);
        // Blockers: the VLM may mark a spot at the drone's height below something it can see
        // (the doorway or shelf behind the counter), from the start or 0.5 m higher.
        for (float up = .25f; !seen && low && up <= 1.2f; up += .25f)
            for (float lift : {0.0f, .5f})
                seen = seen || vx_goal_visible(w, bits.data(), r, vx_vec(start.x, start.y, start.z + lift),
                                               vx_vec(c.x, c.y, c.z + up), 0);
        if (!seen) {
            const VxVec b = vx_inverse_rotate(r, vx_sub(c, start));
            (b.x > 0 && std::fabs(b.y) <= b.x && std::fabs(b.z) > .75f * b.x ? g3d_stats.frustum : g3d_stats.occluded)++;
            continue;
        }
        g3d_stats.placed++;
        m.goals.push_back({c.x, c.y, c.z});
        placed = true;
    }
    if (!placed)
        return false;
    if (level_two) {
        const VxVec from = vx_vec(m.goals[0][0], m.goals[0][1], m.goals[0][2]);
        for (int t = 0; t < 200; t++) {
            const VxVec p = vx_vec(from.x + u(1.5f, 3.5f), from.y + u(-1.5f, 1.5f), from.z + u(-.8f, .8f));
            if (!cell_of(p, x, y, z) || hops[(size_t(z) * g.ny + y) * g.nx + x] < 0)
                continue;
            const VxVec c = center(x, y, z), d = vx_sub(c, from);
            if (std::fabs(d.z) > .7f * std::hypot(d.x, d.y) || !gen_clear(w, bits.data(), from, c))
                continue;
            m.goals.push_back({c.x, c.y, c.z});
            break;
        }
    }
    // The walking person must not stand on the start or a waypoint.
    for (uint32_t i = 0; i < w.mover_count; i++) {
        const VxMover &mv = w.movers[i];
        for (const auto &p : std::vector<std::array<float, 3>>{{start.x, start.y, start.z}, m.goals[0]}) {
            float gap = 0;
            for (int j = 0; j < 3; j++)
                gap = std::max(gap, std::fabs(p[j] - mv.center[j]) - (mv.extent[j] + (int(mv.axis) == j ? mv.amplitude : 0) + .5f));
            if (gap <= 0)
                return false;
        }
    }
    m.start = {start.x, start.y, start.z};
    m.portfolio = g3d_kinds[kind];
    m.name = std::string("g3d-") + g3d_kinds[kind] + "-" + std::to_string(seed);
    w.bits_offset = 0;
    bits_out = std::move(bits);
    return true;
}

// Fills one mission of the given kind, retrying seeds, with the teacher path planned on the
// map without hidden clutter.
inline void gen3d_fill(uint32_t seed, int kind, Mission &m, std::vector<uint32_t> &bits) {
    while (!gen_mission3d(seed, kind, m, bits))
        seed += 104729u;
    VxWorld w = m.world;
    w.bits_offset = 0;
    const auto plan = mission_plan_bits(m, bits.data());
    m.paths = path_mission(w, plan.data(), m.start, m.goals);
}

// Training set: gen_missions plus a fraction of 3D missions (kinds in rotation, blockers
// weighted double since they are the key scenario). RLV2_MIX3D sets the fraction (default 0).
inline MissionSet gen_missions_mix(uint32_t count, uint32_t base_seed) {
    const char *v = std::getenv("RLV2_MIX3D");
    const float frac = v ? std::stof(v) : 0;
    const uint32_t n3 = uint32_t(count * frac + .5f), n0 = count - n3;
    MissionSet set = n0 ? gen_missions(n0, base_seed) : MissionSet{};
    std::vector<Mission> missions(n3);
    std::vector<std::vector<uint32_t>> bits(n3);
    // RLV2_MIX3D_NO_UNDER=1: replace the fly-under-a-table kind (4) with a second door (no fly-under goal).
    const bool no_under = std::getenv("RLV2_MIX3D_NO_UNDER") && std::string(std::getenv("RLV2_MIX3D_NO_UNDER")) == "1";
    const int order[9] = {0, 1, 2, 3, 0, 1, no_under ? 3 : 4, 5, 6};
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; t++)
        pool.emplace_back([&, t] {
            for (uint32_t i = t; i < n3; i += threads)
                gen3d_fill(base_seed + 500009u + i * 7919u, order[i % 9], missions[i], bits[i]);
        });
    for (auto &p : pool)
        p.join();
    for (uint32_t i = 0; i < n3; i++) {
        missions[i].world.bits_offset = uint32_t(set.bits.size());
        set.bits.insert(set.bits.end(), bits[i].begin(), bits[i].end());
        set.missions.push_back(missions[i]);
    }
    if (n3)
        std::fprintf(stderr, "missions: %u procedural + %u 3D\n", n0, n3);
    return set;
}
