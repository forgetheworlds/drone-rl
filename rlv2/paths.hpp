#pragma once

// Reference paths for the privileged teacher. For each leg of a mission it finds a short path
// through free space that keeps away from walls, and stores it as up to NT_PATH points with the
// remaining path length to the waypoint in w. Moving boxes are ignored; the teacher sees them.

#include "voxel_world.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <vector>

#ifndef NT_PATH
#define NT_PATH 16
#endif

struct PathGrid {
    int nx, ny, nz;
    float scale;
    std::vector<float> clearance; // metres from the cell center to the nearest solid cell
    size_t index(int x, int y, int z) const { return (size_t(z) * ny + y) * nx + x; }
    bool inside(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz; }
    float at(VxVec p) const {
        const int x = int(std::floor(p.x / scale)), y = int(std::floor(p.y / scale)), z = int(std::floor(p.z / scale));
        return inside(x, y, z) ? clearance[index(x, y, z)] : 0;
    }
};

// Exact squared distance transform along one line (Felzenszwalb and Huttenlocher).
inline void path_edt_line(std::vector<float> &f, std::vector<float> &d, std::vector<int> &v, std::vector<float> &z) {
    const int n = int(f.size());
    int k = 0;
    v[0] = 0;
    z[0] = -1e20f;
    z[1] = 1e20f;
    for (int q = 1; q < n; q++) {
        float s;
        while (true) {
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * (q - v[k]));
            if (s > z[k] || k == 0)
                break;
            --k;
        }
        if (s <= z[k] && k == 0) {
            v[0] = q;
            z[0] = -1e20f;
            z[1] = 1e20f;
            continue;
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = 1e20f;
    }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q)
            ++k;
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
}

inline PathGrid path_grid(const VxWorld &w, const uint32_t *bits) {
    PathGrid g{int(w.nx), int(w.ny), int(w.nz), w.scale, {}};
    std::vector<float> e(size_t(g.nx) * g.ny * g.nz);
    for (int z = 0; z < g.nz; z++)
        for (int y = 0; y < g.ny; y++)
            for (int x = 0; x < g.nx; x++)
                e[g.index(x, y, z)] = vx_solid(w, bits, x, y, z) ? 0 : 1e20f;
    const int n[3] = {g.nx, g.ny, g.nz};
    for (int axis = 0; axis < 3; axis++) {
        const int len = n[axis];
        std::vector<float> f(len), d(len), zz(len + 1);
        std::vector<int> v(len);
        const int a = (axis + 1) % 3, b = (axis + 2) % 3;
        for (int i = 0; i < n[a]; i++)
            for (int j = 0; j < n[b]; j++) {
                int c[3];
                c[a] = i;
                c[b] = j;
                for (int q = 0; q < len; q++) {
                    c[axis] = q;
                    f[q] = e[g.index(c[0], c[1], c[2])];
                }
                path_edt_line(f, d, v, zz);
                for (int q = 0; q < len; q++) {
                    c[axis] = q;
                    e[g.index(c[0], c[1], c[2])] = d[q];
                }
            }
    }
    g.clearance.resize(e.size());
    for (size_t i = 0; i < e.size(); i++)
        g.clearance[i] = std::fmax(0.0f, (std::sqrt(e[i]) - .5f) * g.scale);
    return g;
}

inline bool path_segment_clear(const PathGrid &g, VxVec a, VxVec b, float margin) {
    const float length = vx_norm(vx_sub(b, a));
    const int steps = std::max(1, int(length / .05f));
    for (int i = 0; i <= steps; i++)
        if (g.at(vx_add(a, vx_scale(vx_sub(b, a), float(i) / steps))) < margin)
            return false;
    return true;
}

// Path from a to b keeping `margin` from walls: corner points, empty if there is none.
inline std::vector<VxVec> path_corners(const PathGrid &g, VxVec a, VxVec b, float margin) {
    std::vector<VxVec> points;
    auto nearest = [&](VxVec p, int &out) {
        const int x0 = int(std::floor(p.x / g.scale)), y0 = int(std::floor(p.y / g.scale)), z0 = int(std::floor(p.z / g.scale));
        float best = 1e9f;
        out = -1;
        for (int dz = -4; dz <= 4; dz++)
            for (int dy = -4; dy <= 4; dy++)
                for (int dx = -4; dx <= 4; dx++) {
                    const int x = x0 + dx, y = y0 + dy, z = z0 + dz;
                    if (!g.inside(x, y, z) || g.clearance[g.index(x, y, z)] < margin)
                        continue;
                    const float d = float(dx * dx + dy * dy + dz * dz);
                    if (d < best) {
                        best = d;
                        out = int(g.index(x, y, z));
                    }
                }
        return out >= 0;
    };
    int start, goal;
    if (nearest(a, start) && nearest(b, goal)) {
        // Dijkstra from the goal, with a cost that grows close to walls.
        std::vector<float> cost(g.clearance.size(), 1e30f);
        std::vector<int> next(g.clearance.size(), -1);
        using Item = std::pair<float, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
        cost[goal] = 0;
        open.push({0, goal});
        while (!open.empty()) {
            const auto [c, i] = open.top();
            open.pop();
            if (c > cost[i])
                continue;
            if (i == start)
                break;
            const int x = i % g.nx, y = (i / g.nx) % g.ny, z = i / (g.nx * g.ny);
            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        if (!dx && !dy && !dz)
                            continue;
                        const int X = x + dx, Y = y + dy, Z = z + dz;
                        if (!g.inside(X, Y, Z))
                            continue;
                        const int j = int(g.index(X, Y, Z));
                        const float clear = g.clearance[j];
                        if (clear < margin)
                            continue;
                        const float step = g.scale * std::sqrt(float(dx * dx + dy * dy + dz * dz));
                        const float total = c + step * (1 + 3 * std::fmax(0.0f, .6f - clear) / .6f);
                        if (total < cost[j]) {
                            cost[j] = total;
                            next[j] = i;
                            open.push({total, j});
                        }
                    }
        }
        if (next[start] >= 0 || start == goal) {
            std::vector<VxVec> cells;
            for (int i = start; i >= 0; i = i == goal ? -1 : next[i]) {
                const int x = i % g.nx, y = (i / g.nx) % g.ny, z = i / (g.nx * g.ny);
                cells.push_back(vx_vec((x + .5f) * g.scale, (y + .5f) * g.scale, (z + .5f) * g.scale));
            }
            cells.front() = a;
            cells.back() = b;
            // Keep only the corners: jump to the farthest cell still reachable in a clear line.
            size_t i = 0;
            while (i + 1 < cells.size()) {
                size_t j = cells.size() - 1;
                while (j > i + 1 && !path_segment_clear(g, cells[i], cells[j], margin + .03f))
                    --j;
                points.push_back(cells[j]);
                i = j;
            }
        }
    }
    return points;
}

// Path from a to b: NT_PATH points (x, y, z, remaining length), padded with b. Tries a wide
// margin first and narrows it only where the space requires.
inline std::array<float, NT_PATH * 4> path_plan(const PathGrid &g, VxVec a, VxVec b) {
    std::vector<VxVec> points;
    for (float margin : {.45f, .38f, .32f, .27f, .22f}) {
        points = path_corners(g, a, b, margin);
        if (!points.empty())
            break;
    }
    if (points.empty())
        points.push_back(b);
    if (points.size() > NT_PATH)
        points.erase(points.begin() + (NT_PATH - 1), points.end() - 1);
    std::array<float, NT_PATH * 4> out{};
    float remaining = 0;
    for (int k = NT_PATH - 1; k >= 0; k--) {
        const VxVec p = points[std::min(size_t(k), points.size() - 1)];
        if (k + 1 < NT_PATH && size_t(k + 1) < points.size())
            remaining += vx_norm(vx_sub(points[k + 1], p));
        out[k * 4] = p.x;
        out[k * 4 + 1] = p.y;
        out[k * 4 + 2] = p.z;
        out[k * 4 + 3] = remaining;
    }
    return out;
}

// Paths for every leg of a mission, NT_PATH * 4 floats per goal.
inline std::vector<float> path_mission(const VxWorld &w, const uint32_t *bits, const std::array<float, 3> &start,
                                       const std::vector<std::array<float, 3>> &goals) {
    const PathGrid g = path_grid(w, bits);
    std::vector<float> out;
    VxVec a = vx_vec(start[0], start[1], start[2]);
    for (const auto &goal : goals) {
        const VxVec b = vx_vec(goal[0], goal[1], goal[2]);
        const auto p = path_plan(g, a, b);
        out.insert(out.end(), p.begin(), p.end());
        a = b;
    }
    return out;
}
