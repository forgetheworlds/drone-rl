#pragma once

// Static voxel world loaded from a .3dmap file.

#include "navigator.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using safe_nav::Vec3;

class VoxelWorld {
  public:
    int nx = 0, ny = 0, nz = 0;
    float scale = .5f;
    std::vector<uint8_t> occupied;
    explicit VoxelWorld(const std::string &path) {
        std::ifstream file(path);
        std::string magic;
        file >> magic >> nx >> ny >> nz;
        if (magic != "voxel" || nx <= 0 || ny <= 0 || nz <= 0 || uint64_t(nx) * ny * nz > 600000000)
            throw std::runtime_error("Bad public voxel map");
        occupied.assign(size_t(nx) * ny * nz, 0);
        int x, y, z;
        while (file >> x >> y >> z) {
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
                throw std::runtime_error("Bad occupied voxel");
            occupied[(size_t(z) * ny + y) * nx + x] = 1;
        }
    }
    bool solid(int x, int y, int z) const {
        return x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz ||
               occupied[(size_t(z) * ny + y) * nx + x];
    }
    float ray(Vec3 p, Vec3 d, float maximum = 12) const {
        int cell[3] = {int(std::floor(p.x / scale)), int(std::floor(p.y / scale)),
                       int(std::floor(p.z / scale))};
        float point[3] = {p.x, p.y, p.z}, dir[3] = {d.x, d.y, d.z}, tmax[3], delta[3];
        int step[3];
        for (int a = 0; a < 3; a++) {
            step[a] = dir[a] >= 0 ? 1 : -1;
            tmax[a] = std::fabs(dir[a]) < 1e-8f
                          ? 1e30f
                          : ((cell[a] + (step[a] > 0 ? 1 : 0)) * scale - point[a]) / dir[a];
            delta[a] = std::fabs(dir[a]) < 1e-8f ? 1e30f : scale / std::fabs(dir[a]);
        }
        float distance = 0;
        while (distance < maximum) {
            if (solid(cell[0], cell[1], cell[2]))
                return distance;
            int a = tmax[0] < tmax[1] ? (tmax[0] < tmax[2] ? 0 : 2) : (tmax[1] < tmax[2] ? 1 : 2);
            distance = tmax[a];
            tmax[a] += delta[a];
            cell[a] += step[a];
        }
        return maximum;
    }
    bool contact(Vec3 p, float radius = .18f) const {
        int x = int(std::floor(p.x / scale)), y = int(std::floor(p.y / scale)),
            z = int(std::floor(p.z / scale));
        int n = int(std::ceil(radius / scale));
        for (int dz = -n; dz <= n; dz++)
            for (int dy = -n; dy <= n; dy++)
                for (int dx = -n; dx <= n; dx++) {
                    if (!solid(x + dx, y + dy, z + dz))
                        continue;
                    float lo[3] = {(x + dx) * scale, (y + dy) * scale, (z + dz) * scale},
                          hi[3] = {(x + dx + 1) * scale, (y + dy + 1) * scale,
                                   (z + dz + 1) * scale},
                          point[3] = {p.x, p.y, p.z};
                    float d2 = 0;
                    for (int a = 0; a < 3; a++) {
                        float d = point[a] - std::clamp(point[a], lo[a], hi[a]);
                        d2 += d * d;
                    }
                    if (d2 <= radius * radius)
                        return true;
                }
        return false;
    }
    std::array<float, 320> depth(const safe_nav::Pose &pose) const {
        std::array<float, 320> result;
        for (int k = 0; k < 320; k++) {
            Vec3 d = safe_nav::unit({1, 1 - (k % 20 + .5f) / 10, .75f * (1 - (k / 20 + .5f) / 8)});
            result[k] = ray(pose.position, safe_nav::rotate(pose, d));
        }
        return result;
    }
};
