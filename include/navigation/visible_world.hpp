#pragma once

// Simulation world: static voxel map plus moving boxes, depth rendering and contact checks.

#include "voxel_world.hpp"

struct MovingBox {
    Vec3 center, half;
    int axis = 1;
    float amplitude = 0, period = 1, phase = 0;
};
class VisibleWorld : public VoxelWorld {
  public:
    std::vector<MovingBox> movers;
    float time = 0;
    explicit VisibleWorld(const std::string &path) : VoxelWorld(path) {}
    Vec3 center(const MovingBox &box) const {
        Vec3 p = box.center;
        float offset = box.amplitude * std::cos(6.28318530718f * time / box.period + box.phase);
        if (box.axis == 0)
            p.x += offset;
        else if (box.axis == 1)
            p.y += offset;
        else
            p.z += offset;
        return p;
    }
    float ray(Vec3 p, Vec3 d, float maximum = 12) const {
        float nearest = VoxelWorld::ray(p, d, maximum);
        for (const auto &box : movers) {
            Vec3 c = center(box);
            float low[3] = {c.x - box.half.x, c.y - box.half.y, c.z - box.half.z};
            float high[3] = {c.x + box.half.x, c.y + box.half.y, c.z + box.half.z};
            float point[3] = {p.x, p.y, p.z}, direction[3] = {d.x, d.y, d.z};
            float entry = 0, exit = nearest;
            for (int axis = 0; axis < 3; axis++) {
                if (std::fabs(direction[axis]) < 1e-8f) {
                    if (point[axis] < low[axis] || point[axis] > high[axis])
                        exit = -1;
                } else {
                    float a = (low[axis] - point[axis]) / direction[axis],
                          b = (high[axis] - point[axis]) / direction[axis];
                    entry = std::max(entry, std::min(a, b));
                    exit = std::min(exit, std::max(a, b));
                }
            }
            if (exit >= entry)
                nearest = entry;
        }
        return nearest;
    }
    bool contact(Vec3 p, float radius = .18f) const {
        if (VoxelWorld::contact(p, radius))
            return true;
        for (const auto &box : movers) {
            Vec3 c = center(box);
            float dx = std::max(0.f, std::fabs(p.x - c.x) - box.half.x),
                  dy = std::max(0.f, std::fabs(p.y - c.y) - box.half.y),
                  dz = std::max(0.f, std::fabs(p.z - c.z) - box.half.z);
            if (dx * dx + dy * dy + dz * dz <= radius * radius)
                return true;
        }
        return false;
    }
    std::array<float, 320> tile_minimum(const safe_nav::Pose &pose) const {
        std::array<float, 320> result;
        result.fill(12);
        for (int row = 0; row < 240; row++)
            for (int col = 0; col < 320; col++) {
                auto d = safe_nav::rotate(
                    pose, safe_nav::unit({1, 1 - (col + .5f) / 160,
                                          (safe_nav::sensor_slope * (1 - (row + .5f) / 120))}));
                int tile = (row / 15) * 20 + col / 16;
                result[tile] = std::min(result[tile], ray(pose.position, d));
            }
        return result;
    }
    std::array<float, 320> depth(const safe_nav::Pose &pose) const {
        std::array<float, 320> frame;
        for (int k = 0; k < 320; k++) {
            auto d = safe_nav::rotate(
                pose, safe_nav::unit({1, 1 - (k % 20 + .5f) / 10,
                                      (safe_nav::sensor_slope * (1 - (k / 20 + .5f) / 8))}));
            frame[k] = ray(pose.position, d);
        }
        return frame;
    }
};
