#pragma once

// Rolling voxel map built from depth frames, A* route search over observed free space,
// and velocity/yaw-rate command selection.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace safe_nav {

inline float sensor_slope = 0.75f;
struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
inline float norm(Vec3 p) {
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
inline Vec3 unit(Vec3 p) {
    return p * (1 / std::max(norm(p), 1e-6f));
}
inline float wrap(float a) {
    return std::atan2(std::sin(a), std::cos(a));
}
struct Pose {
    Vec3 position;
    std::array<float, 9> rotation;
};
inline Vec3 rotate(const Pose &p, Vec3 v) {
    auto &r = p.rotation;
    return {r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z,
            r[6] * v.x + r[7] * v.y + r[8] * v.z};
}
inline Vec3 inverse_rotate(const Pose &p, Vec3 v) {
    auto &r = p.rotation;
    return {r[0] * v.x + r[3] * v.y + r[6] * v.z, r[1] * v.x + r[4] * v.y + r[7] * v.z,
            r[2] * v.x + r[5] * v.y + r[8] * v.z};
}
struct Command {
    Vec3 world_velocity;
    float yaw_rate = 0;
    bool path_found = false;
};

class Navigator {
  public:
    static constexpr uint32_t map_layout_version = 2;
    static constexpr int nx = 24, ny = 24, nz = 12, cells = nx * ny * nz;
    static constexpr float resolution = .25f, body_radius = .18f, margin = .08f;
    static constexpr float max_speed = .65f, brake_acceleration = .8f;
    static constexpr int feature_count = 226;

    struct RouteSearchStorage {
        uint16_t *costs;
        uint16_t *frontier;
        int16_t *positions;
        uint8_t *parent_directions;
    };
    RouteSearchStorage route_search_storage() {
        return {distance.data(), heap.data(), heap_position.data(), parent.data()};
    }
    uint32_t expanded_nodes = 0;
    uint32_t commands_rejected = 0;
    Vec3 map_cell_center(int i) const { return center(i); }
    int map_cell_state(int i) const { return occupancy[i] < 0 ? -1 : occupancy[i]; }
    bool has_additional_free_evidence(int i) const {
        return occupancy[i] == 0 || occupancy[i] == -2;
    }
    uint8_t map_cell_free_mask(int i) const {
        return (fine_body_checks || refined_free_space) ? free_subcells[i] : 0;
    }
    uint8_t captured_subcell_mask(int i) const { return free_subcells[i]; }
    void enable_fine_body_checks(bool enabled) {
        fine_body_checks = enabled;
        track_subcell_evidence = track_subcell_evidence || enabled;
    }
    void track_subcells(bool enabled) { track_subcell_evidence = enabled; }
    void enable_refined_free_space(bool enabled) {
        refined_free_space = enabled;
        track_subcell_evidence = track_subcell_evidence || enabled;
        for (int i = 0; i < cells; i++) {
            if (!enabled && occupancy[i] == -2)
                occupancy[i] = 0;
            if (enabled && occupancy[i] == 0 && free_subcells[i] == 255)
                occupancy[i] = -2;
        }
    }
    Vec3 map_subcell_center(int cell, int part) const {
        return center(cell) + Vec3{(part & 1) ? .0625f : -.0625f, (part & 2) ? .0625f : -.0625f,
                                   (part & 4) ? .0625f : -.0625f};
    }

    bool measured_clear(Vec3 p) const { return clear(p, body_radius + margin); }

    void reset(Vec3 position) {
        occupancy.fill(0);
        free_subcells.fill(0);
        age.fill(0);
        visits.fill(0);
        initialized = true;
        tick = 0;
        waypoint_valid = false;
        waypoint = {};
        last_goal = {};
        inspection_active = false;
        inspection_mask = 0;
        inspection_hold = 0;
        origin = {std::floor(position.x / resolution) - nx / 2.0f,
                  std::floor(position.y / resolution) - ny / 2.0f,
                  std::floor(position.z / resolution) - nz / 2.0f};

        int launch = index(position);
        if (launch >= 0) {
            occupancy[launch] = -1;
            free_subcells[launch] = 255;
        }
    }

    void observe(const Pose &pose, const std::array<float, 320> &depth) {
        if (!initialized)
            reset(pose.position);
        shift(pose.position);
        ++tick;
        for (int i = 0; i < cells; i++) {
            if (age[i] < 255)
                ++age[i];

            Vec3 b = inverse_rotate(pose, center(i) - pose.position);
            float distance = norm(b);
            if (b.x < .01f || distance > 4.0f)
                continue;
            float u = (1 - b.y / b.x) * 10 - .5f, v = (1 - b.z / (sensor_slope * b.x)) * 8 - .5f;
            int col = int(std::round(u)), row = int(std::round(v));
            if (col < 1 || col > 18 || row < 1 || row > 14)
                continue;
            float nearest = 12;
            bool valid_footprint = true;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    float range = depth[(row + dy) * 20 + col + dx];

                    if (!std::isfinite(range) || range <= .01f || range > 12)
                        valid_footprint = false;
                    else
                        nearest = std::min(nearest, range);
                }

            if (valid_footprint && distance + .24f < nearest) {
                occupancy[i] = -1;
                free_subcells[i] = 255;
                age[i] = 0;
            }
        }
        for (int k = 0; k < 320; k++) {
            float d = depth[k];
            if (!std::isfinite(d) || d <= .01f || d >= 11.9f)
                continue;
            Vec3 ray = unit({1, 1 - (k % 20 + .5f) / 10, .75f * (1 - (k / 20 + .5f) / 8)});

            int i = index(pose.position + rotate(pose, ray) * (d + .002f));
            if (i >= 0) {
                occupancy[i] = 1;
                free_subcells[i] = 0;
                age[i] = 0;
            }
        }
        if (track_subcell_evidence)
            refine_observations(pose, depth);
    }

    Command plan(const Pose &pose, Vec3 velocity, Vec3 goal) {
        const Vec3 position = pose.position;
        expanded_nodes = 0;
        int start = index(position);
        if (start < 0)
            return {};
        if (norm(goal - last_goal) > .25f) {
            waypoint_valid = false;
            inspection_active = false;
            inspection_mask = 0;
        }
        last_goal = goal;

        if (norm(goal - position) < 1.0f) {
            bool found = false;
            Vec3 target{};
            float score = 1e20f;
            for (int z = -1; z <= 1; z++)
                for (int y = -1; y <= 1; y++)
                    for (int x = -1; x <= 1; x++) {
                        Vec3 offset = {x * .075f, y * .075f, z * .075f};
                        Vec3 candidate = goal + offset;
                        float rank = norm(offset) + .01f * norm(candidate - position);
                        if (rank < score && clear(candidate, body_radius + margin) &&
                            segment_clear(position, candidate, body_radius + margin)) {
                            target = candidate;
                            score = rank;
                            found = true;
                        }
                    }
            if (found) {
                Vec3 difference = target - position;
                float length = norm(difference);
                Command c;
                c.path_found = true;
                c.world_velocity = difference * std::min(1.0f, max_speed / std::max(length, 1e-6f));
                if (length < .04f)
                    c.world_velocity = {};
                if (std::hypot(difference.x, difference.y) > .05f) {
                    float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
                    c.yaw_rate = std::clamp(2 * wrap(std::atan2(difference.y, difference.x) - yaw),
                                            -.5f, .5f);
                }
                return protect(pose, velocity, c);
            }
        }

        int goal_cell = index(goal), focus_cell = index(inspection_focus);
        if (inspection_active && focus_cell >= 0 && point_measured(inspection_focus))
            inspection_active = false;
        Vec3 delta = goal - position;
        if (!inspection_active && goal_cell >= 0 && occupancy[goal_cell] == 0 &&
            !point_measured(goal) && std::fabs(delta.z) > .35f &&
            std::hypot(delta.x, delta.y) < 1.0f) {
            float radius = std::clamp(std::fabs(delta.z) / .50f + .25f, .8f, 3.8f),
                  best_rank = 1e20f;
            for (int sector = 0; sector < 8; sector++) {
                if (inspection_mask & (1u << sector))
                    continue;
                float angle = float(sector) * .7853981634f;
                Vec3 candidate = {goal.x + radius * std::cos(angle),
                                  goal.y + radius * std::sin(angle), position.z};
                float rank = norm(candidate - position);
                if (rank < best_rank && clear(candidate, body_radius + margin) &&
                    segment_clear(position, candidate, body_radius + margin)) {
                    inspection_target = candidate;
                    inspection_focus = goal;
                    inspection_sector = sector;
                    best_rank = rank;
                    inspection_active = true;
                    inspection_hold = 0;
                }
            }
        }
        if (inspection_active) {
            if (!clear(inspection_target, body_radius + margin) ||
                !segment_clear(position, inspection_target, body_radius + margin)) {
                inspection_mask |= uint8_t(1u << inspection_sector);
                inspection_active = false;
            } else {
                Vec3 difference = inspection_target - position;
                float length = norm(difference);
                Command c;
                c.path_found = true;
                c.world_velocity = difference * std::min(1.0f, max_speed / std::max(length, 1e-6f));
                float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
                float error = wrap(
                    std::atan2(inspection_focus.y - position.y, inspection_focus.x - position.x) -
                    yaw);
                c.yaw_rate = std::clamp(2 * error, -.5f, .5f);
                if (length < .15f)
                    c.world_velocity = {};
                if (length < .20f && std::fabs(error) < .10f)
                    ++inspection_hold;
                if (inspection_hold >= 12) {
                    inspection_mask |= uint8_t(1u << inspection_sector);
                    inspection_active = false;
                }
                return protect(pose, velocity, c);
            }
        }
        if (waypoint_valid && (norm(position - waypoint) < .20f || index(waypoint) == start ||
                               !clear(waypoint, body_radius + margin) ||
                               !segment_clear(position, waypoint, body_radius + margin)))
            waypoint_valid = false;
        if (visits[start] < 250)
            ++visits[start];
        distance.fill(65535);
        parent.fill(255);
        heap_position.fill(-1);
        heap_size = 0;
        route_goal = goal;
        distance[start] = 0;
        push(start);
        Vec3 search_goal = {
            std::clamp(goal.x, (origin.x + 1.5f) * resolution, (origin.x + nx - 1.5f) * resolution),
            std::clamp(goal.y, (origin.y + 1.5f) * resolution, (origin.y + ny - 1.5f) * resolution),
            std::clamp(goal.z, (origin.z + 1.5f) * resolution,
                       (origin.z + nz - 1.5f) * resolution)};
        int search_goal_cell = index(search_goal);
        int best = start;
        float best_score = std::numeric_limits<float>::infinity();
        bool reached = false;
        while (heap_size && expanded_nodes < cells) {
            int current = pop();
            ++expanded_nodes;
            float remaining = norm(center(current) - goal);

            float score = remaining + .15f * distance[current] * resolution / 100 +
                          .002f * std::min<int>(visits[current], 40);
            if (distance[current] >= 100 && score < best_score) {
                best_score = score;
                best = current;
            }
            if (current == search_goal_cell) {
                best = current;
                reached = remaining < .30f;
                break;
            }
            int x = current % nx, y = (current / nx) % ny, z = current / (nx * ny);
            for (int axis = 0; axis < 3; axis++)
                for (int sign : {-1, 1}) {
                    int xx = x + (axis == 0 ? sign : 0), yy = y + (axis == 1 ? sign : 0),
                        zz = z + (axis == 2 ? sign : 0);
                    if (xx < 1 || xx >= nx - 1 || yy < 1 || yy >= ny - 1 || zz < 1 || zz >= nz - 1)
                        continue;
                    int next = (zz * ny + yy) * nx + xx;

                    int cost = occupancy[next] == 0 ? 200 : 100;
                    if (!clear_center(next, false) || distance[current] + cost >= distance[next])
                        continue;
                    distance[next] = distance[current] + cost;
                    parent[next] = uint8_t(axis * 2 + (sign > 0 ? 1 : 0));
                    push(next);
                }
        }
        Command command;

        int path_length = 0, node = best;
        while (parent[node] != 255 && path_length < cells) {
            heap[path_length++] = uint16_t(node);
            int dir = parent[node], step = (dir / 2 == 0   ? 1
                                            : dir / 2 == 1 ? nx
                                                           : nx * ny) *
                                           (dir % 2 ? 1 : -1);
            node -= step;
        }
        int observed = start;
        Vec3 unknown_target = goal;
        bool has_unknown = false;
        for (int step = path_length - 1; step >= 0; step--) {
            int cell = heap[step];
            if (!clear_center(cell)) {
                unknown_target = center(cell);
                has_unknown = true;
                break;
            }
            observed = cell;
        }
        best = observed;
        if (has_unknown)
            reached = false;
        if (best == start && has_unknown) {
            Vec3 bearing = unknown_target - position;
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            command.yaw_rate =
                std::hypot(bearing.x, bearing.y) > .05f
                    ? std::clamp(2 * wrap(std::atan2(bearing.y, bearing.x) - yaw), -.5f, .5f)
                    : .5f;
            if (std::fabs(bearing.z) > .15f && std::hypot(bearing.x, bearing.y) < .65f) {
                if (norm(inspection_focus - unknown_target) > .25f)
                    inspection_mask = 0;
                float radius = std::clamp(std::fabs(bearing.z) / .5f + .5f, .8f, 3.8f),
                      rank = 1e20f;
                for (int sector = 0; sector < 8; sector++) {
                    if (inspection_mask & (1u << sector))
                        continue;
                    float angle = sector * .7853981634f;
                    Vec3 candidate = {unknown_target.x + radius * std::cos(angle),
                                      unknown_target.y + radius * std::sin(angle), position.z};
                    float d = norm(candidate - position);
                    if (d < rank && clear(candidate, body_radius + margin) &&
                        segment_clear(position, candidate, body_radius + margin)) {
                        inspection_target = candidate;
                        inspection_focus = unknown_target;
                        inspection_sector = sector;
                        inspection_active = true;
                        inspection_hold = 0;
                        rank = d;
                    }
                }
            }
            return protect(pose, velocity, command);
        }
        if (best != start || reached) {
            int first = best;
            int count = 0;
            while (parent[first] != 255 && count++ < cells) {
                int dir = parent[first], delta = (dir / 2 == 0   ? 1
                                                  : dir / 2 == 1 ? nx
                                                                 : nx * ny) *
                                                 (dir % 2 ? 1 : -1);
                int previous = first - delta;
                if (previous == start)
                    break;
                first = previous;
            }

            Vec3 target = center(best);
            int probe = best;
            count = 0;
            while (parent[probe] != 255 && count++ < cells) {
                if (segment_clear(position, center(probe), body_radius + margin)) {
                    target = center(probe);
                    break;
                }
                int dir = parent[probe], delta = (dir / 2 == 0   ? 1
                                                  : dir / 2 == 1 ? nx
                                                                 : nx * ny) *
                                                 (dir % 2 ? 1 : -1);
                probe -= delta;
            }
            if (reached && segment_clear(position, goal, body_radius + margin))
                target = goal;
            if (!segment_clear(position, target, body_radius + margin))
                target = center(first);
            if (!waypoint_valid) {
                waypoint = target;
                waypoint_valid = true;
            }
            target = waypoint;
            Vec3 difference = target - position;
            float waypoint_distance = norm(difference);
            float goal_distance = norm(goal - position);
            float speed = std::min(max_speed, std::sqrt(2 * brake_acceleration *
                                                        std::max(0.0f, goal_distance - .16f)));
            speed = std::min(speed, std::sqrt(2 * brake_acceleration *
                                              std::max(0.0f, waypoint_distance - .15f)));
            command.world_velocity = unit(difference) * speed;
            command.path_found = true;
            if (reached && goal_distance < .18f)
                command.world_velocity = {};
        }
        Vec3 heading = command.path_found ? command.world_velocity : goal - position;
        if (std::hypot(heading.x, heading.y) > .05f) {
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            command.yaw_rate =
                std::clamp(2 * wrap(std::atan2(heading.y, heading.x) - yaw), -.5f, .5f);
        }

        if (!command.path_found)
            command.yaw_rate = .5f;
        return protect(pose, velocity, command);
    }

    Command protect(const Pose &pose, Vec3 velocity, Command command) {

        float speed = norm(velocity), requested = norm(command.world_velocity);
        float stopping = speed * .15f + speed * speed / (2 * brake_acceleration);
        if (requested > 0) {
            Vec3 direction = unit(command.world_velocity);
            bool accepted = false;

            for (int attempt = 0; attempt < 5; attempt++) {
                float travel = requested * .15f + requested * requested / (2 * brake_acceleration);
                Vec3 endpoint = pose.position + direction * travel;
                if (segment_clear(pose.position, endpoint, body_radius + margin)) {
                    command.world_velocity = direction * requested;
                    accepted = true;
                    break;
                }
                requested *= .5f;
            }
            if (!accepted) {
                command.world_velocity = {};
                command.path_found = false;
                command.yaw_rate = .5f;
                ++commands_rejected;
            }
        }
        if (speed > .08f && !segment_clear(pose.position, pose.position + unit(velocity) * stopping,
                                           body_radius + margin)) {
            command.world_velocity = {};
            command.path_found = false;
            command.yaw_rate = .5f;
            ++commands_rejected;
        }
        return command;
    }

    void features(const Pose &pose, Vec3 velocity, Vec3 goal, float out[feature_count]) const {
        int cursor = 0;

        for (int z = 0; z < 3; z++)
            for (int y = 0; y < 6; y++)
                for (int x = 0; x < 6; x++) {
                    float sum = 0, unknown = 0;
                    for (int dz = 0; dz < 4; dz++)
                        for (int dy = 0; dy < 4; dy++)
                            for (int dx = 0; dx < 4; dx++) {
                                int i = (((z * 4 + dz) * ny + y * 4 + dy) * nx + x * 4 + dx);
                                sum += map_cell_state(i);
                                unknown += occupancy[i] == 0 ? 1.0f : 0.0f;
                            }
                    out[cursor++] = sum / 64;
                    out[cursor++] = unknown / 64;
                }
        Vec3 g = unit(goal - pose.position);
        for (float a : {g.x, g.y, g.z})
            out[cursor++] = a;
        for (float a : {velocity.x, velocity.y, velocity.z})
            out[cursor++] = std::clamp(a / 2, -1.0f, 1.0f);
        out[cursor++] = pose.rotation[0];
        out[cursor++] = pose.rotation[3];
        out[cursor++] = std::min(norm(goal - pose.position) / 12, 1.0f);
        out[cursor++] = measured_clear(pose.position) ? 1.0f : 0.0f;
    }

  private:
    std::array<int8_t, cells> occupancy{};
    std::array<uint8_t, cells> free_subcells{};
    bool track_subcell_evidence = false, refined_free_space = false, fine_body_checks = false;
    std::array<uint8_t, cells> age{}, parent{}, visits{};
    std::array<uint16_t, cells> distance{}, heap{};
    std::array<int16_t, cells> heap_position{};
    Vec3 origin{}, route_goal{}, waypoint{}, last_goal{};
    Vec3 inspection_target{}, inspection_focus{};
    uint8_t inspection_mask = 0, inspection_sector = 0;
    int inspection_hold = 0;
    bool inspection_active = false;
    int heap_size = 0;
    uint32_t tick = 0;
    bool initialized = false;
    bool waypoint_valid = false;

    bool point_measured(Vec3 p) const {
        int cell = index(p);
        if (cell < 0)
            return false;
        if (occupancy[cell] < 0)
            return true;
        if (occupancy[cell] > 0 || !refined_free_space)
            return false;
        Vec3 lower = center(cell) - Vec3{.125f, .125f, .125f};
        int part = (p.x - lower.x >= .125f ? 1 : 0) | (p.y - lower.y >= .125f ? 2 : 0) |
                   (p.z - lower.z >= .125f ? 4 : 0);
        return (free_subcells[cell] & (1u << part)) != 0;
    }
    void refine_observations(const Pose &pose, const std::array<float, 320> &depth) {
        for (int cell = 0; cell < cells; cell++) {
            if (occupancy[cell] != 0 || free_subcells[cell] == 255)
                continue;
            if (norm(center(cell) - pose.position) > 4.12f)
                continue;
            for (int part = 0; part < 8; part++) {
                if (free_subcells[cell] & (1u << part))
                    continue;
                Vec3 point = inverse_rotate(pose, map_subcell_center(cell, part) - pose.position);
                float distance = norm(point);
                if (point.x < .01f || distance > 4)
                    continue;
                int col = int(std::round((1 - point.y / point.x) * 10 - .5f));
                int row = int(std::round((1 - point.z / (.75f * point.x)) * 8 - .5f));
                if (col < 1 || col > 18 || row < 1 || row > 14)
                    continue;
                float nearest = 12;
                bool valid = true;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        float range = depth[(row + dy) * 20 + col + dx];
                        if (!std::isfinite(range) || range <= .01f || range > 12)
                            valid = false;
                        else
                            nearest = std::min(nearest, range);
                    }

                if (valid && distance + .12f < nearest)
                    free_subcells[cell] |= uint8_t(1u << part);
            }
            if (refined_free_space && free_subcells[cell] == 255)
                occupancy[cell] = -2;
        }
    }
    int index(Vec3 p) const {
        int x = int(std::floor(p.x / resolution - origin.x)),
            y = int(std::floor(p.y / resolution - origin.y)),
            z = int(std::floor(p.z / resolution - origin.z));
        return x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz ? -1
                                                                        : (z * ny + y) * nx + x;
    }
    Vec3 center(int i) const {
        return {(origin.x + i % nx + .5f) * resolution,
                (origin.y + (i / nx) % ny + .5f) * resolution,
                (origin.z + i / (nx * ny) + .5f) * resolution};
    }

    bool clear_center(int i, bool require_observed = true) const {
        constexpr float radius = body_radius + margin;
        static_assert(radius * radius >= 3 * .25f * resolution * resolution &&
                          radius < 1.5f * resolution,
                      "Recheck center footprint stencil when geometry changes");
        if (i < 0 || i >= cells || occupancy[i] > 0 ||
            (require_observed &&
             !(occupancy[i] < 0 || (refined_free_space && free_subcells[i] != 0))))
            return false;
        int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
        if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1 || z < 1 || z >= nz - 1)
            return false;
        for (int dz = -1; dz <= 1; dz++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    if (occupancy[((z + dz) * ny + y + dy) * nx + x + dx] > 0)
                        return false;
        return true;
    }
    bool clear(Vec3 p, float radius, bool require_observed = true) const {
        int i = index(p);
        if (i < 0)
            return false;
        if (occupancy[i] > 0 || (require_observed && !point_measured(p)))
            return false;
        int x = i % nx, y = (i / nx) % ny, z = i / (nx * ny),
            n = int(std::ceil(radius / resolution));
        for (int dz = -n; dz <= n; dz++)
            for (int dy = -n; dy <= n; dy++)
                for (int dx = -n; dx <= n; dx++) {
                    int xx = x + dx, yy = y + dy, zz = z + dz;
                    Vec3 lo = {(origin.x + xx) * resolution, (origin.y + yy) * resolution,
                               (origin.z + zz) * resolution};
                    Vec3 d = {p.x - std::clamp(p.x, lo.x, lo.x + resolution),
                              p.y - std::clamp(p.y, lo.y, lo.y + resolution),
                              p.z - std::clamp(p.z, lo.z, lo.z + resolution)};
                    if (d.x * d.x + d.y * d.y + d.z * d.z > radius * radius)
                        continue;
                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz ||
                        occupancy[(zz * ny + yy) * nx + xx] > 0)
                        return false;
                }
        return true;
    }
    bool segment_clear(Vec3 a, Vec3 b, float radius) const {
        int steps = std::max(1, int(std::ceil(norm(b - a) / .08f)));
        for (int i = 0; i <= steps; i++)
            if (!clear(a + (b - a) * (float(i) / steps), radius))
                return false;
        return true;
    }
    void shift(Vec3 position) {
        Vec3 desired = {std::floor(position.x / resolution) - nx / 2.0f,
                        std::floor(position.y / resolution) - ny / 2.0f,
                        std::floor(position.z / resolution) - nz / 2.0f};
        int dx = int(desired.x - origin.x), dy = int(desired.y - origin.y),
            dz = int(desired.z - origin.z);
        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) < 3)
            return;

        for (int iz = 0; iz < nz; iz++)
            for (int iy = 0; iy < ny; iy++)
                for (int ix = 0; ix < nx; ix++) {
                    int x = dx >= 0 ? ix : nx - 1 - ix, y = dy >= 0 ? iy : ny - 1 - iy,
                        z = dz >= 0 ? iz : nz - 1 - iz;
                    int xx = x + dx, yy = y + dy, zz = z + dz, to = (z * ny + y) * nx + x;
                    if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) {
                        occupancy[to] = 0;
                        free_subcells[to] = 0;
                        age[to] = 0;
                        visits[to] = 0;
                    } else {
                        int from = (zz * ny + yy) * nx + xx;
                        occupancy[to] = occupancy[from];
                        free_subcells[to] = free_subcells[from];
                        age[to] = age[from];
                        visits[to] = visits[from];
                    }
                }
        origin = desired;
    }
    float priority(int i) const {
        Vec3 d = center(i) - route_goal;
        return distance[i] + 100 * (std::fabs(d.x) + std::fabs(d.y) + std::fabs(d.z)) / resolution;
    }
    bool precedes(int a, int b) const {
        float pa = priority(a), pb = priority(b);
        return std::fabs(pa - pb) < .01f ? distance[a] > distance[b] : pa < pb;
    }
    void exchange(int a, int b) {
        std::swap(heap[a], heap[b]);
        heap_position[heap[a]] = a;
        heap_position[heap[b]] = b;
    }
    void push(int cell) {
        int pos = heap_position[cell];
        if (pos < 0) {
            pos = heap_size++;
            heap[pos] = uint16_t(cell);
            heap_position[cell] = int16_t(pos);
        }
        while (pos && precedes(heap[pos], heap[(pos - 1) / 2])) {
            exchange(pos, (pos - 1) / 2);
            pos = (pos - 1) / 2;
        }
    }
    int pop() {
        int cell = heap[0];
        heap_position[cell] = -1;
        --heap_size;
        if (heap_size) {
            heap[0] = heap[heap_size];
            heap_position[heap[0]] = 0;
            int pos = 0;
            while (pos * 2 + 1 < heap_size) {
                int child = pos * 2 + 1;
                if (child + 1 < heap_size && precedes(heap[child + 1], heap[child]))
                    ++child;
                if (!precedes(heap[child], heap[pos]))
                    break;
                exchange(pos, child);
                pos = child;
            }
        }
        return cell;
    }
};
} // namespace safe_nav
