#pragma once

// Finds an observed-free viewpoint from which the camera can inspect unknown space that
// blocks the current plan.

#include "known_space_backup.hpp"

namespace safe_nav {

class KnownSpaceViews {
  public:
    int last_seed = -1, last_reachable = 0;
    Vec3 diagnostic_target() const { return target; }
    bool diagnostic_active() const { return active; }
    Command propose(const Navigator &map, const Pose &pose, Vec3 goal, Vec3 focus,
                    float focus_padding = 0, float minimum_focus_range = 2 * Navigator::resolution,
                    float clearance_height = std::numeric_limits<float>::quiet_NaN(),
                    const Vec3 *secondary_focus = nullptr) {
        float requested_step =
            secondary_focus ? Navigator::resolution : Navigator::resolution * .5f;
        if (step != requested_step) {
            active = false;
            step = requested_step;
        }
        if (secondary_focus && norm(*secondary_focus - last_secondary) > Navigator::resolution) {
            active = false;
            last_secondary = *secondary_focus;
        }
        const float size = Navigator::resolution,
                    radius = Navigator::body_radius + Navigator::margin;
        if (active &&
            (norm(focus - last_focus) > size || norm(target - pose.position) < step * .5f ||
             !body_coverage(map, target, radius).known_free()))
            active = false;
        last_focus = focus;
        if (!active) {
            graph_origin = {pose.position.x - nx * step / 2, pose.position.y - ny * step / 2,
                            pose.position.z - nz * step / 2};
            parent.fill(-1);
            int first = nearest_seed(map, pose.position);
            int head = 0, tail = 0, best = -1, depth = 0, boundary = 1;
            float rank = std::numeric_limits<float>::infinity();
            last_seed = first;
            if (first >= 0) {
                queue[tail++] = uint16_t(first);
                parent[first] = -2;
            }
            while (head < tail) {
                int current = queue[head++];
                Vec3 candidate = point(current), view = focus - candidate;
                float horizontal = std::hypot(view.x, view.y), range = norm(view);
                if ((!std::isfinite(clearance_height) ||
                     std::fabs(candidate.z - clearance_height) <= step) &&
                    norm(candidate - pose.position) >= step && range >= minimum_focus_range &&
                    range <= 4 - size && std::fabs(view.z) <= .5f * horizontal &&
                    view_line_open(map, candidate, focus, focus_padding) &&
                    (!secondary_focus ||
                     joint_view_open(map, candidate, focus, *secondary_focus))) {
                    float cost = depth * step + norm(candidate - goal);
                    if (cost < rank) {
                        rank = cost;
                        best = current;
                    }
                }
                int x = current % nx, y = (current / nx) % ny, z = current / (nx * ny);
                for (int axis = 0; axis < 3; axis++)
                    for (int sign : {-1, 1}) {
                        int xx = x + (axis == 0 ? sign : 0), yy = y + (axis == 1 ? sign : 0),
                            zz = z + (axis == 2 ? sign : 0);
                        if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz)
                            continue;
                        int next = (zz * ny + yy) * nx + xx;
                        if (std::isfinite(clearance_height) &&
                            std::fabs(point(next).z - clearance_height) > step)
                            continue;
                        if (parent[next] != -1 || !center_known(map, next))
                            continue;
                        if (!swept_body_known_free(map, candidate, point(next), radius))
                            continue;
                        parent[next] = int16_t(current);
                        queue[tail++] = uint16_t(next);
                    }
                if (head == boundary) {
                    boundary = tail;
                    ++depth;
                }
            }
            last_reachable = tail;
            if (best >= 0) {
                int next = best;
                while (parent[next] >= 0 &&
                       !swept_body_known_free(map, pose.position, point(next), radius))
                    next = parent[next];
                target = point(next);
                selected_cell = next;
                active = true;
            }
        }
        Command result;
        if (active) {
            Vec3 d = target - pose.position;
            result.world_velocity = unit(d) * std::min(Navigator::max_speed, norm(d));
            result.path_found = true;
        }
        Vec3 view = focus - pose.position;
        if (secondary_focus) {
            Vec3 other = *secondary_focus - pose.position;
            view.z = 0;
            other.z = 0;
            view = unit(view) + unit(other);
        }
        if (std::hypot(view.x, view.y) > .05f) {
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            result.yaw_rate = std::clamp(2 * wrap(std::atan2(view.y, view.x) - yaw), -.5f, .5f);
        }
        return result;
    }
    Command next_path_edge(const Pose &pose, Vec3 focus,
                           const Vec3 *secondary_focus = nullptr) const {
        Command result;
        if (!active)
            return result;
        int next = selected_cell;
        while (parent[next] >= 0 && parent[parent[next]] >= 0)
            next = parent[next];
        Vec3 difference = point(next) - pose.position;
        result.world_velocity = unit(difference) * std::min(Navigator::max_speed, norm(difference));
        result.path_found = norm(difference) > 1e-4f;
        Vec3 view = focus - pose.position;
        if (secondary_focus) {
            Vec3 other = *secondary_focus - pose.position;
            view.z = 0;
            other.z = 0;
            view = unit(view) + unit(other);
        }
        if (std::hypot(view.x, view.y) > .05f) {
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            result.yaw_rate = std::clamp(2 * wrap(std::atan2(view.y, view.x) - yaw), -.5f, .5f);
        }
        return result;
    }
    void reset() { active = false; }

  private:
    static constexpr int nx = 16, ny = 16, nz = 16, cells = nx * ny * nz;
    float step = Navigator::resolution * .5f;
    Vec3 last_secondary{};
    Vec3 graph_origin{};
    Vec3 point(int i) const {
        return graph_origin +
               Vec3{float(i % nx) * step, float((i / nx) % ny) * step, float(i / (nx * ny)) * step};
    }
    std::array<int16_t, cells> parent{};
    std::array<uint16_t, cells> queue{};
    Vec3 target{}, last_focus{};
    int selected_cell = 0;
    bool active = false;
    bool center_known(const Navigator &map, int cell) const {
        return body_coverage(map, point(cell), Navigator::body_radius + Navigator::margin)
            .known_free();
    }
    int nearest_seed(const Navigator &map, Vec3 position) const {
        int best = -1;
        float distance = 1e20f;
        for (int i = 0; i < cells; i++) {
            Vec3 p = point(i);
            float d = norm(p - position);
            if (d > Navigator::resolution * 3 || d >= distance || !center_known(map, i))
                continue;
            if (swept_body_known_free(map, position, p,
                                      Navigator::body_radius + Navigator::margin)) {
                best = i;
                distance = d;
            }
        }
        return best;
    }
    static bool joint_view_open(const Navigator &map, Vec3 candidate, Vec3 first, Vec3 second) {
        Vec3 a = first - candidate, b = second - candidate;
        float ar = std::hypot(a.x, a.y), br = std::hypot(b.x, b.y);
        if (ar < .5f || br < .5f || std::fabs(b.z) > .5f * br)
            return false;
        float cosine = (a.x * b.x + a.y * b.y) / (ar * br);

        return cosine >= std::cos(2 * std::atan(.8f)) &&
               view_line_open(map, candidate, second, .5f);
    }
    static bool view_line_open(const Navigator &map, Vec3 from, Vec3 to, float focus_padding) {
        float size = Navigator::resolution;
        Vec3 origin = map.map_cell_center(0) - Vec3{size * .5f, size * .5f, size * .5f};
        int samples = std::max(1, int(std::ceil(norm(to - from) / (size * .25f))));
        for (int step = 0; step <= samples; step++) {
            Vec3 world = from + (to - from) * (float(step) / samples);

            if (focus_padding > 0 && norm(world - to) < focus_padding)
                continue;
            Vec3 p = world - origin;
            int x = int(std::floor(p.x / size)), y = int(std::floor(p.y / size)),
                z = int(std::floor(p.z / size));
            if (x < 0 || x >= Navigator::nx || y < 0 || y >= Navigator::ny || z < 0 ||
                z >= Navigator::nz ||
                map.map_cell_state((z * Navigator::ny + y) * Navigator::nx + x) > 0)
                return false;
        }
        return true;
    }
};
} // namespace safe_nav
