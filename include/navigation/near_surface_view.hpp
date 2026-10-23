#pragma once

// Camera yaw and speed limits that keep nearby surfaces and the direction of travel in view.

#include "navigator.hpp"

namespace safe_nav {

class NearSurfaceView {
  public:
    void reset() {
        active = false;
        range_count = 0;
        range_next = 0;
        have_nearest = false;
        change_priority_ticks = 0;
    }
    void use_range_barrier(bool enabled, bool limit_speed_only = false) {
        range_barrier = enabled;
        speed_only = limit_speed_only;
    }
    void use_change_priority(bool enabled) { change_priority = enabled; }

    void use_motion_coordination(bool enabled) { motion_coordination = enabled; }

    void set_motion_floor(float speed) { motion_floor = speed; }

    void set_forward_window(float radians) { forward_window = radians; }
    void observe(const Navigator &map, const Pose &pose, const std::array<float, 320> &depth) {
        const float reach = body_reaction_range;
        Vec3 nearest{}, associated{}, changed_point{};
        float changed_range = reach;
        bool changed_found = false;
        Vec3 map_origin =
            map.map_cell_center(0) - Vec3{Navigator::resolution * .5f, Navigator::resolution * .5f,
                                          Navigator::resolution * .5f};
        float nearest_range = reach, association = .5f;
        bool found = false, matched = false;
        for (int ray = 0; ray < 320; ray++) {
            float range = depth[ray];
            if (!std::isfinite(range) || range <= .01f || range >= 11.9f)
                continue;
            Vec3 point = pose.position + rotate(pose, unit({1, 1 - (ray % 20 + .5f) / 10,
                                                            .75f * (1 - (ray / 20 + .5f) / 8)})) *
                                             range;
            Vec3 delta = point - pose.position;
            if (std::fabs(delta.z) > Navigator::body_radius + Navigator::margin + .25f)
                continue;
            float horizontal = std::hypot(delta.x, delta.y);
            if (change_priority && horizontal < changed_range) {
                Vec3 cell = point +
                            rotate(pose, unit({1, 1 - (ray % 20 + .5f) / 10,
                                               .75f * (1 - (ray / 20 + .5f) / 8)})) *
                                .002f -
                            map_origin;
                int x = int(std::floor(cell.x / Navigator::resolution)),
                    y = int(std::floor(cell.y / Navigator::resolution)),
                    z = int(std::floor(cell.z / Navigator::resolution));
                if (x >= 0 && x < Navigator::nx && y >= 0 && y < Navigator::ny && z >= 0 &&
                    z < Navigator::nz &&
                    map.map_cell_state((z * Navigator::ny + y) * Navigator::nx + x) < 0) {
                    changed_range = horizontal;
                    changed_point = point;
                    changed_found = true;
                }
            }
            if (horizontal < nearest_range) {
                nearest_range = horizontal;
                nearest = point;
                found = true;
            }
            if (active) {
                float separation = norm(point - focus);
                if (separation < association) {
                    association = separation;
                    associated = point;
                    matched = true;
                }
            }
        }
        have_nearest = found;
        if (found) {
            nearest_point = nearest;
            latest_range = nearest_range;
            ranges[range_next] = nearest_range;
            range_next = (range_next + 1) % ranges.size();
            range_count = std::min(range_count + 1, int(ranges.size()));
        } else
            range_count = 0;
        if (active && matched)
            focus = associated;
        float retained_range = std::hypot(focus.x - pose.position.x, focus.y - pose.position.y);
        if (active && (retained_range > reach || std::fabs(focus.z - pose.position.z) > reach))
            active = false;
        if (change_priority_ticks > 0)
            --change_priority_ticks;
        if (changed_found) {
            focus = changed_point;
            active = true;
            change_priority_ticks = 16;
        }
        if (found && change_priority_ticks == 0 &&
            (!active || nearest_range + Navigator::resolution < retained_range)) {
            focus = nearest;
            active = true;
        }
    }
    Command constrain(const Pose &pose, Command desired) const {
        if (!active)
            return desired;
        float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
        float bearing =
            wrap(std::atan2(focus.y - pose.position.y, focus.x - pose.position.x) - yaw);

        float inner_angle = std::atan(.8f);
        float low = std::clamp((bearing - inner_angle) / .8f, -.5f, .5f);
        float high = std::clamp((bearing + inner_angle) / .8f, -.5f, .5f);
        desired.yaw_rate = std::clamp(desired.yaw_rate, low, high);
        if (motion_coordination) {

            Vec3 travel{desired.world_velocity.x, desired.world_velocity.y, 0};
            const float speed = norm(travel);
            if (speed > motion_floor) {
                const float yaw_now = std::atan2(pose.rotation[3], pose.rotation[0]);
                const float bearing = wrap(std::atan2(travel.y, travel.x) - yaw_now);
                if (std::fabs(bearing) > forward_window) {
                    const float want = bearing > 0 ? motion_turn_rate : -motion_turn_rate;
                    desired.yaw_rate = want > 0 ? std::max(desired.yaw_rate, want)
                                                : std::min(desired.yaw_rate, want);
                    desired.yaw_rate = std::clamp(desired.yaw_rate, -.5f, .5f);
                }
            }
        }
        if (range_barrier && have_nearest && range_count >= 4) {

            float middle = (range_count - 1) * .5f, numerator = 0, denominator = 0;
            int first = (range_next - range_count + int(ranges.size())) % ranges.size();
            for (int i = 0; i < range_count; i++) {
                float time = (i - middle) * .05f;
                numerator += time * ranges[(first + i) % ranges.size()];
                denominator += time * time;
            }
            float closing = std::clamp(-numerator / std::max(denominator, 1e-6f), 0.f,
                                       Navigator::max_speed + 1.5f);
            Vec3 away = pose.position - nearest_point;
            away.z = 0;
            away = unit(away);
            float requested_closing =
                -(desired.world_velocity.x * away.x + desired.world_velocity.y * away.y);
            float clearance = Navigator::body_radius + Navigator::margin + Navigator::resolution;
            float allowed = std::max(0.f, (latest_range - clearance) / .8f);
            float excess = std::max(closing, requested_closing) - allowed;
            if (excess > 0) {
                if (speed_only) {
                    float factor = std::clamp(
                        allowed / std::max(closing, std::max(requested_closing, 1e-6f)), 0.f, 1.f);
                    desired.world_velocity = desired.world_velocity * factor;
                } else {
                    desired.world_velocity = desired.world_velocity + away * excess;
                    float speed = norm(desired.world_velocity);
                    if (speed > Navigator::max_speed)
                        desired.world_velocity =
                            desired.world_velocity * (Navigator::max_speed / speed);
                }
            }
        }
        return desired;
    }
    bool has_focus() const { return active; }
    Vec3 focus_position() const { return focus; }

  private:
    static constexpr float body_reaction_range =
        Navigator::body_radius + Navigator::margin + (Navigator::max_speed + 1.5f) * .8f;
    std::array<float, 16> ranges{};
    int range_count = 0, range_next = 0;
    float latest_range = 0;
    Vec3 nearest_point{};
    bool have_nearest = false, range_barrier = false, change_priority = false, speed_only = false;
    int change_priority_ticks = 0;
    Vec3 focus{};
    bool active = false;

    static constexpr float camera_half_fov = .7598f;
    static constexpr float motion_turn_rate = .25f;
    float motion_floor = .05f;
    float forward_window = .7598f;
    bool motion_coordination = false;
};
} // namespace safe_nav
