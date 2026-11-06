#pragma once

// Detects surfaces that change between depth frames (moving obstacles), keeps them in view
// and forecasts whether a route segment stays clear of them.

#include "navigator.hpp"

namespace safe_nav {

class SurfaceWatch {
  public:
    uint32_t changed_frames = 0, view_overrides = 0;

    void require_measured_motion(bool enabled) { confirm_motion = enabled; }
    Vec3 diagnostic_center(int i) const { return tracks[i].center; }
    bool diagnostic_active(int i) const { return usable(tracks[i]); }
    float diagnostic_age(int i) const { return (tick - tracks[i].seen) * .05f; }
    Vec3 diagnostic_velocity(int i) const { return tracks[i].velocity; }
    void reset() {
        tracks = {};
        history_count = 0;
        history_next = 0;
        tick = 0;
        changed_frames = 0;
        view_overrides = 0;
    }
    void observe(const Navigator &map, const Pose &pose, const std::array<float, 320> &depth,
                 std::array<uint8_t, 40> *changed_output = nullptr,
                 const std::array<float, 320> *tile_minimum = nullptr) {
        if (changed_output)
            changed_output->fill(0);
        ++tick;
        Vec3 points[320];
        bool hit[320]{};
        Vec3 novel_sum{};
        int novel = 0;
        Vec3 origin = map.map_cell_center(0) - Vec3{.125f, .125f, .125f};
        for (int k = 0; k < 320; k++) {
            float r = depth[k];
            if (!std::isfinite(r) || r <= .01f || r >= 11.9f)
                continue;
            points[k] =
                pose.position +
                rotate(pose, unit({1, 1 - (k % 20 + .5f) / 10, .75f * (1 - (k / 20 + .5f) / 8)})) *
                    (r + .002f);
            hit[k] = true;
            int x = int(std::floor((points[k].x - origin.x) / .25f)),
                y = int(std::floor((points[k].y - origin.y) / .25f)),
                z = int(std::floor((points[k].z - origin.z) / .25f));
            bool changed = x >= 0 && x < 24 && y >= 0 && y < 24 && z >= 0 && z < 12 &&
                           map.map_cell_state((z * 24 + y) * 24 + x) < 0;
            if (!changed)
                for (int old = 0; old < history_count; old++)
                    if (was_free(history[old], points[k])) {
                        changed = true;
                        break;
                    }
            if (changed) {
                if (changed_output)
                    (*changed_output)[k / 8] |= uint8_t(1u << (k % 8));
                novel_sum = novel_sum + points[k];
                ++novel;
            }
        }
        if (novel) {
            ++changed_frames;
            Vec3 center = novel_sum * (1.f / novel);
            int best = -1;
            float rank = 1e20f;
            for (int i = 0; i < 4; i++)
                if (tracks[i].active) {
                    float d =
                        std::hypot(center.x - tracks[i].center.x, center.y - tracks[i].center.y);
                    if (d < rank && d < 1.0f) {
                        rank = d;
                        best = i;
                    }
                }
            if (best < 0)
                for (int i = 0; i < 4; i++)
                    if (!tracks[i].active) {
                        best = i;
                        break;
                    }
            if (best < 0) {
                best = 0;
                for (int i = 1; i < 4; i++)
                    if (tracks[i].seen < tracks[best].seen)
                        best = i;
            }
            auto &track = tracks[best];
            if (!track.active) {
                track.center = center;
                track.reference = center;
                track.velocity = {};
                track.seen = tick;
                track.reference_tick = tick;
                track.active = true;
                track.motion_reference = center;
                track.motion_tick = tick;
            } else {
                float dt = (tick - track.motion_tick) * .05f;
                float displacement = std::hypot(center.x - track.motion_reference.x,
                                                center.y - track.motion_reference.y);
                if (dt >= .2f && displacement > Navigator::resolution * .25f)
                    track.confirmed = true;
                update(track, center);
            }
        }

        for (auto &track : tracks)
            if (usable(track)) {
                Vec3 sum{};
                int count = 0;
                for (int k = 0; k < 320; k++)
                    if (hit[k] &&
                        std::hypot(points[k].x - track.center.x, points[k].y - track.center.y) <
                            .45f &&
                        std::fabs(points[k].z - track.center.z) < .25f) {
                        sum = sum + points[k];
                        ++count;
                    }
                if (count >= 2) {
                    Vec3 center = sum * (1.f / count);
                    center.z = track.center.z;
                    update(track, center);
                }
            }
        if (tile_minimum && (history_count == 0 || tick % 20 == 0)) {
            history[history_next] = {pose, *tile_minimum};
            history_next = (history_next + 1) % 3;
            history_count = std::min(history_count + 1, 3);
        }
    }
    bool needs_depth_keyframe() const { return history_count == 0 || (tick + 1) % 20 == 0; }
    Command keep_changed_surface_visible(const Pose &pose, Command route) {
        float best = 1e20f;
        Track *selected = nullptr;
        for (auto &track : tracks)
            if (usable(track)) {
                float age = (tick - track.seen) * .05f;
                Vec3 d = track.center - pose.position;
                float distance = std::hypot(d.x, d.y);

                float reach = Navigator::body_radius + Navigator::margin +
                              (Navigator::max_speed + 1.5f) * .8f + 1.5f * std::min(age, 1.f);
                if (distance < reach && std::fabs(d.z) < 1.5f && distance < best) {
                    best = distance;
                    selected = &track;
                }
            }
        if (selected) {
            Vec3 d = selected->center - pose.position;
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            route.yaw_rate = std::clamp(2 * wrap(std::atan2(d.y, d.x) - yaw), -.5f, .5f);
            ++view_overrides;
        }
        return route;
    }
    bool stale_focus(Vec3 position, Vec3 &focus) const {
        float nearest = 4.f;
        bool found = false;
        for (const auto &track : tracks)
            if (usable(track) && (tick - track.seen) * .05f > .4f) {
                float distance = norm(track.center - position);
                if (distance < nearest) {
                    nearest = distance;
                    focus = track.center;
                    found = true;
                }
            }
        return found;
    }
    Vec3 away_direction(Vec3 position) const {
        float nearest = 1e20f;
        Vec3 away{};
        for (const auto &track : tracks)
            if (usable(track)) {
                Vec3 d = position - track.center;
                d.z = 0;
                float distance = norm(d);
                if (distance < nearest) {
                    nearest = distance;
                    away = unit(d);
                }
            }
        return away;
    }
    bool separating(Vec3 position, Vec3 velocity, float future_s) const {
        bool relevant = false;
        for (const auto &track : tracks)
            if (usable(track)) {
                float total = (tick - track.seen) * .05f + future_s;
                Vec3 d = position - track.center;
                d.z = 0;
                if (norm(d) > 4)
                    continue;
                relevant = true;
                Vec3 relative = velocity;
                if (d.x * relative.x + d.y * relative.y <= 0)
                    return false;
            }
        return relevant;
    }
    bool segment_forecast_clear(Vec3 from, Vec3 to, float future_s) const {
        for (const auto &track : tracks)
            if (usable(track)) {
                float total = (tick - track.seen) * .05f + future_s;
                Vec3 prediction = track.center;
                float dx = to.x - from.x, dy = to.y - from.y, length2 = dx * dx + dy * dy;
                float fraction =
                    length2 > 1e-10f
                        ? std::clamp(((prediction.x - from.x) * dx + (prediction.y - from.y) * dy) /
                                         length2,
                                     0.f, 1.f)
                        : 0;
                float radius = Navigator::body_radius + Navigator::margin + .5f + 1.5f * total;
                if (std::hypot(from.x + fraction * dx - prediction.x,
                               from.y + fraction * dy - prediction.y) <= radius)
                    return false;
            }
        return true;
    }
    bool forecast_clear(Vec3 point, float future_s) const {
        for (const auto &track : tracks)
            if (usable(track)) {
                float age = (tick - track.seen) * .05f, total = age + future_s;
                Vec3 prediction = track.center;

                float radius = Navigator::body_radius + Navigator::margin + .5f + 1.5f * total;
                if (std::hypot(point.x - prediction.x, point.y - prediction.y) <= radius)
                    return false;
            }
        return true;
    }

  private:
    struct DepthKeyframe {
        Pose pose;
        std::array<float, 320> ranges;
    };
    std::array<DepthKeyframe, 3> history{};
    int history_count = 0, history_next = 0;
    static bool was_free(const DepthKeyframe &frame, Vec3 point) {
        Vec3 body = inverse_rotate(frame.pose, point - frame.pose.position);
        float distance = norm(body);
        if (body.x <= .01f || distance >= 11.9f)
            return false;
        int col = int(std::round((1 - body.y / body.x) * 10 - .5f)),
            row = int(std::round((1 - body.z / (.75f * body.x)) * 8 - .5f));
        if (col < 1 || col > 18 || row < 1 || row > 14)
            return false;
        float nearest = 12;
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++) {
                float r = frame.ranges[(row + y) * 20 + col + x];
                if (!std::isfinite(r) || r <= .01f || r > 12)
                    return false;
                nearest = std::min(nearest, r);
            }
        return distance + .24f < nearest;
    }
    struct Track {
        Vec3 center{}, reference{}, velocity{}, motion_reference{};
        uint32_t seen = 0, reference_tick = 0, motion_tick = 0;
        bool active = false, confirmed = false;
    };
    bool usable(const Track &track) const {
        return track.active && (!confirm_motion || track.confirmed);
    }
    void update(Track &track, Vec3 center) {
        float dt = (tick - track.reference_tick) * .05f;
        if (dt >= .25f) {
            Vec3 velocity = (center - track.reference) * (1.f / dt);
            velocity.z = 0;
            float speed = norm(velocity);
            if (speed > 1.5f)
                velocity = velocity * (1.5f / speed);
            track.velocity = velocity;
            track.reference = center;
            track.reference_tick = tick;
        }
        track.center = center;
        track.seen = tick;
    }
    std::array<Track, 4> tracks{};
    uint32_t tick = 0;
    bool confirm_motion = false;
};
} // namespace safe_nav
