#pragma once

// Deployment-facing wrapper: validates depth frames and waypoints, arms the navigator and
// produces timed commands.

#include "observed_route_navigator.hpp"

namespace safe_nav {

enum class RuntimeStatus {
    disarmed,
    ready,
    accepted,
    arrived,
    invalid_input,
    stale_input,
    wrong_frame,
    goal_not_visible,
    visibility_unverified,
    expired_goal,
    requires_supervisor
};
struct DepthFrame {
    Pose capture_pose{};
    std::array<float, 320> ranges{};
    uint64_t capture_us = 0;
    uint32_t sequence = 0, frame_id = 0;
    uint32_t calibration_id = 0;
};
struct GeometricWaypoint {
    Vec3 position{};
    uint64_t received_us = 0, expires_us = 0;
    uint32_t revision = 0, frame_id = 0;
    float uncertainty_m = 0;

    float sight_range_m = 0;
    uint32_t sight_sequence = 0;
    uint64_t sight_capture_us = 0;
};
struct HardwareRelease {
    uint32_t vehicle_profile_id = 0, controller_id = 0;
    bool memory_verified = false, timing_verified = false, motor_mapping_verified = false;
    bool identified_dynamics = false, takeover_verified = false;
    bool verified() const {
        return vehicle_profile_id && controller_id && memory_verified && timing_verified &&
               motor_mapping_verified && identified_dynamics && takeover_verified;
    }
};
struct RuntimeDecision {
    RuntimeStatus status = RuntimeStatus::disarmed;
    Command command{};
    bool command_valid = false;
};
class NavigationRuntime {
  public:
    static constexpr uint64_t depth_max_age_us = 75000, ego_max_age_us = 20000;
    static constexpr uint32_t source_calibration_id = 0x20160001;
    explicit NavigationRuntime(MeasuredRoute *route = nullptr, NearSurfaceView *view = nullptr,
                               bool resolve_arrival_support = false)
        : navigator(true, true), attention(view) {
        configuration_valid = !resolve_arrival_support || (route && view);
        if (route && resolve_arrival_support)
            route->use_continuous_arrival(true);
        navigator.use_measured_route(route);
        navigator.use_near_surface_view(view);
        navigator.use_joint_arrival_view(resolve_arrival_support);
    }
    RuntimeStatus reset(Vec3 launch, uint32_t frame) {
        disarm();
        have_depth = false;
        have_goal = false;
        frame_id = 0;
        last_motor_us = 0;
        last_plan_us = 0;
        view_pending = false;
        arrival_ticks = 0;
        if (!frame || !finite(launch))
            return RuntimeStatus::invalid_input;
        navigator.reset(launch);
        frame_id = frame;
        return RuntimeStatus::disarmed;
    }
    RuntimeStatus observe(const DepthFrame &next, uint64_t now) {
        if (!frame_id || next.frame_id != frame_id)
            return fault(RuntimeStatus::wrong_frame);
        if (next.calibration_id != source_calibration_id || !valid_pose(next.capture_pose) ||
            next.capture_us > now || now - next.capture_us > depth_max_age_us ||
            (have_depth &&
             (next.capture_us <= depth.capture_us || next.sequence <= depth.sequence)))
            return fault(RuntimeStatus::invalid_input);
        for (float r : next.ranges)
            if (!std::isfinite(r) || r <= .01f || r > 12)
                return fault(RuntimeStatus::invalid_input);
        depth = next;
        have_depth = true;
        navigator.observe(next.capture_pose, next.ranges);
        return RuntimeStatus::ready;
    }
    RuntimeStatus accept(const GeometricWaypoint &next, const Pose &current, uint64_t now) {
        if (!frame_id || next.frame_id != frame_id)
            return RuntimeStatus::wrong_frame;
        if (!finite(next.position) || !valid_pose(current) || !std::isfinite(next.uncertainty_m) ||
            next.uncertainty_m < 0 || next.uncertainty_m > .1f || next.received_us > now ||
            now - next.received_us > 100000 || next.expires_us <= now ||
            (have_goal && next.revision <= goal.revision))
            return RuntimeStatus::invalid_input;
        if (!fresh_depth(now))
            return RuntimeStatus::stale_input;

        if (norm(current.position - depth.capture_pose.position) > .03f)
            return RuntimeStatus::stale_input;
        for (int i = 0; i < 9; i++)
            if (std::fabs(current.rotation[i] - depth.capture_pose.rotation[i]) > .03f)
                return RuntimeStatus::stale_input;
        Vec3 b = inverse_rotate(depth.capture_pose, next.position - depth.capture_pose.position);
        float distance = norm(b);
        if (b.x <= 0 || std::fabs(b.y) >= b.x || std::fabs(b.z) >= .75f * b.x || distance <= .25f ||
            distance >= 11.9f)
            return RuntimeStatus::goal_not_visible;
        if (next.sight_sequence != depth.sequence || next.sight_capture_us != depth.capture_us ||
            !std::isfinite(next.sight_range_m) || next.sight_range_m <= .01f ||
            next.sight_range_m > 12)
            return RuntimeStatus::visibility_unverified;

        if (distance + .05f > next.sight_range_m)
            return RuntimeStatus::goal_not_visible;
        goal = next;
        have_goal = true;
        view_pending = false;
        arrival_ticks = 0;
        navigator.next_goal();
        return RuntimeStatus::accepted;
    }
    RuntimeStatus request_view(float heading, uint32_t frame, uint64_t received, uint64_t expires,
                               uint64_t now) {
        if (!frame_id || frame != frame_id)
            return RuntimeStatus::wrong_frame;
        if (!std::isfinite(heading) || received > now || now - received > 100000 ||
            expires <= now || expires - now > 20000000)
            return RuntimeStatus::invalid_input;
        if (!armed)
            return RuntimeStatus::disarmed;
        view_heading = wrap(heading);
        view_expiry = expires;
        view_pending = true;
        arrival_ticks = 0;
        return RuntimeStatus::accepted;
    }
    template <class Model>
    RuntimeStatus arm(const HardwareRelease &release, const Model &measured, Model &workspace,
                      uint64_t now) {
        if (!configuration_valid || !release.verified())
            return RuntimeStatus::disarmed;
        if (!fresh_depth(now))
            return RuntimeStatus::stale_input;
        auto ready = navigator.check_handover(measured, workspace);
        armed = ready.verified;
        if (armed && attention) {
            attention->reset();
            attention->observe(navigator.map(), depth.capture_pose, depth.ranges);
        }
        last_motor_us = 0;
        last_plan_us = 0;
        return armed ? RuntimeStatus::ready : RuntimeStatus::requires_supervisor;
    }
    template <class Model>
    RuntimeDecision plan(const Model &measured, Model &workspace, uint64_t ego_us,
                         uint32_t ego_frame, uint64_t now) {
        RuntimeDecision d;
        if (!armed)
            return d;
        if (ego_frame != frame_id) {
            d.status = fault(RuntimeStatus::wrong_frame);
            return d;
        }
        if (command_pending ||
            (last_plan_us && (now <= last_plan_us || now - last_plan_us < 45000))) {
            d.status = fault(RuntimeStatus::invalid_input);
            return d;
        }
        if (ego_us > now || now - ego_us > ego_max_age_us || !fresh_depth(now)) {
            d.status = fault(RuntimeStatus::stale_input);
            return d;
        }
        if (!have_goal && !view_pending) {
            d.status = RuntimeStatus::ready;
            return d;
        }
        if (now >= (view_pending ? view_expiry : goal.expires_us)) {
            d.status = fault(RuntimeStatus::expired_goal);
            return d;
        }
        if (!valid_pose(measured.pose()) || !finite(measured.velocity())) {
            d.status = fault(RuntimeStatus::invalid_input);
            return d;
        }
        last_plan_us = now;
        BackupDecision selected;
        if (view_pending) {
            Command look;
            auto pose = measured.pose();
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            look.yaw_rate = std::clamp(2 * wrap(view_heading - yaw), -.5f, .5f);
            selected = navigator.inspect(measured, look, workspace);
        } else
            selected = navigator.plan(measured, goal.position, workspace);
        if (!selected.verified) {
            d.status = fault(RuntimeStatus::requires_supervisor);
            return d;
        }
        bool arrived = !view_pending && norm(measured.pose().position - goal.position) < .25f &&
                       norm(measured.velocity()) < .2f;
        arrival_ticks = arrived ? arrival_ticks + 1 : 0;
        d.status = arrival_ticks >= 6 ? RuntimeStatus::arrived : RuntimeStatus::accepted;
        d.command = selected.command;
        d.command_valid = true;
        return d;
    }
    RuntimeDecision command_tick(Vec3 velocity, uint64_t ego_us, uint32_t ego_frame, uint64_t now) {
        RuntimeDecision d;
        if (!armed)
            return d;
        if (ego_frame != frame_id) {
            d.status = fault(RuntimeStatus::wrong_frame);
            return d;
        }
        if (command_pending) {
            d.status = fault(RuntimeStatus::invalid_input);
            return d;
        }
        if (!finite(velocity) || ego_us > now || now - ego_us > ego_max_age_us ||
            !fresh_depth(now) ||
            (last_motor_us && (now <= last_motor_us ||
                               (now - last_motor_us > 15000 || now - last_motor_us < 5000)))) {
            d.status = fault(RuntimeStatus::stale_input);
            return d;
        }
        if ((!have_goal && !view_pending) ||
            now >= (view_pending ? view_expiry : goal.expires_us)) {
            d.status = fault(RuntimeStatus::expired_goal);
            return d;
        }
        if (!navigator.command_for_next_tick(d.command, velocity)) {
            d.status = fault(RuntimeStatus::requires_supervisor);
            return d;
        }
        last_motor_us = now;
        command_pending = true;
        d.command_valid = true;
        d.status = RuntimeStatus::accepted;
        return d;
    }

    bool command_applied() {
        if (!command_pending) {
            disarm();
            return false;
        }
        command_pending = false;
        navigator.advance_executed_tick();
        return true;
    }
    void disarm() {
        armed = false;
        command_pending = false;
        navigator.invalidate();
    }

  private:
    ObservedRouteNavigator navigator;
    NearSurfaceView *attention = nullptr;
    DepthFrame depth;
    GeometricWaypoint goal;
    uint32_t frame_id = 0;
    uint64_t last_motor_us = 0, last_plan_us = 0, view_expiry = 0;
    float view_heading = 0;
    int arrival_ticks = 0;
    bool configuration_valid = true, armed = false, have_depth = false, have_goal = false,
         command_pending = false, view_pending = false;
    bool fresh_depth(uint64_t now) const {
        return have_depth && now >= depth.capture_us && now - depth.capture_us <= depth_max_age_us;
    }
    RuntimeStatus fault(RuntimeStatus status) {
        disarm();
        return status;
    }
    static bool finite(Vec3 v) {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    }
    static bool valid_pose(const Pose &p) {
        if (!finite(p.position))
            return false;
        for (float v : p.rotation)
            if (!std::isfinite(v))
                return false;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                float dot = 0;
                for (int k = 0; k < 3; k++)
                    dot += p.rotation[k * 3 + i] * p.rotation[k * 3 + j];
                if (std::fabs(dot - (i == j ? 1.f : 0.f)) > .001f)
                    return false;
            }
        auto &r = p.rotation;
        float determinant = r[0] * (r[4] * r[8] - r[5] * r[7]) -
                            r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
        return std::fabs(determinant - 1) < .001f;
    }
};
} // namespace safe_nav
