// Checks the runtime input contracts and flies a short two-waypoint mission through RAPTOR.

#include "navigation/runtime.hpp"
#include "navigation/raptor_model.hpp"
#include <iostream>
#include <limits>
int main(int argc, char **argv) {
    safe_nav::NavigationRuntime runtime;
    runtime.reset({}, 7);
    safe_nav::DepthFrame frame;
    frame.capture_pose.rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    frame.ranges.fill(12);
    frame.capture_us = 100000;
    frame.sequence = 1;
    frame.frame_id = 7;
    frame.calibration_id = runtime.source_calibration_id;
    auto require = [](bool passed, const char *name) {
        if (!passed) {
            std::cerr << name << '\n';
            std::exit(1);
        }
    };
    safe_nav::NavigationRuntime invalid_launch;
    require(invalid_launch.reset({}, 0) == safe_nav::RuntimeStatus::invalid_input,
            "zero frame rejected");
    require(invalid_launch.reset({std::numeric_limits<float>::quiet_NaN(), 0, 0}, 7) ==
                safe_nav::RuntimeStatus::invalid_input,
            "nonfinite launch rejected before gridding");
    require(runtime.observe(frame, 100000) == safe_nav::RuntimeStatus::ready, "valid frame");
    safe_nav::GeometricWaypoint goal;
    goal.position = {2, 0, 0};
    goal.frame_id = 7;
    goal.received_us = 100000;
    goal.expires_us = 200000;
    goal.revision = 1;
    goal.sight_range_m = 12;
    goal.sight_sequence = frame.sequence;
    goal.sight_capture_us = frame.capture_us;
    require(runtime.accept(goal, frame.capture_pose, 100000) == safe_nav::RuntimeStatus::accepted,
            "visible target");
    goal.position = {-2, 0, 0};
    goal.revision = 2;
    require(runtime.accept(goal, frame.capture_pose, 100000) ==
                safe_nav::RuntimeStatus::goal_not_visible,
            "rear target rejected");
    require(!runtime.command_tick({}, 100000, 7, 100000).command_valid, "disarmed cannot command");
    frame.capture_us = 110000;
    frame.sequence = 2;
    frame.ranges.fill(.5f);
    require(runtime.observe(frame, 110000) == safe_nav::RuntimeStatus::ready, "foreground depth");
    goal.position = {2, 0, 0};
    goal.received_us = 110000;
    goal.sight_range_m = .5f;
    goal.sight_sequence = frame.sequence;
    goal.sight_capture_us = frame.capture_us;
    require(runtime.accept(goal, frame.capture_pose, 110000) ==
                safe_nav::RuntimeStatus::goal_not_visible,
            "occluded target rejected");
    frame.sequence = 3;
    frame.capture_us = 120000;
    frame.ranges[0] = std::numeric_limits<float>::quiet_NaN();
    require(runtime.observe(frame, 120000) == safe_nav::RuntimeStatus::invalid_input,
            "invalid frame rejected");
    if (argc >= 2) {
        RaptorWeights weights;
        require(raptor_load_weights(argv[1], &weights), "source controller load");
        auto dynamics = rl_physics_crazyflie_default();
        RLPhysicsState physics{};
        physics.orientation_wxyz[0] = 1;
        float thrust = dynamics.mass * 9.81f / 4, c0 = dynamics.rotor_thrust_coefficients[0],
              c1 = dynamics.rotor_thrust_coefficients[1],
              c2 = dynamics.rotor_thrust_coefficients[2];
        float hover = (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
        std::fill(physics.rpm, physics.rpm + 4, hover);
        safe_nav::RaptorControlState control;
        raptor_reset(weights, control.hidden);
        safe_nav::RaptorVehicleModel actual(weights, dynamics, physics, control),
            workspace = actual;
        safe_nav::NearSurfaceView attention;
        safe_nav::MeasuredRoute route;
        safe_nav::NavigationRuntime integrated(argc >= 4 ? &route : nullptr,
                                               argc >= 3 ? &attention : nullptr, argc >= 4);
        integrated.reset({}, 7);
        safe_nav::DepthFrame measured = frame;
        measured.ranges.fill(12);
        uint64_t now = 1000000;
        measured.sequence = 0;

        for (int elevation = -1; elevation <= 1; elevation++)
            for (int heading = 0; heading < 8; heading++) {
                float yaw = heading * 6.28318530718f / 8, pitch = elevation * 1.0471975512f,
                      cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch),
                      sp = std::sin(pitch);
                measured.capture_pose = {{},
                                         {cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0, cp}};
                measured.capture_us = ++now;
                measured.sequence++;
                require(integrated.observe(measured, now) == safe_nav::RuntimeStatus::ready,
                        "preflight fixture");
            }
        measured.capture_pose = actual.pose();
        measured.capture_us = ++now;
        measured.sequence++;
        require(integrated.observe(measured, now) == safe_nav::RuntimeStatus::ready,
                "current source view");
        safe_nav::HardwareRelease missing;
        require(integrated.arm(missing, actual, workspace, now) ==
                    safe_nav::RuntimeStatus::disarmed,
                "unverified hardware refuses arming");
        safe_nav::HardwareRelease source_fixture{1, 1, true, true, true, true, true};
        require(integrated.arm(source_fixture, actual, workspace, now) ==
                    safe_nav::RuntimeStatus::ready,
                "explicit source fixture handover");
        require(integrated.command_tick(actual.velocity(), now, 8, now).status ==
                    safe_nav::RuntimeStatus::wrong_frame,
                "wrong ego epoch disarms");
        require(integrated.arm(source_fixture, actual, workspace, now) ==
                    safe_nav::RuntimeStatus::ready,
                "source rearm after epoch rejection");
        safe_nav::GeometricWaypoint target;
        target.position = {2, 0, 0};
        target.frame_id = 7;
        target.received_us = now;
        target.expires_us = now + 30000000;
        target.revision = 1;
        target.sight_range_m = 12;
        target.sight_sequence = measured.sequence;
        target.sight_capture_us = measured.capture_us;
        require(integrated.accept(target, actual.pose(), now) == safe_nav::RuntimeStatus::accepted,
                "source goal acceptance");
        uint64_t initial = now;
        int physical_ticks = 0, phase = 0;
        bool arrived = false;
        for (int tick = 0; tick < 3000; tick++) {
            now = initial + tick * 10000;
            if (tick % 5 == 0) {
                if (tick) {
                    measured.capture_us = now;
                    measured.sequence++;
                    measured.capture_pose = actual.pose();
                    require(integrated.observe(measured, now) == safe_nav::RuntimeStatus::ready,
                            "fresh measured source frame");
                }
                if (phase == 1) {
                    auto next = target;
                    next.position = {2, 2, 0};
                    next.revision = 2;
                    next.received_us = now;
                    next.expires_us = now + 30000000;
                    next.sight_range_m = 12;
                    next.sight_sequence = measured.sequence;
                    next.sight_capture_us = measured.capture_us;
                    auto accepted = integrated.accept(next, actual.pose(), now);
                    if (accepted == safe_nav::RuntimeStatus::accepted)
                        phase = 2;
                    else
                        require(accepted == safe_nav::RuntimeStatus::goal_not_visible,
                                "pending goal remains unaccepted");
                }
                auto selected = integrated.plan(actual, workspace, now, 7, now);
                require(selected.command_valid, "source plan");
                if (selected.status == safe_nav::RuntimeStatus::arrived) {
                    if (phase == 0) {
                        require(integrated.request_view(1.57079632679f, 7, now, now + 20000000,
                                                        now) == safe_nav::RuntimeStatus::accepted,
                                "checked upstream view request");
                        phase = 1;
                    } else if (phase == 2) {
                        arrived = true;
                        break;
                    }
                }
            }
            auto command = integrated.command_tick(actual.velocity(), now, 7, now);
            require(command.command_valid, "100Hz command");
            require(actual.substep(command.command), "actual source motor/physics step");
            require(integrated.command_applied(), "applied command acknowledgement");
            ++physical_ticks;
        }
        require(arrived, "integrated source arrival");
        auto duplicate = integrated.plan(actual, workspace, now, 7, now);
        require(!duplicate.command_valid &&
                    duplicate.status == safe_nav::RuntimeStatus::invalid_input,
                "duplicate navigation tick rejected");
        require(!integrated.command_tick(actual.velocity(), now, 7, now).command_valid,
                "cadence fault disarms");
        require(!integrated.command_applied(), "unsolicited acknowledgement rejected");
        std::cout << "integrated_source_arrivals=2 actual_100Hz_steps=" << physical_ticks
                  << " cadence_rejection=1 hardware_verified=0\n";
    }
    std::cout << "runtime_boundary_checks=9 navigator_runtime_bytes=" << sizeof(runtime)
              << " optional_route_bytes=" << sizeof(safe_nav::MeasuredRoute)
              << " optional_attention_bytes=" << sizeof(safe_nav::NearSurfaceView)
              << " hardware_verified=0\n";
}
