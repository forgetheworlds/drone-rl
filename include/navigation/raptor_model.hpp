#pragma once

// Vehicle model used for planning: RAPTOR controller state plus L2F dynamics.

#include "navigator.hpp"
#include "surface_watch.hpp"
#include "../physics.hpp"
#include "../raptor.hpp"

namespace safe_nav {
struct RaptorControlState {
    float hidden[16]{};
    float previous_motor[4]{};
    Vec3 reference{};
    float reference_yaw = 0;
};

inline bool finite_motion_state(const RLPhysicsState &state) {
    for (float v : state.position)
        if (!std::isfinite(v))
            return false;
    float quaternion_norm2 = 0;
    for (float v : state.orientation_wxyz) {
        if (!std::isfinite(v))
            return false;
        quaternion_norm2 += v * v;
    }
    if (std::fabs(quaternion_norm2 - 1) > 1e-3f)
        return false;
    for (float v : state.linear_velocity)
        if (!std::isfinite(v))
            return false;
    for (float v : state.angular_velocity_body)
        if (!std::isfinite(v))
            return false;
    return true;
}

inline bool raptor_motor_command(const RaptorWeights &weights, RaptorControlState &control,
                                 const RLPhysicsState &measured, Command command, float dt,
                                 float motors[4]) {
    if (!finite_motion_state(measured) || !std::isfinite(dt) || std::fabs(dt - .01f) > 1e-6f ||
        !std::isfinite(command.world_velocity.x) || !std::isfinite(command.world_velocity.y) ||
        !std::isfinite(command.world_velocity.z) || !std::isfinite(command.yaw_rate))
        return false;
    for (float v : control.hidden)
        if (!std::isfinite(v))
            return false;
    for (float v : control.previous_motor)
        if (!std::isfinite(v) || v < -1 || v > 1)
            return false;
    if (!std::isfinite(control.reference.x) || !std::isfinite(control.reference.y) ||
        !std::isfinite(control.reference.z) || !std::isfinite(control.reference_yaw))
        return false;
    RaptorControlState next = control;
    next.reference = next.reference + command.world_velocity * dt;
    Vec3 position = {measured.position[0], measured.position[1], measured.position[2]};
    Vec3 error = next.reference - position;
    if (norm(error) > .25f)
        next.reference = position + unit(error) * .25f;
    next.reference_yaw += command.yaw_rate * dt;
    float target[3] = {next.reference.x, next.reference.y, next.reference.z};
    float orientation[4] = {std::cos(next.reference_yaw / 2), 0, 0,
                            std::sin(next.reference_yaw / 2)};
    float velocity[3] = {command.world_velocity.x, command.world_velocity.y,
                         command.world_velocity.z};
    float observation[22], output[4];
    raptor_pack_observation(measured.position, measured.orientation_wxyz, measured.linear_velocity,
                            measured.angular_velocity_body, target, orientation, velocity,
                            next.previous_motor, observation);
    raptor_forward(weights, observation, next.hidden, output);
    for (float v : output)
        if (!std::isfinite(v))
            return false;
    raptor_clip_action(output);
    for (float v : next.hidden)
        if (!std::isfinite(v))
            return false;
    std::copy(output, output + 4, next.previous_motor);
    control = next;
    std::copy(output, output + 4, motors);
    return true;
}

class RaptorVehicleModel {
  public:
    RLPhysicsState physics{};
    RaptorControlState control{};
    RaptorVehicleModel(const RaptorWeights &weights, const RLPhysicsParams &dynamics,
                       const RLPhysicsState &initial_physics,
                       const RaptorControlState &initial_control)
        : physics(initial_physics), control(initial_control), weights(&weights),
          dynamics(&dynamics) {}
    Pose pose() const {
        Pose result;
        result.position = {physics.position[0], physics.position[1], physics.position[2]};
        raptor_quaternion_matrix(physics.orientation_wxyz, result.rotation.data());
        return result;
    }
    Vec3 velocity() const {
        return {physics.linear_velocity[0], physics.linear_velocity[1], physics.linear_velocity[2]};
    }
    Vec3 reference_position() const { return control.reference; }
    const SurfaceWatch *moving_surface_guard = nullptr;
    bool forecast_clear(Vec3 point, float horizon_s) const {
        return !moving_surface_guard || moving_surface_guard->forecast_clear(point, horizon_s);
    }
    bool substep(Command command) {
        for (float v : physics.rpm)
            if (!std::isfinite(v) || v < dynamics->action_min || v > dynamics->action_max)
                return false;
        RaptorControlState next_control = control;
        float motors[4], wind[3] = {0, 0, 0};
        if (!raptor_motor_command(*weights, next_control, physics, command, dynamics->dt, motors))
            return false;
        RLPhysicsState next_physics{};
        rl_physics_step(physics, motors, wind, *dynamics, next_physics);
        if (!finite_motion_state(next_physics))
            return false;
        for (float v : next_physics.rpm)
            if (!std::isfinite(v))
                return false;
        physics = next_physics;
        control = next_control;
        return true;
    }

  private:
    const RaptorWeights *weights;
    const RLPhysicsParams *dynamics;
};
} // namespace safe_nav
