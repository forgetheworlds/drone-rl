#pragma once

// Simulated vehicle: RAPTOR controller driving L2F physics at 100 Hz.

#include "navigator.hpp"
#include "../physics.hpp"
#include "../raptor.hpp"
#include <stdexcept>

namespace safe_nav {

class Vehicle {
  public:
    RLPhysicsState state{};
    RLPhysicsParams dynamics = rl_physics_crazyflie_default();
    Vehicle(const char *weights, Vec3 start, float yaw = 0) {
        if (!raptor_load_weights(weights, &controller))
            throw std::runtime_error("RAPTOR weights rejected");
        state.position[0] = start.x;
        state.position[1] = start.y;
        state.position[2] = start.z;
        state.orientation_wxyz[0] = std::cos(yaw / 2);
        state.orientation_wxyz[3] = std::sin(yaw / 2);
        float thrust = dynamics.mass * 9.81f / 4;
        float c0 = dynamics.rotor_thrust_coefficients[0],
              c1 = dynamics.rotor_thrust_coefficients[1],
              c2 = dynamics.rotor_thrust_coefficients[2];
        float hover = (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
        std::fill(state.rpm, state.rpm + 4, hover);
        reference = start;
        reference_yaw = yaw;
        raptor_reset(controller, hidden);
    }
    Pose pose() const {
        Pose p;
        p.position = {state.position[0], state.position[1], state.position[2]};
        raptor_quaternion_matrix(state.orientation_wxyz, p.rotation.data());
        return p;
    }
    Vec3 velocity() const {
        return {state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2]};
    }
    Vec3 reference_position() const { return reference; }
    void substep(Command command) {
        reference = reference + command.world_velocity * dynamics.dt;
        Vec3 error = reference - pose().position;
        if (norm(error) > .25f)
            reference = pose().position + unit(error) * .25f;
        reference_yaw += command.yaw_rate * dynamics.dt;
        float position[3] = {reference.x, reference.y, reference.z};
        float orientation[4] = {std::cos(reference_yaw / 2), 0, 0, std::sin(reference_yaw / 2)};
        float velocity[3] = {command.world_velocity.x, command.world_velocity.y,
                             command.world_velocity.z};
        float observation[22], motors[4], wind[3] = {0, 0, 0};
        raptor_pack_observation(state.position, state.orientation_wxyz, state.linear_velocity,
                                state.angular_velocity_body, position, orientation, velocity,
                                previous_motor, observation);
        raptor_forward(controller, observation, hidden, motors);
        raptor_clip_action(motors);
        RLPhysicsState next{};
        rl_physics_step(state, motors, wind, dynamics, next);
        state = next;
        std::copy(motors, motors + 4, previous_motor);
        for (float v : state.position)
            if (!std::isfinite(v))
                throw std::runtime_error("Nonfinite physics state");
    }

  private:
    RaptorWeights controller{};
    float hidden[16]{}, previous_motor[4]{};
    Vec3 reference{};
    float reference_yaw = 0;
};
} // namespace safe_nav
