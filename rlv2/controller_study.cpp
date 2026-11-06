// Flies RAPTOR (drone-rl's command interface) in open space with the true state and with the
// drifting state estimate from sensors.h, and reports how well it tracks velocity commands.
//
//   rlv2_controller_study RAPTOR_BIN [SEQUENCES]

#include "scenarios.hpp"
#include "sensors.h"
#include "navigation/raptor_model.hpp"
#include <cstdio>
#include <iostream>

struct Result {
    double true_error = 0, estimate_error = 0, rate_jitter = 0, motor_jitter = 0, tilt = 0;
    uint64_t samples = 0, faults = 0;
};

static Result fly(const RaptorWeights &weights, uint32_t sequences, float level, bool use_estimate, float stress,
                  float max_speed) {
    const auto dynamics = rl_physics_crazyflie_default();
    ParityPlan plan;
    plan.sequences = sequences;
    plan.ticks = 400;
    plan.hold_ticks = 20;
    plan.max_speed = max_speed;
    plan.max_climb = .5f;
    plan.max_yaw_rate = 1.0f;
    Result r;
    for (uint32_t sequence = 0; sequence < sequences; sequence++) {
        uint32_t rng = vx_seed(sequence, 99);
        VxSensorParams p = vx_sensor_params(rng, level);
        p.bias_walk *= stress;
        p.velocity_noise *= stress;
        RLPhysicsState physics{};
        physics.orientation_wxyz[0] = 1;
        physics.position[2] = 2;
        float thrust = dynamics.mass * 9.81f / 4, c0 = dynamics.rotor_thrust_coefficients[0],
              c1 = dynamics.rotor_thrust_coefficients[1], c2 = dynamics.rotor_thrust_coefficients[2];
        std::fill(physics.rpm, physics.rpm + 4, (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2));
        safe_nav::RaptorControlState control;
        control.reference = {0, 0, 2};
        raptor_reset(weights, control.hidden);
        VxEstimate estimate;
        vx_estimate_reset(estimate, physics.position);
        const auto commands = parity_commands(plan, sequence);
        float previous_motor[4] = {0, 0, 0, 0};
        for (uint32_t tick = 0; tick < plan.ticks; tick++) {
            if (tick % p.depth_period == 0)
                vx_estimate_correct(estimate, p);
            safe_nav::Command command;
            command.world_velocity = {commands[tick * 4], commands[tick * 4 + 1], commands[tick * 4 + 2]};
            command.yaw_rate = commands[tick * 4 + 3];
            for (int k = 0; k < 5; k++) {
                vx_estimate_step(estimate, physics.linear_velocity, p, .01f, rng);
                RLPhysicsState measured = physics;
                if (use_estimate) {
                    for (int j = 0; j < 3; j++) {
                        measured.position[j] = estimate.position[j];
                        measured.linear_velocity[j] = estimate.velocity[j];
                        measured.angular_velocity_body[j] += p.gyro_noise * stress * vx_normal(rng);
                    }
                }
                float motors[4];
                if (!safe_nav::raptor_motor_command(weights, control, measured, command, .01f, motors)) {
                    r.faults++;
                    break;
                }
                RLPhysicsState next{};
                const float wind[3] = {0, 0, 0};
                rl_physics_step(physics, motors, wind, dynamics, next);
                physics = next;
                const bool settled = tick % plan.hold_ticks >= 10 && tick >= 20;
                if (settled) {
                    double e2 = 0, s2 = 0;
                    for (int j = 0; j < 3; j++) {
                        const float c = j == 0 ? command.world_velocity.x : j == 1 ? command.world_velocity.y : command.world_velocity.z;
                        e2 += (physics.linear_velocity[j] - c) * (physics.linear_velocity[j] - c);
                        s2 += (estimate.velocity[j] - c) * (estimate.velocity[j] - c);
                    }
                    r.true_error += e2;
                    r.estimate_error += s2;
                    r.rate_jitter += physics.angular_velocity_body[0] * physics.angular_velocity_body[0] +
                                     physics.angular_velocity_body[1] * physics.angular_velocity_body[1];
                    double m2 = 0;
                    for (int j = 0; j < 4; j++)
                        m2 += (motors[j] - previous_motor[j]) * (motors[j] - previous_motor[j]);
                    r.motor_jitter += m2 / 4;
                    const float *q = physics.orientation_wxyz;
                    const double up = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
                    r.tilt = std::max(r.tilt, std::acos(std::max(-1.0, std::min(1.0, up))) * 57.2958);
                    r.samples++;
                }
                std::copy(motors, motors + 4, previous_motor);
            }
        }
    }
    const double n = std::max<uint64_t>(r.samples, 1);
    r.true_error = std::sqrt(r.true_error / n);
    r.estimate_error = std::sqrt(r.estimate_error / n);
    r.rate_jitter = std::sqrt(r.rate_jitter / n);
    r.motor_jitter = std::sqrt(r.motor_jitter / n);
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "rlv2_controller_study RAPTOR_BIN [SEQUENCES]\n";
        return 1;
    }
    RaptorWeights weights;
    if (!raptor_load_weights(argv[1], &weights)) {
        std::cerr << "Invalid RAPTOR weights\n";
        return 1;
    }
    const uint32_t sequences = argc > 2 ? std::stoul(argv[2]) : 128;
    std::printf("max_speed_mps,case,true_velocity_rms_mps,estimate_velocity_rms_mps,roll_pitch_rate_rms_radps,motor_step_rms,max_tilt_deg,faults\n");
    struct Case { const char *name; float level; bool estimate; float stress; };
    for (float speed : {.5f, 1.0f, 1.5f, 2.0f})
    for (Case c : {Case{"true state", 0, false, 1}, Case{"estimate, mild", .5f, true, 1},
                   Case{"estimate, full", 1, true, 1}, Case{"estimate, 2x noise", 1, true, 2},
                   Case{"estimate, 4x noise", 1, true, 4}}) {
        Result r = fly(weights, sequences, c.level, c.estimate, c.stress, speed);
        std::printf("%.1f,%s,%.3f,%.3f,%.3f,%.4f,%.1f,%llu\n", speed, c.name, r.true_error, r.estimate_error, r.rate_jitter,
                    r.motor_jitter, r.tilt, (unsigned long long)r.faults);
    }
}
