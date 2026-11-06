#pragma once

// Minimal scalar-array port of rl-tools L2F quadrotor dynamics.
// Reference: rl-tools/rl-tools include/rl_tools/rl/environments/l2f/,
// RAPTOR-pinned upstream commit e43ae4bcda4556321a63f4eb5dcc826cd637aa39.
// State order follows L2F StateBase, then StateRotors: position, q[wxyz],
// world linear velocity, body angular velocity, and four normalized rotor speeds.

#include <cmath>
#include <cstddef>

struct RLPhysicsState {
    float position[3];
    float orientation_wxyz[4];
    float linear_velocity[3];
    float angular_velocity_body[3];
    float rpm[4];
};

struct RLPhysicsParams {
    float rotor_positions[12];             // [rotor][xyz], body frame
    float rotor_thrust_directions[12];      // [rotor][xyz], body frame
    float rotor_torque_directions[12];      // [rotor][xyz], body frame
    float rotor_thrust_coefficients[12];    // [rotor][constant, linear, quadratic]
    float rotor_torque_constants[4];
    float rotor_time_constants_rising[4];
    float rotor_time_constants_falling[4];
    float inertia[9];                       // row-major body-frame inertia
    float inertia_inverse[9];               // row-major
    float gravity_world[3];
    float mass;
    float action_min;
    float action_max;
    float dt;
    float position_limit;
    float velocity_limit;
    float angular_velocity_limit;
};

static_assert(sizeof(RLPhysicsState) == 17 * sizeof(float), "RLPhysicsState must stay tightly packed");
static_assert(sizeof(RLPhysicsParams) == 88 * sizeof(float), "RLPhysicsParams layout changed");

namespace rl_physics {
inline void cross(const float a[3], const float b[3], float out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// The L2F quaternion is scalar-first and rotates a body vector into world coordinates.
inline void rotate_body_to_world(const float q[4], const float v[3], float out[3]) {
    const float tx = 2.0f * (q[2] * v[2] - q[3] * v[1]);
    const float ty = 2.0f * (q[3] * v[0] - q[1] * v[2]);
    const float tz = 2.0f * (q[1] * v[1] - q[2] * v[0]);
    out[0] = v[0] + q[0] * tx + q[2] * tz - q[3] * ty;
    out[1] = v[1] + q[0] * ty + q[3] * tx - q[1] * tz;
    out[2] = v[2] + q[0] * tz + q[1] * ty - q[2] * tx;
}

inline void derivative(const RLPhysicsParams& p, const RLPhysicsState& s,
                       const float motor_setpoint[4], const float wind_force_world[3],
                       RLPhysicsState& d) {
    float thrust[3] = {0.0f, 0.0f, 0.0f};
    float torque[3] = {0.0f, 0.0f, 0.0f};
    for (int r = 0; r < 4; ++r) {
        const float speed = s.rpm[r];
        const float* c = &p.rotor_thrust_coefficients[3 * r];
        const float magnitude = c[0] + c[1] * speed + c[2] * speed * speed;
        float rotor_thrust[3];
        const float* direction = &p.rotor_thrust_directions[3 * r];
        for (int i = 0; i < 3; ++i) {
            rotor_thrust[i] = direction[i] * magnitude;
            thrust[i] += rotor_thrust[i];
            torque[i] += p.rotor_torque_directions[3 * r + i] * magnitude * p.rotor_torque_constants[r];
        }
        float arm_torque[3];
        cross(&p.rotor_positions[3 * r], rotor_thrust, arm_torque);
        for (int i = 0; i < 3; ++i) torque[i] += arm_torque[i];
    }

    for (int i = 0; i < 3; ++i) d.position[i] = s.linear_velocity[i];
    const float* q = s.orientation_wxyz;
    const float* w = s.angular_velocity_body;
    d.orientation_wxyz[0] = -q[1] * w[0] - q[2] * w[1] - q[3] * w[2];
    d.orientation_wxyz[1] =  q[0] * w[0] + q[2] * w[2] - q[3] * w[1];
    d.orientation_wxyz[2] =  q[0] * w[1] + q[3] * w[0] - q[1] * w[2];
    d.orientation_wxyz[3] =  q[0] * w[2] + q[1] * w[1] - q[2] * w[0];
    for (int i = 0; i < 4; ++i) d.orientation_wxyz[i] *= 0.5f;

    rotate_body_to_world(q, thrust, d.linear_velocity);
    for (int i = 0; i < 3; ++i)
        d.linear_velocity[i] = d.linear_velocity[i] / p.mass + p.gravity_world[i] + wind_force_world[i] / p.mass;

    float angular_momentum[3] = {0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            angular_momentum[row] += p.inertia[3 * row + col] * w[col];
    float gyroscopic[3];
    cross(w, angular_momentum, gyroscopic);
    for (int i = 0; i < 3; ++i) angular_momentum[i] = torque[i] - gyroscopic[i];
    for (int row = 0; row < 3; ++row) {
        d.angular_velocity_body[row] = 0.0f;
        for (int col = 0; col < 3; ++col)
            d.angular_velocity_body[row] += p.inertia_inverse[3 * row + col] * angular_momentum[col];
    }

    for (int r = 0; r < 4; ++r) {
        const float tau = motor_setpoint[r] >= s.rpm[r]
            ? p.rotor_time_constants_rising[r] : p.rotor_time_constants_falling[r];
        d.rpm[r] = (motor_setpoint[r] - s.rpm[r]) / tau;
    }
}

inline void add_scaled(const RLPhysicsState& s, const RLPhysicsState& d, float scale, RLPhysicsState& out) {
    for (int i = 0; i < 3; ++i) out.position[i] = s.position[i] + scale * d.position[i];
    for (int i = 0; i < 4; ++i) out.orientation_wxyz[i] = s.orientation_wxyz[i] + scale * d.orientation_wxyz[i];
    for (int i = 0; i < 3; ++i) out.linear_velocity[i] = s.linear_velocity[i] + scale * d.linear_velocity[i];
    for (int i = 0; i < 3; ++i) out.angular_velocity_body[i] = s.angular_velocity_body[i] + scale * d.angular_velocity_body[i];
    for (int i = 0; i < 4; ++i) out.rpm[i] = s.rpm[i] + scale * d.rpm[i];
}

inline void rk4(const RLPhysicsParams& p, const RLPhysicsState& s, const float setpoint[4],
                const float wind_force_world[3], RLPhysicsState& out) {
    RLPhysicsState k1{}, k2{}, k3{}, k4{}, temp{};
    derivative(p, s, setpoint, wind_force_world, k1);
    add_scaled(s, k1, 0.5f * p.dt, temp);
    derivative(p, temp, setpoint, wind_force_world, k2);
    add_scaled(s, k2, 0.5f * p.dt, temp);
    derivative(p, temp, setpoint, wind_force_world, k3);
    add_scaled(s, k3, p.dt, temp);
    derivative(p, temp, setpoint, wind_force_world, k4);

    for (int i = 0; i < 3; ++i)
        out.position[i] = s.position[i] + p.dt * (k1.position[i] + 2.0f*k2.position[i] + 2.0f*k3.position[i] + k4.position[i]) / 6.0f;
    for (int i = 0; i < 4; ++i)
        out.orientation_wxyz[i] = s.orientation_wxyz[i] + p.dt * (k1.orientation_wxyz[i] + 2.0f*k2.orientation_wxyz[i] + 2.0f*k3.orientation_wxyz[i] + k4.orientation_wxyz[i]) / 6.0f;
    for (int i = 0; i < 3; ++i)
        out.linear_velocity[i] = s.linear_velocity[i] + p.dt * (k1.linear_velocity[i] + 2.0f*k2.linear_velocity[i] + 2.0f*k3.linear_velocity[i] + k4.linear_velocity[i]) / 6.0f;
    for (int i = 0; i < 3; ++i)
        out.angular_velocity_body[i] = s.angular_velocity_body[i] + p.dt * (k1.angular_velocity_body[i] + 2.0f*k2.angular_velocity_body[i] + 2.0f*k3.angular_velocity_body[i] + k4.angular_velocity_body[i]) / 6.0f;
    for (int i = 0; i < 4; ++i)
        out.rpm[i] = s.rpm[i] + p.dt * (k1.rpm[i] + 2.0f*k2.rpm[i] + 2.0f*k3.rpm[i] + k4.rpm[i]) / 6.0f;
}
} // namespace rl_physics

// `action_normalized` is RAPTOR's [-1,1] motor action in front-right,
// back-right, back-left, front-left order. `wind_force_world` is force in N;
// it follows L2F StateRandomForce semantics and is divided by vehicle mass.
inline void rl_physics_step(const RLPhysicsState& state, const float action_normalized[4],
                            const float wind_force_world[3], const RLPhysicsParams& params,
                            RLPhysicsState& next_state) {
    float setpoint[4];
    const float half_range = (params.action_max - params.action_min) * 0.5f;
    for (int i = 0; i < 4; ++i) {
        float a = action_normalized[i];
        if (a < -1.0f) a = -1.0f;
        if (a >  1.0f) a =  1.0f;
        setpoint[i] = a * half_range + params.action_min + half_range;
    }
    rl_physics::rk4(params, state, setpoint, wind_force_world, next_state);

    const float norm = std::sqrt(next_state.orientation_wxyz[0]*next_state.orientation_wxyz[0] +
                                 next_state.orientation_wxyz[1]*next_state.orientation_wxyz[1] +
                                 next_state.orientation_wxyz[2]*next_state.orientation_wxyz[2] +
                                 next_state.orientation_wxyz[3]*next_state.orientation_wxyz[3]);
    for (float& q : next_state.orientation_wxyz) q /= norm;
    for (int i = 0; i < 3; ++i) {
        if (next_state.position[i] < -params.position_limit) next_state.position[i] = -params.position_limit;
        if (next_state.position[i] >  params.position_limit) next_state.position[i] =  params.position_limit;
        if (next_state.linear_velocity[i] < -params.velocity_limit) next_state.linear_velocity[i] = -params.velocity_limit;
        if (next_state.linear_velocity[i] >  params.velocity_limit) next_state.linear_velocity[i] =  params.velocity_limit;
        if (next_state.angular_velocity_body[i] < -params.angular_velocity_limit) next_state.angular_velocity_body[i] = -params.angular_velocity_limit;
        if (next_state.angular_velocity_body[i] >  params.angular_velocity_limit) next_state.angular_velocity_body[i] =  params.angular_velocity_limit;
    }
    for (int i = 0; i < 4; ++i) {
        if (next_state.rpm[i] < params.action_min) next_state.rpm[i] = params.action_min;
        if (next_state.rpm[i] > params.action_max) next_state.rpm[i] = params.action_max;
    }
}

// RLtools L2F DEFAULT_PARAMETERS_FACTORY uses the Crazyflie profile by default.
// Values and rotor order are transcribed from parameters/dynamics/crazyflie.h
// at the RAPTOR-pinned rl-tools commit e43ae4bcda4556321a63f4eb5dcc826cd637aa39.
inline RLPhysicsParams rl_physics_crazyflie_default() {
    RLPhysicsParams p{};
    const float positions[12] = {
         .028f,-.028f,0, -.028f,-.028f,0,
        -.028f, .028f,0,  .028f, .028f,0
    };
    const float torque_z[4] = {-1, 1, -1, 1};
    const float inertia_diag[3] = {9.416556729130406e-6f, 9.644051701582312e-6f, 1.745951732253285e-5f};
    const float inertia_inv_diag[3] = {106195.93007988465f, 103690.85846314249f, 57275.35197719487f};
    for (int r = 0; r < 4; ++r) {
        const int i = 3 * r;
        p.rotor_positions[i] = positions[i];
        p.rotor_positions[i + 1] = positions[i + 1];
        p.rotor_thrust_directions[i + 2] = 1.0f;
        p.rotor_torque_directions[i + 2] = torque_z[r];
        p.rotor_thrust_coefficients[i] = 0.00352526f;
        p.rotor_thrust_coefficients[i + 1] = 0.01437313f;
        p.rotor_thrust_coefficients[i + 2] = 0.09223048f;
        p.rotor_torque_constants[r] = 4.665e-3f;
        p.rotor_time_constants_rising[r] = 0.05545454545454546f;
        p.rotor_time_constants_falling[r] = 0.24939393939393945f;
    }
    for (int i = 0; i < 3; ++i) {
        p.inertia[4 * i] = inertia_diag[i];
        p.inertia_inverse[4 * i] = inertia_inv_diag[i];
    }
    p.gravity_world[2] = -9.81f;
    p.mass = 0.027f + 0.0017f + 0.0003f + 0.0016f;
    p.action_min = 0.0f;
    p.action_max = 1.0f;
    p.dt = 0.01f;
    p.position_limit = 100000.0f;
    p.velocity_limit = 100000.0f;
    p.angular_velocity_limit = 100000.0f;
    return p;
}

// Optional upstream L2F X500 simulator profile. It is a distinct simplified
// model from the measured `x500::real` registry profile.
inline RLPhysicsParams rl_physics_x500_sim() {
    RLPhysicsParams p{};
    const float positions[12] = {
         .176776695296636f,-.176776695296636f,0, -.176776695296636f, .176776695296636f,0,
         .176776695296636f, .176776695296636f,0, -.176776695296636f,-.176776695296636f,0
    };
    const float torque_z[4] = {-1, -1, 1, 1};
    for (int r = 0; r < 4; ++r) {
        const int i = 3*r;
        for (int j = 0; j < 3; ++j) p.rotor_positions[i+j] = positions[i+j];
        p.rotor_thrust_directions[i+2] = 1.0f;
        p.rotor_torque_directions[i+2] = torque_z[r];
        p.rotor_thrust_coefficients[i+2] = 8.74f;
        p.rotor_torque_constants[r] = 0.11697849233439939f;
        p.rotor_time_constants_rising[r] = 0.03f;
        p.rotor_time_constants_falling[r] = 0.03f;
    }
    p.inertia[0] = p.inertia[4] = 0.0216666666666666f;
    p.inertia[8] = 0.04f;
    p.inertia_inverse[0] = p.inertia_inverse[4] = 46.153846153846295f;
    p.inertia_inverse[8] = 25.0f;
    p.gravity_world[2] = -9.81f;
    p.mass = 2.0f;
    p.action_min = 0.0f;
    p.action_max = 1.0f;
    p.dt = 0.01f;
    p.position_limit = 100000.0f;
    p.velocity_limit = 100000.0f;
    p.angular_velocity_limit = 100000.0f;
    return p;
}
