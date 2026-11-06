#include <metal_stdlib>
using namespace metal;

// Keep field order and scalar-array representation identical to physics.hpp.
// Reference: RAPTOR-pinned rl-tools L2F source at e43ae4bcda4556321a63f4eb5dcc826cd637aa39.
struct RLPhysicsState {
    float position[3];
    float orientation_wxyz[4];
    float linear_velocity[3];
    float angular_velocity_body[3];
    float rpm[4];
};

struct RLPhysicsParams {
    float rotor_positions[12];
    float rotor_thrust_directions[12];
    float rotor_torque_directions[12];
    float rotor_thrust_coefficients[12];
    float rotor_torque_constants[4];
    float rotor_time_constants_rising[4];
    float rotor_time_constants_falling[4];
    float inertia[9];
    float inertia_inverse[9];
    float gravity_world[3];
    float mass;
    float action_min;
    float action_max;
    float dt;
    float position_limit;
    float velocity_limit;
    float angular_velocity_limit;
};

inline void rl_physics_cross(thread const float* a, thread const float* b, thread float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

inline void rl_physics_rotate_body_to_world(thread const float* q, thread const float* v, thread float* out) {
    const float tx = 2.0f * (q[2]*v[2] - q[3]*v[1]);
    const float ty = 2.0f * (q[3]*v[0] - q[1]*v[2]);
    const float tz = 2.0f * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + q[2]*tz - q[3]*ty;
    out[1] = v[1] + q[0]*ty + q[3]*tx - q[1]*tz;
    out[2] = v[2] + q[0]*tz + q[1]*ty - q[2]*tx;
}

inline void rl_physics_derivative(thread const RLPhysicsParams& p, thread const RLPhysicsState& s,
                                  thread const float* setpoint, thread const float* wind_force_world,
                                  thread RLPhysicsState& d) {
    float thrust[3] = {0.0f, 0.0f, 0.0f};
    float torque[3] = {0.0f, 0.0f, 0.0f};
    for (uint r = 0; r < 4; ++r) {
        const float speed = s.rpm[r];
        const uint c = 3*r;
        const float magnitude = p.rotor_thrust_coefficients[c] + p.rotor_thrust_coefficients[c+1]*speed + p.rotor_thrust_coefficients[c+2]*speed*speed;
        float rotor_thrust[3];
        for (uint i = 0; i < 3; ++i) {
            rotor_thrust[i] = p.rotor_thrust_directions[c+i] * magnitude;
            thrust[i] += rotor_thrust[i];
            torque[i] += p.rotor_torque_directions[c+i] * magnitude * p.rotor_torque_constants[r];
        }
        torque[0] += p.rotor_positions[c+1]*rotor_thrust[2] - p.rotor_positions[c+2]*rotor_thrust[1];
        torque[1] += p.rotor_positions[c+2]*rotor_thrust[0] - p.rotor_positions[c]*rotor_thrust[2];
        torque[2] += p.rotor_positions[c]*rotor_thrust[1] - p.rotor_positions[c+1]*rotor_thrust[0];
    }
    for (uint i = 0; i < 3; ++i) d.position[i] = s.linear_velocity[i];
    thread const float* q = s.orientation_wxyz;
    thread const float* w = s.angular_velocity_body;
    d.orientation_wxyz[0] = -q[1]*w[0] - q[2]*w[1] - q[3]*w[2];
    d.orientation_wxyz[1] =  q[0]*w[0] + q[2]*w[2] - q[3]*w[1];
    d.orientation_wxyz[2] =  q[0]*w[1] + q[3]*w[0] - q[1]*w[2];
    d.orientation_wxyz[3] =  q[0]*w[2] + q[1]*w[1] - q[2]*w[0];
    for (uint i = 0; i < 4; ++i) d.orientation_wxyz[i] *= 0.5f;
    rl_physics_rotate_body_to_world(q, thrust, d.linear_velocity);
    for (uint i = 0; i < 3; ++i)
        d.linear_velocity[i] = d.linear_velocity[i]/p.mass + p.gravity_world[i] + wind_force_world[i]/p.mass;

    float angular_momentum[3] = {0.0f, 0.0f, 0.0f};
    for (uint row = 0; row < 3; ++row)
        for (uint col = 0; col < 3; ++col)
            angular_momentum[row] += p.inertia[3*row+col] * w[col];
    float gyroscopic[3];
    rl_physics_cross(w, angular_momentum, gyroscopic);
    for (uint i = 0; i < 3; ++i) angular_momentum[i] = torque[i] - gyroscopic[i];
    for (uint row = 0; row < 3; ++row) {
        d.angular_velocity_body[row] = 0.0f;
        for (uint col = 0; col < 3; ++col)
            d.angular_velocity_body[row] += p.inertia_inverse[3*row+col] * angular_momentum[col];
    }
    for (uint r = 0; r < 4; ++r) {
        const float tau = setpoint[r] >= s.rpm[r]
            ? p.rotor_time_constants_rising[r] : p.rotor_time_constants_falling[r];
        d.rpm[r] = (setpoint[r] - s.rpm[r]) / tau;
    }
}

inline void rl_physics_add_scaled(thread const RLPhysicsState& s, thread const RLPhysicsState& d,
                                  float scale, thread RLPhysicsState& out) {
    for (uint i = 0; i < 3; ++i) out.position[i] = s.position[i] + scale*d.position[i];
    for (uint i = 0; i < 4; ++i) out.orientation_wxyz[i] = s.orientation_wxyz[i] + scale*d.orientation_wxyz[i];
    for (uint i = 0; i < 3; ++i) out.linear_velocity[i] = s.linear_velocity[i] + scale*d.linear_velocity[i];
    for (uint i = 0; i < 3; ++i) out.angular_velocity_body[i] = s.angular_velocity_body[i] + scale*d.angular_velocity_body[i];
    for (uint i = 0; i < 4; ++i) out.rpm[i] = s.rpm[i] + scale*d.rpm[i];
}

inline void rl_physics_rk4(thread const RLPhysicsParams& p, thread const RLPhysicsState& s,
                           thread const float* setpoint, thread const float* wind_force_world,
                           thread RLPhysicsState& out) {
    RLPhysicsState k1{}, k2{}, k3{}, k4{}, temp{};
    rl_physics_derivative(p, s, setpoint, wind_force_world, k1);
    rl_physics_add_scaled(s, k1, 0.5f*p.dt, temp);
    rl_physics_derivative(p, temp, setpoint, wind_force_world, k2);
    rl_physics_add_scaled(s, k2, 0.5f*p.dt, temp);
    rl_physics_derivative(p, temp, setpoint, wind_force_world, k3);
    rl_physics_add_scaled(s, k3, p.dt, temp);
    rl_physics_derivative(p, temp, setpoint, wind_force_world, k4);
    for (uint i = 0; i < 3; ++i)
        out.position[i] = s.position[i] + p.dt*(k1.position[i]+2.0f*k2.position[i]+2.0f*k3.position[i]+k4.position[i])/6.0f;
    for (uint i = 0; i < 4; ++i)
        out.orientation_wxyz[i] = s.orientation_wxyz[i] + p.dt*(k1.orientation_wxyz[i]+2.0f*k2.orientation_wxyz[i]+2.0f*k3.orientation_wxyz[i]+k4.orientation_wxyz[i])/6.0f;
    for (uint i = 0; i < 3; ++i)
        out.linear_velocity[i] = s.linear_velocity[i] + p.dt*(k1.linear_velocity[i]+2.0f*k2.linear_velocity[i]+2.0f*k3.linear_velocity[i]+k4.linear_velocity[i])/6.0f;
    for (uint i = 0; i < 3; ++i)
        out.angular_velocity_body[i] = s.angular_velocity_body[i] + p.dt*(k1.angular_velocity_body[i]+2.0f*k2.angular_velocity_body[i]+2.0f*k3.angular_velocity_body[i]+k4.angular_velocity_body[i])/6.0f;
    for (uint i = 0; i < 4; ++i)
        out.rpm[i] = s.rpm[i] + p.dt*(k1.rpm[i]+2.0f*k2.rpm[i]+2.0f*k3.rpm[i]+k4.rpm[i])/6.0f;
}

// `action` uses RAPTOR's [-1,1] range and rotor order. Wind is a force in N.
inline void rl_physics_step(thread const RLPhysicsState& state, thread const float* action,
                            thread const float* wind_force_world, thread const RLPhysicsParams& params,
                            thread RLPhysicsState& next_state) {
    float setpoint[4];
    const float half_range = (params.action_max - params.action_min) * 0.5f;
    for (uint i = 0; i < 4; ++i) {
        const float a = clamp(action[i], -1.0f, 1.0f);
        setpoint[i] = a*half_range + params.action_min + half_range;
    }
    rl_physics_rk4(params, state, setpoint, wind_force_world, next_state);
    const float norm = sqrt(next_state.orientation_wxyz[0]*next_state.orientation_wxyz[0] +
                            next_state.orientation_wxyz[1]*next_state.orientation_wxyz[1] +
                            next_state.orientation_wxyz[2]*next_state.orientation_wxyz[2] +
                            next_state.orientation_wxyz[3]*next_state.orientation_wxyz[3]);
    for (uint i = 0; i < 4; ++i) next_state.orientation_wxyz[i] /= norm;
    for (uint i = 0; i < 3; ++i) {
        next_state.position[i] = clamp(next_state.position[i], -params.position_limit, params.position_limit);
        next_state.linear_velocity[i] = clamp(next_state.linear_velocity[i], -params.velocity_limit, params.velocity_limit);
        next_state.angular_velocity_body[i] = clamp(next_state.angular_velocity_body[i], -params.angular_velocity_limit, params.angular_velocity_limit);
    }
    for (uint i = 0; i < 4; ++i) next_state.rpm[i] = clamp(next_state.rpm[i], params.action_min, params.action_max);
}
