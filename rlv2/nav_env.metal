// Waypoint navigation environment on voxel worlds.
//
// The host builds the Metal source from physics.metal, raptor.metal, voxel_world.h and this
// file, in that order. One tick is 50 ms: the depth frame and goal visibility are taken at the
// start of the tick, then RAPTOR and L2F run five 10 ms substeps, with a contact check after
// each one. This is the same loop as drone-rl's src/benchmark.cpp.

struct NavEnv {
    RLPhysicsState physics;
    float hidden[16];
    float previous_motor[4];
    float reference[3];
    float reference_yaw;
    uint mission, goal, steps, tick;
    uint settle, visible, episodes, contacts;
    uint arrivals, rng, done, pad;
};

struct NavMission {
    float start[3];
    float launch_yaw;
    uint goal_offset, goal_count, pad0, pad1;
};

struct NavConfig {
    uint env_count, mission_count, record_ticks, tick;
    float max_speed, max_climb, max_yaw_rate, timeout_ticks;
};

struct ParityRecord {
    float position[3], velocity[3], orientation[4];
    uint alive, contact, contact_substep, visible;
    float depth[320];
};

inline void nav_quaternion_matrix(thread const float *q, thread float *r) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1 - 2 * y * y - 2 * z * z;
    r[1] = 2 * x * y - 2 * w * z;
    r[2] = 2 * x * z + 2 * w * y;
    r[3] = 2 * x * y + 2 * w * z;
    r[4] = 1 - 2 * x * x - 2 * z * z;
    r[5] = 2 * y * z - 2 * w * x;
    r[6] = 2 * x * z - 2 * w * y;
    r[7] = 2 * y * z + 2 * w * x;
    r[8] = 1 - 2 * x * x - 2 * y * y;
}

// Same as raptor_pack_observation in raptor.hpp.
inline void nav_pack_observation(thread const RLPhysicsState &s, thread const float *target,
                                 thread const float *target_orientation,
                                 thread const float *target_velocity,
                                 thread const float *previous_action, thread float *observation) {
    const float inverse[4] = {target_orientation[0], -target_orientation[1],
                              -target_orientation[2], -target_orientation[3]};
    float rotation[9];
    nav_quaternion_matrix(inverse, rotation);
    float position_error[3], velocity_error[3];
    for (uint i = 0; i < 3; ++i) {
        position_error[i] = s.position[i] - target[i];
        velocity_error[i] = s.linear_velocity[i] - target_velocity[i];
    }
    for (uint row = 0; row < 3; ++row) {
        float p = 0, v = 0;
        for (uint col = 0; col < 3; ++col) {
            p += rotation[row * 3 + col] * position_error[col];
            v += rotation[row * 3 + col] * velocity_error[col];
        }
        observation[row] = max(-0.5f, min(0.5f, p));
        observation[12 + row] = max(-1.0f, min(1.0f, v));
    }
    const float w = inverse[0], x = inverse[1], y = inverse[2], z = inverse[3];
    const float bw = s.orientation_wxyz[0], bx = s.orientation_wxyz[1], by = s.orientation_wxyz[2],
                bz = s.orientation_wxyz[3];
    const float relative[4] = {w * bw - x * bx - y * by - z * bz, w * bx + x * bw + y * bz - z * by,
                               w * by - x * bz + y * bw + z * bx, w * bz + x * by - y * bx + z * bw};
    nav_quaternion_matrix(relative, observation + 3);
    for (uint i = 0; i < 3; ++i)
        observation[15 + i] = s.angular_velocity_body[i];
    for (uint i = 0; i < 4; ++i)
        observation[18 + i] = previous_action[i];
}

// Same as raptor_motor_command in drone-rl's raptor_model.hpp: the command is a world
// velocity and yaw rate. The position reference moves with the command and is kept within
// 0.25 m of the vehicle.
inline void nav_raptor_command(device const float *weights, thread NavEnv &e, float4 command,
                               float dt, thread float *motors) {
    for (uint i = 0; i < 3; ++i)
        e.reference[i] = e.reference[i] + command[i] * dt;
    float error[3] = {e.reference[0] - e.physics.position[0], e.reference[1] - e.physics.position[1],
                      e.reference[2] - e.physics.position[2]};
    const float distance = sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
    if (distance > .25f) {
        const float inverse = 1 / max(distance, 1e-6f);
        for (uint i = 0; i < 3; ++i)
            e.reference[i] = e.physics.position[i] + (error[i] * inverse) * .25f;
    }
    e.reference_yaw += command[3] * dt;
    const float orientation[4] = {cos(e.reference_yaw / 2), 0, 0, sin(e.reference_yaw / 2)};
    const float velocity[3] = {command[0], command[1], command[2]};
    float observation[22];
    nav_pack_observation(e.physics, e.reference, orientation, velocity, e.previous_motor, observation);
    raptor_forward(weights, observation, e.hidden, motors);
    raptor_clip_action(motors);
    for (uint i = 0; i < 4; ++i)
        e.previous_motor[i] = motors[i];
}

// One 10 ms substep. Returns true on contact at the end of the substep.
inline bool nav_substep(device const float *weights, constant RLPhysicsParams &params,
                        device const VxWorld &world, device const uint *bits, thread NavEnv &e,
                        float4 command) {
    RLPhysicsParams p = params;
    float motors[4];
    nav_raptor_command(weights, e, command, p.dt, motors);
    const float wind[3] = {0, 0, 0};
    RLPhysicsState next;
    rl_physics_step(e.physics, motors, wind, p, next);
    e.physics = next;
    ++e.steps;
    return vx_contact(world, bits, vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]),
                      e.steps * .01f);
}

inline void nav_reset(thread NavEnv &e, device const NavMission &m, device const float *weights,
                      constant RLPhysicsParams &params) {
    for (uint i = 0; i < 3; ++i) {
        e.physics.position[i] = m.start[i];
        e.physics.linear_velocity[i] = 0;
        e.physics.angular_velocity_body[i] = 0;
        e.reference[i] = m.start[i];
    }
    e.physics.orientation_wxyz[0] = cos(m.launch_yaw / 2);
    e.physics.orientation_wxyz[1] = 0;
    e.physics.orientation_wxyz[2] = 0;
    e.physics.orientation_wxyz[3] = sin(m.launch_yaw / 2);
    const float thrust = params.mass * 9.81f / 4, c0 = params.rotor_thrust_coefficients[0],
                c1 = params.rotor_thrust_coefficients[1], c2 = params.rotor_thrust_coefficients[2];
    const float hover = (-c1 + sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
    for (uint i = 0; i < 4; ++i) {
        e.physics.rpm[i] = hover;
        e.previous_motor[i] = 0;
    }
    e.reference_yaw = m.launch_yaw;
    raptor_reset(weights, e.hidden);
    e.goal = 0;
    e.steps = 0;
    e.tick = 0;
    e.settle = 0;
    e.visible = 0;
    e.done = 0;
}

inline uint nav_random(thread uint &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

kernel void nav_reset_kernel(device NavEnv *envs [[buffer(0)]], device const NavMission *missions [[buffer(1)]],
                             device const float *weights [[buffer(2)]],
                             constant RLPhysicsParams &params [[buffer(3)]],
                             constant NavConfig &config [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    NavEnv e = envs[i];
    e.mission = i % config.mission_count;
    e.rng = 0x9e3779b9u * (i + 1);
    e.episodes = e.contacts = e.arrivals = 0;
    nav_reset(e, missions[e.mission], weights, params);
    envs[i] = e;
}

// ---------------------------------------------------------------------------------------------
// Parity: replays fixed command sequences and records every tick.

kernel void nav_parity_depth(device const NavEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                             device const uint *bits [[buffer(2)]], device ParityRecord *records [[buffer(3)]],
                             constant NavConfig &config [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    const uint i = id / VX_RAYS, k = id % VX_RAYS;
    if (i >= config.env_count || envs[i].done)
        return;
    float q[4], rotation[9];
    for (uint j = 0; j < 4; ++j)
        q[j] = envs[i].physics.orientation_wxyz[j];
    vx_rotation(q, rotation);
    const VxVec p = vx_vec(envs[i].physics.position[0], envs[i].physics.position[1], envs[i].physics.position[2]);
    records[size_t(i) * config.record_ticks + config.tick].depth[k] =
        vx_ray(worlds[envs[i].mission], bits, p, vx_rotate(rotation, vx_camera_ray(k, .75f)), envs[i].steps * .01f);
}

kernel void nav_parity_step(device NavEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                            device const uint *bits [[buffer(2)]], device ParityRecord *records [[buffer(3)]],
                            constant NavConfig &config [[buffer(4)]], device const NavMission *missions [[buffer(5)]],
                            device const float4 *goals [[buffer(6)]], device const float *weights [[buffer(7)]],
                            constant RLPhysicsParams &params [[buffer(8)]],
                            device const float4 *commands [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    NavEnv e = envs[i];
    device ParityRecord &r = records[size_t(i) * config.record_ticks + config.tick];
    device const VxWorld &world = worlds[e.mission];
    r.alive = 0;
    r.contact = 0;
    r.contact_substep = 0;
    r.visible = 0;
    if (!e.done) {
        r.alive = 1;
        float rotation[9];
        vx_rotation(e.physics.orientation_wxyz, rotation);
        const float4 g = goals[missions[e.mission].goal_offset];
        r.visible = vx_goal_visible(world, bits, rotation,
                                    vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]),
                                    vx_vec(g.x, g.y, g.z), e.steps * .01f);
        const float4 command = commands[size_t(i) * config.record_ticks + config.tick];
        for (uint k = 0; k < 5; ++k) {
            if (nav_substep(weights, params, world, bits, e, command)) {
                r.contact = 1;
                r.contact_substep = k;
                e.done = 1;
                break;
            }
        }
    }
    for (uint j = 0; j < 3; ++j) {
        r.position[j] = e.physics.position[j];
        r.velocity[j] = e.physics.linear_velocity[j];
    }
    for (uint j = 0; j < 4; ++j)
        r.orientation[j] = e.physics.orientation_wxyz[j];
    envs[i] = e;
}

// ---------------------------------------------------------------------------------------------
// Throughput loop: depth, pooled observation and actor, then the step with waypoints and
// automatic reset. Every env tick counted is a live tick.

#define NAV_BINS 40
#define NAV_OBS 52
#define NAV_HIDDEN 128

kernel void nav_depth(device const NavEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                      device const uint *bits [[buffer(2)]], device float *depth [[buffer(3)]],
                      constant NavConfig &config [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    const uint i = id / VX_RAYS, k = id % VX_RAYS;
    if (i >= config.env_count)
        return;
    float q[4], rotation[9];
    for (uint j = 0; j < 4; ++j)
        q[j] = envs[i].physics.orientation_wxyz[j];
    vx_rotation(q, rotation);
    const VxVec p = vx_vec(envs[i].physics.position[0], envs[i].physics.position[1], envs[i].physics.position[2]);
    depth[id] = vx_ray(worlds[envs[i].mission], bits, p, vx_rotate(rotation, vx_camera_ray(k, .75f)),
                       envs[i].steps * .01f);
}

// Observation: 40 depth bins (min over 2 x 8 ray tiles), goal in the body frame, goal visible,
// body velocity, and the previous action. The actor is a 52-128-128-4 tanh MLP.
kernel void nav_act(device const NavEnv *envs [[buffer(0)]], device const float *depth [[buffer(1)]],
                    device const NavMission *missions [[buffer(2)]], device const float4 *goals [[buffer(3)]],
                    device const float *actor [[buffer(4)]], device float4 *actions [[buffer(5)]],
                    constant NavConfig &config [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    const device NavEnv &e = envs[i];
    float obs[NAV_OBS];
    for (uint b = 0; b < NAV_BINS; ++b) {
        const uint col0 = (b % 10) * 2, row0 = (b / 10) * 4;
        float m = VX_MAX_RANGE;
        for (uint r = 0; r < 4; ++r)
            for (uint c = 0; c < 2; ++c)
                m = min(m, depth[size_t(i) * VX_RAYS + (row0 + r) * VX_RAY_COLS + col0 + c]);
        obs[b] = m / VX_MAX_RANGE;
    }
    float q[4], rotation[9];
    for (uint j = 0; j < 4; ++j)
        q[j] = e.physics.orientation_wxyz[j];
    vx_rotation(q, rotation);
    const float4 g = goals[missions[e.mission].goal_offset + e.goal];
    const VxVec to_goal = vx_inverse_rotate(
        rotation, vx_vec(g.x - e.physics.position[0], g.y - e.physics.position[1], g.z - e.physics.position[2]));
    const VxVec velocity = vx_inverse_rotate(
        rotation, vx_vec(e.physics.linear_velocity[0], e.physics.linear_velocity[1], e.physics.linear_velocity[2]));
    obs[40] = clamp(to_goal.x / 12, -1.0f, 1.0f);
    obs[41] = clamp(to_goal.y / 12, -1.0f, 1.0f);
    obs[42] = clamp(to_goal.z / 12, -1.0f, 1.0f);
    obs[43] = e.visible;
    obs[44] = velocity.x;
    obs[45] = velocity.y;
    obs[46] = velocity.z;
    for (uint j = 0; j < 4; ++j)
        obs[47 + j] = actions[i][j];
    obs[51] = float(e.tick) / config.timeout_ticks;

    float h1[NAV_HIDDEN], h2[NAV_HIDDEN];
    device const float *w = actor;
    for (uint r = 0; r < NAV_HIDDEN; ++r) {
        float v = w[NAV_HIDDEN * NAV_OBS + r];
        for (uint c = 0; c < NAV_OBS; ++c)
            v += w[r * NAV_OBS + c] * obs[c];
        h1[r] = tanh(v);
    }
    w += NAV_HIDDEN * NAV_OBS + NAV_HIDDEN;
    for (uint r = 0; r < NAV_HIDDEN; ++r) {
        float v = w[NAV_HIDDEN * NAV_HIDDEN + r];
        for (uint c = 0; c < NAV_HIDDEN; ++c)
            v += w[r * NAV_HIDDEN + c] * h1[c];
        h2[r] = tanh(v);
    }
    w += NAV_HIDDEN * NAV_HIDDEN + NAV_HIDDEN;
    float4 a;
    for (uint r = 0; r < 4; ++r) {
        float v = w[4 * NAV_HIDDEN + r];
        for (uint c = 0; c < NAV_HIDDEN; ++c)
            v += w[r * NAV_HIDDEN + c] * h2[c];
        a[r] = tanh(v);
    }
    // The actor has random weights, so on its own it only drifts. For the speed test the action
    // is a pull toward the goal plus the actor output and per-env noise. That way the drones
    // cover the maps, hit things, reach goals and reset, like they would while training.
    uint noise = (i + 1) * 0x9e3779b9u ^ (e.steps + 1) * 0x85ebca6bu;
    float pull[4];
    const float distance = max(sqrt(to_goal.x * to_goal.x + to_goal.y * to_goal.y + to_goal.z * to_goal.z), 1e-6f);
    const float gain = min(distance, 1.0f) / distance;
    pull[0] = to_goal.x * gain;
    pull[1] = to_goal.y * gain;
    pull[2] = to_goal.z * gain;
    pull[3] = clamp(atan2(to_goal.y, to_goal.x), -1.0f, 1.0f);
    for (uint r = 0; r < 4; ++r) {
        const float jitter = float(nav_random(noise) >> 8) * (1.0f / 16777216.0f) - .5f;
        a[r] = clamp(pull[r] + .5f * a[r] + .6f * jitter, -1.0f, 1.0f);
    }
    actions[i] = a;
}

// Actions are a velocity in the yaw frame plus a yaw rate. A goal counts as reached after the
// drone holds within 0.25 m and under 0.2 m/s for 6 ticks, as in drone-rl.
kernel void nav_step(device NavEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                     device const uint *bits [[buffer(2)]], device const float4 *actions [[buffer(3)]],
                     constant NavConfig &config [[buffer(4)]], device const NavMission *missions [[buffer(5)]],
                     device const float4 *goals [[buffer(6)]], device const float *weights [[buffer(7)]],
                     constant RLPhysicsParams &params [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    NavEnv e = envs[i];
    device const VxWorld &world = worlds[e.mission];
    device const NavMission &m = missions[e.mission];
    const float4 a = actions[i];
    const float yaw = e.reference_yaw, cy = cos(yaw), sy = sin(yaw);
    const float vx = a.x * config.max_speed, vy = a.y * config.max_speed;
    const float4 command = float4(cy * vx - sy * vy, sy * vx + cy * vy, a.z * config.max_climb,
                                  a.w * config.max_yaw_rate);
    bool contact = false;
    for (uint k = 0; k < 5 && !contact; ++k)
        contact = nav_substep(weights, params, world, bits, e, command);
    ++e.tick;
    const float4 g = goals[m.goal_offset + e.goal];
    const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
    const float speed = vx_norm(vx_vec(e.physics.linear_velocity[0], e.physics.linear_velocity[1],
                                       e.physics.linear_velocity[2]));
    e.settle = (vx_norm(vx_sub(p, vx_vec(g.x, g.y, g.z))) < .25f && speed < .2f) ? e.settle + 1 : 0;
    bool finished = false;
    if (e.settle >= 6) {
        ++e.arrivals;
        e.settle = 0;
        finished = ++e.goal >= m.goal_count;
    }
    if (contact)
        ++e.contacts;
    if (contact || finished || e.tick >= uint(config.timeout_ticks)) {
        ++e.episodes;
        e.mission = nav_random(e.rng) % config.mission_count;
        nav_reset(e, missions[e.mission], weights, params);
    } else {
        float rotation[9];
        vx_rotation(e.physics.orientation_wxyz, rotation);
        const float4 next = goals[m.goal_offset + e.goal];
        e.visible = vx_goal_visible(world, bits, rotation, p, vx_vec(next.x, next.y, next.z), e.steps * .01f);
    }
    envs[i] = e;
}
