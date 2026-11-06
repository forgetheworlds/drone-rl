// Training environment kernels. The host builds the source from physics.metal, raptor.metal,
// voxel_world.h, sensors.h, nav_task.h, nav_env.metal and this file.
//
// Per tick: train_depth (true ranges), train_observe (sensors and observations), the policy on
// the CPU, then train_step (controller, physics, rewards and automatic reset).


// RAPTOR as in drone-rl's raptor_motor_command, fed with the measured (estimated) state.
inline void nt_raptor(device const float *weights, thread TrainEnv &e, thread const RLPhysicsState &measured,
                      thread const float *command, float dt, thread float *motors) {
    for (uint i = 0; i < 3; ++i)
        e.reference[i] = e.reference[i] + command[i] * dt;
    float error[3] = {e.reference[0] - measured.position[0], e.reference[1] - measured.position[1],
                      e.reference[2] - measured.position[2]};
    const float distance = sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
    if (distance > .25f) {
        const float inverse = 1 / max(distance, 1e-6f);
        for (uint i = 0; i < 3; ++i)
            e.reference[i] = measured.position[i] + (error[i] * inverse) * .25f;
    }
    e.reference_yaw += command[3] * dt;
    const float orientation[4] = {cos(e.reference_yaw / 2), 0, 0, sin(e.reference_yaw / 2)};
    const float velocity[3] = {command[0], command[1], command[2]};
    float observation[22];
    nav_pack_observation(measured, e.reference, orientation, velocity, e.previous_motor, observation);
    raptor_forward(weights, observation, e.hidden, motors);
    raptor_clip_action(motors);
    for (uint i = 0; i < 4; ++i)
        e.previous_motor[i] = motors[i];
}

inline float nt_hover(constant RLPhysicsParams &p) {
    const float thrust = p.mass * 9.81f / 4, c0 = p.rotor_thrust_coefficients[0], c1 = p.rotor_thrust_coefficients[1],
                c2 = p.rotor_thrust_coefficients[2];
    return (-c1 + sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
}

inline void nt_start(thread TrainEnv &e, device const NavMission *missions, device const float *weights,
                     constant RLPhysicsParams &params, constant NavTaskConfig &config) {
    e.mission = vx_random(e.rng) % config.mission_count;
    device const NavMission &m = missions[e.mission];
    float start[3] = {m.start[0], m.start[1], m.start[2]};
    nt_reset(e, start, m.launch_yaw, m.goal_count, nt_hover(params), config.sensor_level);
    if (config.attitude) {
        const float still[3] = {0, 0, 0};
        va_begin(e.attitude, vx_seed(e.rng, 0xa77u), config.sensor_level, e.sensor.gyro_noise, e.sensor.accel_noise,
                 e.sensor.vo_pixel, 1, start, still);
        e.attitude.compensate = 1;
    }
    e.dyn[0] = e.dyn[1] = e.dyn[2] = e.dyn[3] = 1;
#ifndef __METAL_VERSION__
    // RLV2_DYN_RAND=1 (CPU trainer): same ranges as grade.cpp; RAPTOR keeps the nominal hover estimate.
    static const bool dyn_rand = std::getenv("RLV2_DYN_RAND") && std::string(std::getenv("RLV2_DYN_RAND")) == "1";
    if (dyn_rand) {
        uint ds = vx_seed(e.rng, 0xd1u);
        e.dyn[0] = 1 + .3f * vx_uniform(ds), e.dyn[1] = .85f + .15f * vx_uniform(ds);
        e.dyn[2] = 1 + vx_uniform(ds), e.dyn[3] = 1 + .5f * vx_uniform(ds);
    }
#endif
    raptor_reset(weights, e.hidden);
}

kernel void train_reset(device TrainEnv *envs [[buffer(0)]], device const NavMission *missions [[buffer(1)]],
                        device const float *weights [[buffer(2)]], constant RLPhysicsParams &params [[buffer(3)]],
                        constant NavTaskConfig &config [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    TrainEnv e = envs[i];
    e.rng = vx_seed(i, config.seed);
    e.episodes = e.contacts = e.arrivals = e.completions = 0;
    nt_start(e, missions, weights, params, config);
    envs[i] = e;
}

kernel void train_depth(device const TrainEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                        device const uint *bits [[buffer(2)]], device float *depth [[buffer(3)]],
                        constant NavTaskConfig &config [[buffer(4)]], uint id [[thread_position_in_grid]]) {
    const uint i = id / VX_RAYS, k = id % VX_RAYS;
    if (i >= config.env_count)
        return;
    float q[4], rotation[9];
    for (uint j = 0; j < 4; ++j)
        q[j] = envs[i].physics.orientation_wxyz[j];
    vx_rotation(q, rotation);
    const VxVec p = vx_vec(envs[i].physics.position[0], envs[i].physics.position[1], envs[i].physics.position[2]);
    depth[id] = vx_ray(worlds[envs[i].mission], bits, p, vx_rotate(rotation, vx_camera_ray(k, .75f)), envs[i].steps * .01f);
}

// Nearest mover: position and velocity at time t.
inline bool nt_nearest_mover(device const VxWorld &w, VxVec p, float t, thread float *out) {
    float best = 1e9f;
    for (uint i = 0; i < w.mover_count; ++i) {
        const VxVec c = vx_mover_center(w.movers[i], t);
        const float d = vx_norm(vx_sub(c, p));
        if (d < best) {
            best = d;
            const float w0 = 6.28318530718f / w.movers[i].period;
            const float speed = -w.movers[i].amplitude * w0 * sin(w0 * t + w.movers[i].phase);
            out[0] = c.x;
            out[1] = c.y;
            out[2] = c.z;
            out[3] = w.movers[i].axis == 0 ? speed : 0;
            out[4] = w.movers[i].axis == 1 ? speed : 0;
            out[5] = w.movers[i].axis == 2 ? speed : 0;
        }
    }
    return best < 1e8f;
}

kernel void train_observe(device TrainEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                          device const float *depth [[buffer(2)]], device const NavMission *missions [[buffer(3)]],
                          device const float4 *goals [[buffer(4)]], device float *teacher [[buffer(5)]],
                          device float *student [[buffer(6)]], constant NavTaskConfig &config [[buffer(7)]],
                          device const uint *bits [[buffer(8)]], device const float *paths [[buffer(9)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    TrainEnv e = envs[i];
    const float4 g = goals[missions[e.mission].goal_offset + e.goal];
    float mover[6];
    const bool has_mover = nt_nearest_mover(worlds[e.mission],
                                            vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]),
                                            e.steps * .01f, mover);
    float t[NT_TEACHER_OBS], s[NT_STUDENT_OBS];
    float proximity[NT_PROXIMITY];
    nt_proximity(worlds[e.mission], bits, vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]),
                 nt_yaw(e.physics.orientation_wxyz), proximity);
    device const float *path = paths + size_t(missions[e.mission].goal_offset + e.goal) * NT_PATH * 4;
    nt_observe(e, depth + size_t(i) * VX_RAYS, vx_vec(g.x, g.y, g.z), mover, has_mover, proximity, path, t, s);
    for (uint j = 0; j < NT_TEACHER_OBS; ++j)
        teacher[size_t(i) * NT_TEACHER_OBS + j] = t[j];
    for (uint j = 0; j < NT_STUDENT_OBS; ++j)
        student[size_t(i) * NT_STUDENT_OBS + j] = s[j];
    envs[i] = e;
}

// rewards[i] gets the tick reward, dones[i] 0 (running), 1 (ended) or 2 (out of time).
kernel void train_step(device TrainEnv *envs [[buffer(0)]], device const VxWorld *worlds [[buffer(1)]],
                       device const uint *bits [[buffer(2)]], device const float *actions [[buffer(3)]],
                       constant NavTaskConfig &config [[buffer(4)]], device const NavMission *missions [[buffer(5)]],
                       device const float4 *goals [[buffer(6)]], device const float *weights [[buffer(7)]],
                       constant RLPhysicsParams &params [[buffer(8)]], device float *rewards [[buffer(9)]],
                       device uint *dones [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    if (i >= config.env_count)
        return;
    TrainEnv e = envs[i];
    device const VxWorld &world = worlds[e.mission];
    const NavTaskConfig c = config;
    float a[4], command[4];
    for (uint j = 0; j < 4; ++j)
        a[j] = actions[size_t(i) * 4 + j];
    nt_command(e, a, c, command);
    RLPhysicsParams p = params;
    p.mass *= e.dyn[0];
    for (uint j = 0; j < 12; ++j)
        p.rotor_thrust_coefficients[j] *= e.dyn[1];
    for (uint j = 0; j < 4; ++j)
        p.rotor_time_constants_rising[j] *= e.dyn[2], p.rotor_time_constants_falling[j] *= e.dyn[3];
    bool contact = false;
    for (uint k = 0; k < 5 && !contact; ++k) {
        const RLPhysicsState measured = nt_measure(e);
        float motors[4];
        nt_raptor(weights, e, measured, command, p.dt, motors);
        const float wind[3] = {0, 0, 0};
        RLPhysicsState next;
        rl_physics_step(e.physics, motors, wind, p, next);
        e.physics = next;
        ++e.steps;
        contact = vx_contact(world, bits, vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]),
                             e.steps * .01f);
    }
    const float4 g = goals[missions[e.mission].goal_offset + e.goal];
    nt_finish_tick(e, contact, vx_vec(g.x, g.y, g.z));
    rewards[i] = e.reward;
    dones[i] = e.done;
    if (e.done)
        nt_start(e, missions, weights, params, config);
    envs[i] = e;
}
