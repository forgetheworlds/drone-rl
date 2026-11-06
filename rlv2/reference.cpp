// Flies the parity command sequences through drone-rl's own C++ simulator (VisibleWorld plus
// RaptorVehicleModel) and writes one ParityRecord per tick. With --geometry-check it also
// checks this repo's voxel_world.h against drone-rl's world code on every recorded pose.
//
// Build against a drone-rl checkout: see rlv2/CMakeLists.txt (DRONE_RL_DIR).

#include "missions.hpp"
#include "scenarios.hpp"
#include "navigation/raptor_model.hpp"
#include "navigation/visible_world.hpp"
#include <cstdio>
#include <cstring>
#include <iostream>

static bool drone_rl_visible(const VisibleWorld &world, const safe_nav::Pose &pose, Vec3 goal) {
    Vec3 d = goal - pose.position;
    float distance = safe_nav::norm(d);
    Vec3 body = safe_nav::inverse_rotate(pose, d);
    bool frustum = body.x > 0 && std::fabs(body.y) <= body.x && std::fabs(body.z) <= .75f * body.x;
    bool range = distance > .25f && distance < 11.9f;
    float ray_range = world.ray(pose.position, safe_nav::unit(d));
    return frustum && range && ray_range + .001f >= distance;
}

int main(int argc, char **argv) {
    try {
        if (argc < 4)
            throw std::runtime_error(
                "rlv2_reference MISSIONS_DIR RAPTOR_BIN OUTPUT [SEQUENCES] [TICKS] [--geometry-check]");
        const std::string missions_dir = argv[1], weights_path = argv[2], output = argv[3];
        ParityPlan plan;
        bool geometry_check = false;
        int positional = 0;
        for (int i = 4; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--geometry-check")
                geometry_check = true;
            else if (positional++ == 0)
                plan.sequences = std::stoul(a);
            else
                plan.ticks = std::stoul(a);
        }
        MissionSet set = load_missions(missions_dir);
        RaptorWeights weights;
        if (!raptor_load_weights(weights_path.c_str(), &weights))
            throw std::runtime_error("Invalid RAPTOR weights");
        const auto dynamics = rl_physics_crazyflie_default();
        safe_nav::sensor_slope = .75f;

        std::FILE *out = std::fopen(output.c_str(), "wb");
        if (!out)
            throw std::runtime_error("Cannot write " + output);
        const uint32_t header[4] = {0x31585652u, plan.sequences, plan.ticks, plan.seed};
        std::fwrite(header, sizeof(header), 1, out);

        double max_ray_error = 0;
        uint64_t checks = 0, contact_mismatch = 0, visible_mismatch = 0, contacts = 0;
        std::map<std::string, VisibleWorld> worlds;
        for (uint32_t sequence = 0; sequence < plan.sequences; sequence++) {
            const Mission &m = set.missions[sequence % set.missions.size()];
            auto found = worlds.find(m.map_path);
            if (found == worlds.end())
                found = worlds.emplace(m.map_path, VisibleWorld(m.map_path)).first;
            VisibleWorld &world = found->second;
            world.scale = m.world.scale;
            world.movers.clear();
            for (uint32_t i = 0; i < m.world.mover_count; i++) {
                const VxMover &v = m.world.movers[i];
                MovingBox box;
                box.center = {v.center[0], v.center[1], v.center[2]};
                box.half = {v.extent[0], v.extent[1], v.extent[2]};
                box.axis = int(v.axis);
                box.amplitude = v.amplitude;
                box.period = v.period;
                box.phase = v.phase;
                world.movers.push_back(box);
            }
            const Vec3 start{m.start[0], m.start[1], m.start[2]};
            const Vec3 goal{m.goals[0][0], m.goals[0][1], m.goals[0][2]};
            RLPhysicsState physics{};
            physics.orientation_wxyz[0] = std::cos(m.launch_yaw / 2);
            physics.orientation_wxyz[3] = std::sin(m.launch_yaw / 2);
            for (int j = 0; j < 3; j++)
                physics.position[j] = m.start[j];
            float thrust = dynamics.mass * 9.81f / 4, c0 = dynamics.rotor_thrust_coefficients[0],
                  c1 = dynamics.rotor_thrust_coefficients[1],
                  c2 = dynamics.rotor_thrust_coefficients[2];
            float hover = (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
            std::fill(physics.rpm, physics.rpm + 4, hover);
            safe_nav::RaptorControlState control;
            control.reference = start;
            control.reference_yaw = m.launch_yaw;
            raptor_reset(weights, control.hidden);
            safe_nav::RaptorVehicleModel vehicle(weights, dynamics, physics, control);

            const auto commands = parity_commands(plan, sequence);
            bool alive = true;
            uint32_t steps = 0;
            for (uint32_t tick = 0; tick < plan.ticks; tick++) {
                ParityRecord r{};
                if (alive) {
                    world.time = steps * .01f;
                    const auto pose = vehicle.pose();
                    const auto depth = world.depth(pose);
                    std::memcpy(r.depth, depth.data(), sizeof(r.depth));
                    r.visible = drone_rl_visible(world, pose, goal);
                    if (geometry_check) {
                        float rotation[9];
                        vx_rotation(vehicle.physics.orientation_wxyz, rotation);
                        const VxVec p = vx_vec(pose.position.x, pose.position.y, pose.position.z);
                        for (uint32_t k = 0; k < VX_RAYS; k++) {
                            float ray = vx_ray(m.world, set.bits.data(), p,
                                               vx_rotate(rotation, vx_camera_ray(k, .75f)), world.time);
                            max_ray_error = std::max(max_ray_error, double(std::fabs(ray - depth[k])));
                        }
                        visible_mismatch +=
                            vx_goal_visible(m.world, set.bits.data(), rotation, p,
                                            vx_vec(goal.x, goal.y, goal.z), world.time) != bool(r.visible);
                        checks++;
                    }
                    safe_nav::Command command;
                    command.world_velocity = {commands[tick * 4], commands[tick * 4 + 1],
                                              commands[tick * 4 + 2]};
                    command.yaw_rate = commands[tick * 4 + 3];
                    for (uint32_t k = 0; k < 5; k++) {
                        if (!vehicle.substep(command))
                            throw std::runtime_error("Vehicle model fault");
                        ++steps;
                        world.time = steps * .01f;
                        const auto q = vehicle.pose().position;
                        const bool hit = world.contact(q);
                        if (geometry_check)
                            contact_mismatch += vx_contact(m.world, set.bits.data(),
                                                           vx_vec(q.x, q.y, q.z), world.time) != hit;
                        if (hit) {
                            r.contact = 1;
                            r.contact_substep = k;
                            alive = false;
                            ++contacts;
                            break;
                        }
                    }
                    r.alive = 1;
                }
                for (int j = 0; j < 3; j++) {
                    r.position[j] = vehicle.physics.position[j];
                    r.velocity[j] = vehicle.physics.linear_velocity[j];
                }
                for (int j = 0; j < 4; j++)
                    r.orientation[j] = vehicle.physics.orientation_wxyz[j];
                std::fwrite(&r, sizeof(r), 1, out);
            }
        }
        std::fclose(out);
        std::cout << "sequences=" << plan.sequences << " ticks=" << plan.ticks
                  << " missions=" << set.missions.size() << " contact_sequences=" << contacts << "\n";
        if (geometry_check)
            std::cout << "geometry_check poses=" << checks << " max_ray_error_m=" << max_ray_error
                      << " contact_mismatches=" << contact_mismatch
                      << " visibility_mismatches=" << visible_mismatch << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
