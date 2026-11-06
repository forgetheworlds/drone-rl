// rlv2: voxel-world navigation on Metal.
//
//   rlv2 parity MISSIONS_DIR REFERENCE_BIN   compare against drone-rl (see reference.cpp)
//   rlv2 bench MISSIONS_DIR [TICKS_PER_BATCH] [actor|no-actor] [REPEATS]   throughput
//
// MISSIONS_DIR is drone-rl's missions/ folder. Those missions are evaluation data; the bench
// only uses them to measure speed.

#include "../physics.hpp"
#include "../raptor.hpp"
#include "paths.hpp"
#include "gpu.h"
#include "scenarios.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

struct NavEnv {
    RLPhysicsState physics;
    float hidden[16], previous_motor[4], reference[3], reference_yaw;
    uint32_t mission, goal, steps, tick, settle, visible, episodes, contacts, arrivals, rng, done, pad;
};
struct NavConfig {
    uint32_t env_count, mission_count, record_ticks, tick;
    float max_speed, max_climb, max_yaw_rate, timeout_ticks;
};
static_assert(sizeof(ParityRecord) == 1336, "ParityRecord layout");

static id<MTLBuffer> raptor_buffer(Gpu &gpu) {
    RaptorWeights weights;
    require(raptor_load_weights((std::string(SOURCE_DIR) + "/assets/raptor.bin").c_str(), &weights),
            "RAPTOR load failed");
    static_assert(sizeof(RaptorWeights) == 2084 * 4, "RAPTOR layout");
    return gpu.buffer(sizeof(weights), &weights);
}

// GpuMissions needs a teacher path per goal (paths.hpp), as grade.cpp builds them.
static MissionSet load_missions_with_paths(const std::string &missions_dir) {
    MissionSet set = load_missions(missions_dir);
    for (Mission &m : set.missions)
        m.paths = path_mission(m.world, set.bits.data(), m.start, m.goals);
    return set;
}

static int parity(const std::string &missions_dir, const std::string &reference_path) {
    std::ifstream file(reference_path, std::ios::binary);
    uint32_t header[4];
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    require(file && header[0] == 0x31585652u, "Bad reference file");
    ParityPlan plan;
    plan.sequences = header[1];
    plan.ticks = header[2];
    plan.seed = header[3];
    std::vector<ParityRecord> expected(size_t(plan.sequences) * plan.ticks);
    file.read(reinterpret_cast<char *>(expected.data()), expected.size() * sizeof(ParityRecord));
    require(bool(file), "Truncated reference file");

    Gpu gpu{"physics.metal", "raptor.metal", "rlv2/voxel_world.h", "rlv2/nav_env.metal"};
    MissionSet set = load_missions_with_paths(missions_dir);
    GpuMissions missions(gpu, set);
    id<MTLBuffer> raptor = raptor_buffer(gpu);
    RLPhysicsParams params = rl_physics_crazyflie_default();
    id<MTLBuffer> params_buffer = gpu.buffer(sizeof(params), &params);
    std::vector<float> commands;
    for (uint32_t s = 0; s < plan.sequences; s++) {
        auto c = parity_commands(plan, s);
        commands.insert(commands.end(), c.begin(), c.end());
    }
    id<MTLBuffer> command_buffer = gpu.buffer(commands.size() * 4, commands.data());
    id<MTLBuffer> envs = gpu.buffer(size_t(plan.sequences) * sizeof(NavEnv));
    id<MTLBuffer> records = gpu.buffer(expected.size() * sizeof(ParityRecord));
    std::vector<id<MTLBuffer>> configs;
    for (uint32_t t = 0; t < plan.ticks; t++) {
        NavConfig c{plan.sequences, missions.count, plan.ticks, t, 0, 0, 0, 0};
        configs.push_back(gpu.buffer(sizeof(c), &c));
    }
    const double start = now();
    id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
    gpu.dispatch(cb, "nav_reset_kernel", plan.sequences, {envs, missions.missions, raptor, params_buffer, configs[0]});
    for (uint32_t t = 0; t < plan.ticks; t++) {
        gpu.dispatch(cb, "nav_parity_depth", size_t(plan.sequences) * VX_RAYS,
                     {envs, missions.worlds, missions.bits, records, configs[t]});
        gpu.dispatch(cb, "nav_parity_step", plan.sequences,
                     {envs, missions.worlds, missions.bits, records, configs[t], missions.missions, missions.goals,
                      raptor, params_buffer, command_buffer});
    }
    gpu.finish(cb);
    const double gpu_seconds = now() - start;
    const ParityRecord *actual = static_cast<const ParityRecord *>(records.contents);

    double position = 0, velocity = 0, orientation = 0, ray = 0, ray_close = 0;
    uint64_t ticks = 0, rays = 0, ray_flips = 0, alive_mismatch = 0, contact_mismatch = 0,
             visible_mismatch = 0, contacts = 0;
    const double flip = 1e-3;
    for (size_t i = 0; i < expected.size(); i++) {
        const ParityRecord &e = expected[i], &a = actual[i];
        if (e.alive != a.alive) {
            ++alive_mismatch;
            continue;
        }
        if (!e.alive)
            continue;
        ++ticks;
        contacts += e.contact;
        contact_mismatch += e.contact != a.contact || (e.contact && e.contact_substep != a.contact_substep);
        visible_mismatch += e.visible != a.visible;
        for (int j = 0; j < 3; j++) {
            position = std::max(position, double(std::fabs(e.position[j] - a.position[j])));
            velocity = std::max(velocity, double(std::fabs(e.velocity[j] - a.velocity[j])));
        }
        for (int j = 0; j < 4; j++)
            orientation = std::max(orientation, double(std::fabs(e.orientation[j] - a.orientation[j])));
        for (int k = 0; k < VX_RAYS; k++) {
            const double d = std::fabs(e.depth[k] - a.depth[k]);
            ++rays;
            ray = std::max(ray, d);
            if (d > flip)
                ++ray_flips;
            else
                ray_close = std::max(ray_close, d);
        }
    }
    std::cout << "sequences=" << plan.sequences << " ticks_per_sequence=" << plan.ticks
              << " live_ticks=" << ticks << " reference_contacts=" << contacts << " gpu_s=" << gpu_seconds << "\n"
              << "max_position_error_m=" << position << "\n"
              << "max_velocity_error_mps=" << velocity << "\n"
              << "max_orientation_error=" << orientation << "\n"
              << "max_ray_error_m=" << ray << " (rays=" << rays << ", over 1e-3 m: " << ray_flips << ")\n"
              << "max_ray_error_below_1e-3_m=" << ray_close << "\n"
              << "contact_mismatches=" << contact_mismatch << " alive_mismatches=" << alive_mismatch
              << " visibility_mismatches=" << visible_mismatch << "\n";
    const bool pass = position < 1e-3 && orientation < 1e-3 && alive_mismatch == 0 && contact_mismatch == 0 &&
                      visible_mismatch <= ticks / 1000 && ray_flips <= rays / 10000;
    std::cout << (pass ? "PARITY PASS" : "PARITY FAIL") << "\n";
    return pass ? 0 : 2;
}

static int bench(const std::string &missions_dir, uint32_t ticks_per_batch, bool with_actor, uint32_t repeats) {
    Gpu gpu{"physics.metal", "raptor.metal", "rlv2/voxel_world.h", "rlv2/nav_env.metal"};
    MissionSet set = load_missions_with_paths(missions_dir);
    GpuMissions missions(gpu, set);
    id<MTLBuffer> raptor = raptor_buffer(gpu);
    RLPhysicsParams params = rl_physics_crazyflie_default();
    id<MTLBuffer> params_buffer = gpu.buffer(sizeof(params), &params);
    const uint32_t obs = 52, hidden = 128;
    std::vector<float> actor;
    std::mt19937 rng(7);
    auto layer = [&](uint32_t in, uint32_t out) {
        std::uniform_real_distribution<float> u(-1 / std::sqrt(float(in)), 1 / std::sqrt(float(in)));
        for (uint32_t i = 0; i < in * out; i++)
            actor.push_back(u(rng));
        for (uint32_t i = 0; i < out; i++)
            actor.push_back(0);
    };
    layer(obs, hidden);
    layer(hidden, hidden);
    layer(hidden, 4);
    id<MTLBuffer> actor_buffer = gpu.buffer(actor.size() * 4, actor.data());
    size_t bits_bytes = set.bits.size() * 4;
    std::cout << "device=" << gpu.device.name.UTF8String << " missions=" << missions.count
              << " voxel_bytes=" << bits_bytes << " actor_params=" << actor.size()
              << " ticks_per_batch=" << ticks_per_batch << " actor=" << (with_actor ? "on" : "off") << "\n";
    std::cout << "N,batch_wall_ms,env_ticks_per_s,substeps_per_s,episodes,contacts,arrivals,mean_speed_mps\n";
    for (uint32_t n : {128u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u}) {
        NavConfig c{n, missions.count, 0, 0, 1.5f, .75f, 1.0f, 1200};
        id<MTLBuffer> config = gpu.buffer(sizeof(c), &c);
        id<MTLBuffer> envs = gpu.buffer(size_t(n) * sizeof(NavEnv));
        id<MTLBuffer> depth = gpu.buffer(size_t(n) * VX_RAYS * 4);
        id<MTLBuffer> actions = gpu.buffer(size_t(n) * 16);
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        gpu.dispatch(cb, "nav_reset_kernel", n, {envs, missions.missions, raptor, params_buffer, config});
        gpu.finish(cb);
        double total = 0;
        for (uint32_t repeat = 0; repeat < repeats; repeat++) {
            const double start = now();
            cb = [gpu.queue commandBuffer];
            for (uint32_t t = 0; t < ticks_per_batch; t++) {
                gpu.dispatch(cb, "nav_depth", size_t(n) * VX_RAYS, {envs, missions.worlds, missions.bits, depth, config});
                if (with_actor)
                    gpu.dispatch(cb, "nav_act", n,
                                 {envs, depth, missions.missions, missions.goals, actor_buffer, actions, config});
                gpu.dispatch(cb, "nav_step", n,
                             {envs, missions.worlds, missions.bits, actions, config, missions.missions,
                              missions.goals, raptor, params_buffer});
            }
            gpu.finish(cb);
            if (repeat)
                total += now() - start;
        }
        const NavEnv *e = static_cast<const NavEnv *>(envs.contents);
        uint64_t episodes = 0, contacts = 0, arrivals = 0;
        double speed = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float *v = e[i].physics.linear_velocity;
            speed += std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / n;
            episodes += e[i].episodes;
            contacts += e[i].contacts;
            arrivals += e[i].arrivals;
        }
        const double batch = total / std::max(repeats - 1, 1u);
        const double rate = double(n) * ticks_per_batch / batch;
        std::printf("%u,%.3f,%.0f,%.0f,%llu,%llu,%llu,%.3f\n", n, batch * 1e3, rate, rate * 5,
                    (unsigned long long)episodes, (unsigned long long)contacts, (unsigned long long)arrivals,
                    speed);
    }
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            const std::string command = argc > 1 ? argv[1] : "";
            if (command == "parity" && argc >= 4)
                return parity(argv[2], argv[3]);
            if (command == "bench" && argc >= 3)
                return bench(argv[2], argc > 3 ? std::stoul(argv[3]) : 8, !(argc > 4 && std::string(argv[4]) == "no-actor"),
                             argc > 5 ? std::stoul(argv[5]) : 4);
            std::cerr << "usage: rlv2 parity MISSIONS_DIR REFERENCE_BIN\n"
                         "       rlv2 bench MISSIONS_DIR [TICKS_PER_BATCH] [actor|no-actor] [REPEATS]\n";
            return 1;
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
}
