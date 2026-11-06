#pragma once

// The training environment on CPU threads, running the Metal kernels compiled as C++. Slow, but
// it runs on any machine, which makes it handy for smoke tests away from the Mac.

#include "mlp.hpp"
#include "worldgen.hpp"
#include <chrono>
#include <functional>
#include <numeric>
#include <thread>
// RLV2_DEPTH_PRESET=measured: train under the measured stereo preset (depth_preset.h), as in
// rlv2_grade. Off by default; the Metal trainer never sees it.
static thread_local float g_preset[320];
static thread_local bool g_preset_on = false;
#include <cstdlib>
#include <string>
// RLV2_UNKNOWN_FEATURE=u: student feature for all-hole bins (default: same as far, 1.0).
static const float g_unknown_feature = std::getenv("RLV2_UNKNOWN_FEATURE") ? std::stof(std::getenv("RLV2_UNKNOWN_FEATURE")) : -1.0f;
#define NT_STUDENT_DEPTH(d) (g_unknown_feature >= 0 && (d) >= VX_MAX_RANGE - 1e-3f ? g_unknown_feature : nt_depth_feature(d))
#define NT_STEREO_RAY(rays, k, seed, p) (g_preset_on ? g_preset[k] : vx_stereo_ray(rays[k], k, seed, p))
namespace host {
#include "../physics.hpp"
#include "../raptor.hpp"
}
#include "cpu/metal_cpu.h"
#include "depth_preset.h"
#include "local_map.h"
#include "trainer.hpp"

struct CpuEnv {
    uint32_t count;
    std::vector<TrainEnv> envs;
    std::vector<VxWorld> worlds;
    std::vector<uint32_t> bits;
    std::vector<NavMission> missions;
    std::vector<float4> goals;
    std::vector<float> weights, depth, paths;
    std::vector<StudentRoute> routes;
    RLPhysicsParams params;
    NavTaskConfig config{};

    CpuEnv(const MissionSet &set, uint32_t n, const std::string &raptor_path) : count(n) {
        for (const Mission &m : set.missions) {
            worlds.push_back(m.world);
            NavMission x{};
            std::copy(m.start.begin(), m.start.end(), x.start);
            x.launch_yaw = m.launch_yaw;
            x.goal_offset = uint32_t(goals.size());
            x.goal_count = uint32_t(m.goals.size());
            for (const auto &g : m.goals)
                goals.push_back(float4(g[0], g[1], g[2], 0));
            paths.insert(paths.end(), m.paths.begin(), m.paths.end());
            missions.push_back(x);
        }
        bits = set.bits;
        host::RaptorWeights w;
        if (!host::raptor_load_weights(raptor_path.c_str(), &w))
            throw std::runtime_error("Cannot load " + raptor_path);
        weights.resize(sizeof(w) / 4);
        std::memcpy(weights.data(), &w, sizeof(w));
        const auto p = host::rl_physics_crazyflie_default();
        static_assert(sizeof(p) == sizeof(RLPhysicsParams), "physics params layout");
        std::memcpy(&params, &p, sizeof(params));
        envs.resize(n);
        routes.resize(n);
        depth.resize(size_t(n) * VX_RAYS);
        config = {2.0f, .75f, 1.5f, 1.0f, n, uint32_t(missions.size()), 1, 1};
        // RLV2_ATT_EST=1: train with the IMU attitude estimate instead of the true attitude.
        config.attitude = std::getenv("RLV2_ATT_EST") && std::string(std::getenv("RLV2_ATT_EST")) == "1" ? 1 : 0;
    }
    template <typename F> void parallel(F f) {
        const unsigned t = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> pool;
        for (unsigned k = 0; k < t; k++)
            pool.emplace_back([&, k] {
                for (uint32_t i = k; i < count; i += t)
                    f(i);
            });
        for (auto &p : pool)
            p.join();
    }
    void set_level(float level) { config.sensor_level = level; }
    void reset(uint32_t seed) {
        config.seed = seed;
        parallel([&](uint32_t i) { train_reset(envs.data(), missions.data(), weights.data(), params, config, i); });
    }
    void observe(float *teacher, float *student) {
        parallel([&](uint32_t i) {
            for (uint32_t k = 0; k < VX_RAYS; k++)
                train_depth(envs.data(), worlds.data(), bits.data(), depth.data(), config, i * VX_RAYS + k);
            static const bool preset = dp_preset_on();
            g_preset_on = preset;
            if (preset && envs[i].tick % envs[i].sensor.depth_period == 0)
                preset_frame(i);
            train_observe(envs.data(), worlds.data(), depth.data(), missions.data(), goals.data(), teacher, student,
                          config, bits.data(), paths.data(), i);
            const TrainEnv &e = envs[i];
            const float4 g = goals[missions[e.mission].goal_offset + e.goal];
            sr_update(routes[i], e, &depth[size_t(i) * VX_RAYS], vx_vec(g.x, g.y, g.z),
                      student + size_t(i) * NT_STUDENT_OBS);
        });
    }
    // The measured-preset stereo frame for env i at this tick, into g_preset (this thread).
    void preset_frame(uint32_t i) {
        static const DepthPreset dp = dp_preset_from_env();
        static const bool despeckle = std::getenv("RLV2_PRESET_DESPECKLE") && std::string(std::getenv("RLV2_PRESET_DESPECKLE")) == "1";
        const TrainEnv &e = envs[i];
        const VxWorld &w = worlds[e.mission];
        float rotation[9];
        vx_rotation(e.physics.orientation_wxyz, rotation);
        const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
        const uint32_t salt = vx_seed(e.mission * 2654435761u + 77u, 0x7e47u);
        const uint32_t frame = vx_seed(vx_seed(i * 7919u + 0xd5u, e.episodes), e.tick);
        for (uint32_t k = 0; k < VX_RAYS; k++) {
            const VxVec dir = vx_rotate(rotation, vx_camera_ray(k, .75f));
            const float d = depth[size_t(i) * VX_RAYS + k];
            const bool mover_hit = d < VX_MAX_RANGE - 1e-3f && w.mover_count > 0 &&
                                   vx_ray_static(w, bits.data(), p, dir, d + .1f) > d + .05f;
            g_preset[k] = dp_dead(k, dp) ? VX_MAX_RANGE : dp_ray(d, dp_classify(w, bits.data(), p, dir, d, mover_hit, salt, dp.textureless), vx_seed(frame, k), dp,
                                dp.blob > 1 ? dp_blob_seed(frame, k, dp) : 0);
        }
        if (despeckle)
            dp_despeckle(g_preset);
    }
    void step(const float *actions, float *rewards, uint32_t *dones) {
        parallel([&](uint32_t i) {
            train_step(envs.data(), worlds.data(), bits.data(), actions, config, missions.data(), goals.data(),
                       weights.data(), params, rewards, dones, i);
        });
    }
    EnvTotals totals() const {
        EnvTotals t;
        for (const auto &e : envs) {
            t.episodes += e.episodes;
            t.contacts += e.contacts;
            t.arrivals += e.arrivals;
            t.completions += e.completions;
        }
        return t;
    }
};
