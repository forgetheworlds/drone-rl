// Policy training on Metal.
//
//   rlv2_train teacher OUT_DIR [ITERATIONS] [ENVS] [RESUME_CHECKPOINT]
//   rlv2_train student OUT_DIR TEACHER_CHECKPOINT [ITERATIONS] [ENVS] [RESUME_CHECKPOINT]
//   rlv2_train bench [SECONDS]   training-env throughput (sensors, observations, host copies)
//
// Missions come from the procedural generator (worldgen.hpp); drone-rl's missions are never
// used here. OUT_DIR receives the checkpoint and a CSV log.

#include "../physics.hpp"
#include "../raptor.hpp"
#include "nav_task.h"
#include "gpu.h"
#include "trainer.hpp"
#include "worldgen.hpp"
#include <iostream>
#include <random>

static_assert(sizeof(RaptorWeights) == 2084 * 4, "RAPTOR layout");

struct MetalEnv {
    Gpu gpu{"physics.metal", "raptor.metal", "rlv2/voxel_world.h", "rlv2/sensors.h", "rlv2/attitude.h", "rlv2/nav_task.h",
            "rlv2/nav_env.metal", "rlv2/train_env.metal"};
    GpuMissions missions;
    uint32_t count;
    id<MTLBuffer> envs, depth, teacher, student, actions, rewards, dones, config, params, weights;
    bool fresh = false;

    MetalEnv(const MissionSet &set, uint32_t n) : missions(gpu, set), count(n) {
        envs = gpu.buffer(sizeof(TrainEnv) * n);
        depth = gpu.buffer(size_t(n) * VX_RAYS * 4);
        teacher = gpu.buffer(size_t(n) * NT_TEACHER_OBS * 4);
        student = gpu.buffer(size_t(n) * NT_STUDENT_OBS * 4);
        actions = gpu.buffer(size_t(n) * NT_ACTIONS * 4);
        rewards = gpu.buffer(size_t(n) * 4);
        dones = gpu.buffer(size_t(n) * 4);
        NavTaskConfig c{2.0f, .75f, 1.5f, 1.0f, n, missions.count, 1, 1};
        config = gpu.buffer(sizeof(c), &c);
        const RLPhysicsParams p = rl_physics_crazyflie_default();
        params = gpu.buffer(sizeof(p), &p);
        RaptorWeights w;
        require(raptor_load_weights((std::string(SOURCE_DIR) + "/assets/raptor.bin").c_str(), &w), "RAPTOR load failed");
        weights = gpu.buffer(sizeof(w), &w);
    }
    NavTaskConfig &cfg() { return *static_cast<NavTaskConfig *>(config.contents); }
    void set_level(float level) { cfg().sensor_level = level; }
    void encode_observe(id<MTLCommandBuffer> cb) {
        gpu.dispatch(cb, "train_depth", size_t(count) * VX_RAYS, {envs, missions.worlds, missions.bits, depth, config}, 256);
        gpu.dispatch(cb, "train_observe", count,
                     {envs, missions.worlds, depth, missions.missions, missions.goals, teacher, student, config, missions.bits,
                      missions.paths});
    }
    void reset(uint32_t seed) {
        cfg().seed = seed;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        gpu.dispatch(cb, "train_reset", count, {envs, missions.missions, weights, params, config});
        encode_observe(cb);
        gpu.finish(cb);
        fresh = true;
    }
    void observe(float *t, float *s) {
        require(fresh, "observe() must follow reset() or step()");
        std::memcpy(t, teacher.contents, size_t(count) * NT_TEACHER_OBS * 4);
        std::memcpy(s, student.contents, size_t(count) * NT_STUDENT_OBS * 4);
        fresh = false;
    }
    // Steps and computes the next observation in one GPU submission.
    void step(const float *a, float *r, uint32_t *d) {
        std::memcpy(actions.contents, a, size_t(count) * NT_ACTIONS * 4);
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        gpu.dispatch(cb, "train_step", count,
                     {envs, missions.worlds, missions.bits, actions, config, missions.missions, missions.goals, weights,
                      params, rewards, dones});
        encode_observe(cb);
        gpu.finish(cb);
        std::memcpy(r, rewards.contents, size_t(count) * 4);
        std::memcpy(d, dones.contents, size_t(count) * 4);
        fresh = true;
    }
    EnvTotals totals() const {
        EnvTotals t;
        const TrainEnv *e = static_cast<const TrainEnv *>(envs.contents);
        for (uint32_t i = 0; i < count; i++) {
            t.episodes += e[i].episodes;
            t.contacts += e[i].contacts;
            t.arrivals += e[i].arrivals;
            t.completions += e[i].completions;
        }
        return t;
    }
};

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            const std::string mode = argc > 1 ? argv[1] : "";
            if (mode == "bench") {
                // The env exactly as PPO/DAgger drive it: one GPU submission per 50 ms tick
                // (step + 320-ray depth + sensor model + teacher/student observations), with the
                // host copies of actions, rewards, dones and observations. Random actions.
                const double seconds = argc > 2 ? std::stod(argv[2]) : 4.0;
                const MissionSet set = gen_missions(4096, 1);
                std::printf("N,env_ticks_per_s,rays_per_s,ms_per_tick,episodes,contacts,arrivals\n");
                for (uint32_t n : {1024u, 2048u, 4096u, 8192u, 16384u}) {
                    MetalEnv env(set, n);
                    env.set_level(1.0f);
                    env.reset(1);
                    std::vector<float> t(size_t(n) * NT_TEACHER_OBS), s(size_t(n) * NT_STUDENT_OBS),
                        a(size_t(n) * NT_ACTIONS), r(n);
                    std::vector<uint32_t> d(n);
                    std::mt19937 rng(3);
                    std::uniform_real_distribution<float> u(-1, 1);
                    auto tick = [&]() {
                        env.observe(t.data(), s.data());
                        for (float &x : a)
                            x = u(rng);
                        env.step(a.data(), r.data(), d.data());
                    };
                    for (int i = 0; i < 200; i++)
                        tick();
                    const double t0 = trainer_now();
                    uint64_t ticks = 0;
                    while (trainer_now() - t0 < seconds) {
                        tick();
                        ++ticks;
                    }
                    const double wall = trainer_now() - t0;
                    const EnvTotals tot = env.totals();
                    std::printf("%u,%.0f,%.0f,%.3f,%llu,%llu,%llu\n", n, n * ticks / wall, n * ticks * double(VX_RAYS) / wall,
                                wall / ticks * 1e3, (unsigned long long)tot.episodes, (unsigned long long)tot.contacts,
                                (unsigned long long)tot.arrivals);
                    std::fflush(stdout);
                }
                return 0;
            }
            if (mode == "teacher" && argc > 2) {
                const std::string out = argv[2];
                PpoConfig c;
                c.iterations = argc > 3 ? std::stoi(argv[3]) : c.iterations;
                const uint32_t envs = argc > 4 ? uint32_t(std::stoul(argv[4])) : 4096;
                c.checkpoint = out + "/teacher.bin";
                c.log = out + "/teacher.csv";
                const double t0 = trainer_now();
                const MissionSet set = gen_missions(4096, 1);
                std::printf("generated %zu missions in %.1fs\n", set.missions.size(), trainer_now() - t0);
                MetalEnv env(set, envs);
                train_teacher(env, c, argc > 5 ? argv[5] : "");
                return 0;
            }
            if (mode == "student" && argc > 3) {
                const std::string out = argv[2];
                DaggerConfig c;
                c.iterations = argc > 4 ? std::stoi(argv[4]) : c.iterations;
                const uint32_t envs = argc > 5 ? uint32_t(std::stoul(argv[5])) : 4096;
                c.checkpoint = out + "/student.bin";
                c.log = out + "/student.csv";
                const MissionSet set = gen_missions(4096, 2);
                MetalEnv env(set, envs);
                train_student(env, c, argv[3], argc > 6 ? argv[6] : "");
                return 0;
            }
            std::cerr << "usage: rlv2_train teacher OUT_DIR [ITERATIONS] [ENVS] [RESUME]\n"
                         "       rlv2_train student OUT_DIR TEACHER [ITERATIONS] [ENVS] [RESUME]\n"
                         "       rlv2_train bench [SECONDS]\n";
            return 2;
        } catch (const std::exception &e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
}
