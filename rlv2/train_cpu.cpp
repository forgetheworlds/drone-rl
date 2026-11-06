// Teacher or student training on CPU threads.
//
//   rlv2_train_cpu RAPTOR_BIN ENVS ITERATIONS [MISSIONS] [TEACHER_CHECKPOINT] [OUT_PREFIX] [RESUME]
//   rlv2_train_cpu RAPTOR_BIN ENVS ITERATIONS MISSIONS --student-ppo STUDENT_CHECKPOINT OUT_PREFIX
//   rlv2_train_cpu RAPTOR_BIN ENVS ITERATIONS MISSIONS --teacher-resume TEACHER_CHECKPOINT OUT_PREFIX
//
// RLV2_MIX3D=f makes a fraction f of the training missions 3D/blocker missions (gen3d.hpp).
//
// With a teacher checkpoint it runs DAgger for the student instead of PPO. With --student-ppo
// it fine-tunes a DAgger student with PPO. The student needs
// this CPU build: its route target comes from a local map (local_map.h) built on the host.

#include "cpu_env.hpp"
#include "gen3d.hpp"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: rlv2_train_cpu RAPTOR_BIN ENVS ITERATIONS [MISSIONS] [TEACHER] [OUT_PREFIX] [RESUME]\n";
        return 2;
    }
    const MissionSet set = gen_missions_mix(argc > 4 ? std::stoi(argv[4]) : 256, 1);
    CpuEnv env(set, std::stoi(argv[2]), argv[1]);
    if (argc > 7 && std::string(argv[5]) == "--student-ppo") {
        PpoConfig c;
        c.iterations = std::stoi(argv[3]);
        c.hidden = 256;
        c.lr = 1e-4f;
        c.level_ramp = 20;
        c.save_every = 10;
        c.checkpoint = std::string(argv[7]) + ".bin";
        c.log = std::string(argv[7]) + ".csv";
        train_student_ppo(env, c, argv[6]);
        return 0;
    }
    if (argc > 7 && std::string(argv[5]) == "--teacher-resume") {
        PpoConfig c;
        c.iterations = std::stoi(argv[3]);
        c.lr = 1e-4f;
        c.level_start = 1;
        c.level_ramp = 1;
        c.save_every = 5;
        c.checkpoint = std::string(argv[7]) + ".bin";
        c.log = std::string(argv[7]) + ".csv";
        if (FILE *f = std::fopen(c.log.c_str(), "w")) {  // train_teacher appends when resuming
            std::fprintf(f, "iteration,steps,seconds,level,reward,episodes,completion,contact,timeout,arrivals,"
                            "value_loss,clip_fraction,std\n");
            std::fclose(f);
        }
        train_teacher(env, c, argv[6]);
        return 0;
    }
    if (argc > 5) {
        DaggerConfig d;
        d.iterations = std::stoi(argv[3]);
        d.beta_iterations = 20;
        const std::string out = argc > 6 ? argv[6] : "student_cpu";
        d.checkpoint = out + ".bin";
        d.log = out + ".csv";
        train_student(env, d, argv[5], argc > 7 ? argv[7] : "");
        return 0;
    }
    PpoConfig c;
    c.iterations = std::stoi(argv[3]);
    c.save_every = 5;
    c.checkpoint = "teacher_cpu.bin";
    c.log = "teacher_cpu.csv";
    train_teacher(env, c);
}
