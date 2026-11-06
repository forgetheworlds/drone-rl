// Writes a held-out 3D mission set: OUT/<kind>/g3d-<kind>-<seed>.task + .3dmap.
//   gen3d_write OUT_DIR PER_KIND BASE_SEED
// Training uses base seed 1 (+500009); use a different base (e.g. 900001) for tests.
#include "gen3d.hpp"
#include <filesystem>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: gen3d_write OUT_DIR PER_KIND BASE_SEED\n");
        return 2;
    }
    const std::string out = argv[1];
    const int per = std::stoi(argv[2]);
    const uint32_t base = std::stoul(argv[3]);
    for (int k = 0; k < G3D_KINDS; k++) {
        std::filesystem::create_directories(out + "/" + g3d_kinds[k]);
        int blockers_moving = 0, two = 0;
        for (int i = 0; i < per; i++) {
            Mission m;
            std::vector<uint32_t> bits;
            gen3d_fill(base + uint32_t(k) * 1000003u + uint32_t(i) * 7919u, k, m, bits);
            blockers_moving += m.world.mover_count > 0;
            two += m.goals.size() > 1;
            gen_write(m, bits, out + "/" + g3d_kinds[k] + "/" + m.name);
        }
        std::printf("%s: %d missions, %d with a walking person, %d with two waypoints\n", g3d_kinds[k], per,
                    blockers_moving, two);
    }
    std::printf("goal tries %llu: placed %llu, rejected outside vertical field %llu, occluded/other %llu\n",
                (unsigned long long)g3d_stats.tries.load(), (unsigned long long)g3d_stats.placed.load(),
                (unsigned long long)g3d_stats.frustum.load(), (unsigned long long)g3d_stats.occluded.load());
}
