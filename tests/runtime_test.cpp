// Smoke: load RAPTOR weights, step one physics tick, render depth from a public map.
#include "navigation/vehicle.hpp"
#include "navigation/visible_world.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static void require(bool ok, const char *msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    require(argc >= 3, "usage: runtime_test RAPTOR_BIN VOXEL_MAP");
    const char *weights = argv[1];
    const char *map_path = argv[2];

    safe_nav::Vehicle vehicle(weights, {1.5f, 1.5f, 1.2f}, 0.f);
    safe_nav::Command hover{};
    vehicle.substep(hover);
    auto pose = vehicle.pose();
    require(std::isfinite(pose.position.x) && std::isfinite(pose.position.z), "pose finite");

    VisibleWorld world(map_path);
    auto depth = world.depth(pose);
    require(depth.size() == 320, "320 depth rays");
    float min_d = depth[0], max_d = depth[0];
    for (float d : depth) {
        require(std::isfinite(d) && d > 0, "depth positive finite");
        min_d = std::min(min_d, d);
        max_d = std::max(max_d, d);
    }
    require(max_d >= min_d, "depth range");

    std::printf("runtime_test ok (RAPTOR + VisibleWorld) min_depth=%.2f max_depth=%.2f\n", min_d,
                max_d);
    return 0;
}
