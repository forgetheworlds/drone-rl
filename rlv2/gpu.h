#pragma once

// Metal device, pipelines and mission upload shared by the rlv2 tools.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "missions.hpp"
#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

struct NavMission {
    float start[3], launch_yaw;
    uint32_t goal_offset, goal_count, pad0, pad1;
};

static void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static std::string read_file(const std::string &path) {
    std::ifstream f(path);
    require(bool(f), "Cannot read " + path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Gpu {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;

    // `files` are paths relative to the repository root, joined in order. Headers lose their
    // `#pragma once` and local includes, since everything is one translation unit.
    explicit Gpu(std::initializer_list<const char *> files) {
        require(device != nil, "No Metal device");
        queue = [device newCommandQueue];
        std::string source;
        for (const char *file : files) {
            std::stringstream in(read_file(std::string(SOURCE_DIR) + "/" + file));
            for (std::string line; std::getline(in, line);)
                if (line.rfind("#pragma once", 0) != 0 && line.rfind("#include \"", 0) != 0)
                    source += line + "\n";
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError *error = nil;
        library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                       options:options
                                         error:&error];
        require(library != nil, error ? error.localizedDescription.UTF8String : "Metal compile failed");
    }
    id<MTLComputePipelineState> pipeline(const std::string &name) {
        auto found = pipelines.find(name);
        if (found != pipelines.end())
            return found->second;
        id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        require(fn != nil, "Missing kernel " + name);
        NSError *error = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
        require(p != nil, error ? error.localizedDescription.UTF8String : "Pipeline failed");
        return pipelines[name] = p;
    }
    id<MTLBuffer> buffer(size_t size, const void *data = nullptr) {
        id<MTLBuffer> b = [device newBufferWithLength:std::max(size, size_t(16))
                                              options:MTLResourceStorageModeShared];
        require(b != nil, "Buffer allocation failed");
        if (data)
            std::memcpy(b.contents, data, size);
        return b;
    }
    void dispatch(id<MTLCommandBuffer> cb, const std::string &name, size_t count,
                  std::initializer_list<id<MTLBuffer>> buffers, size_t group = 64) {
        id<MTLComputePipelineState> p = pipeline(name);
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        uint32_t index = 0;
        for (id<MTLBuffer> b : buffers)
            [e setBuffer:b offset:0 atIndex:index++];
        [e dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(std::min(group, size_t(p.maxTotalThreadsPerThreadgroup)), 1, 1)];
        [e endEncoding];
    }
    void finish(id<MTLCommandBuffer> cb) {
        [cb commit];
        [cb waitUntilCompleted];
        require(cb.status != MTLCommandBufferStatusError,
                cb.error ? cb.error.localizedDescription.UTF8String : "GPU failure");
    }
};

// Uploads the worlds, the occupancy bits, mission starts and goals.
struct GpuMissions {
    id<MTLBuffer> worlds, bits, missions, goals, paths;
    uint32_t count;
    GpuMissions(Gpu &gpu, const MissionSet &set) {
        std::vector<VxWorld> worlds_host;
        std::vector<NavMission> missions_host;
        std::vector<float> goals_host, paths_host;
        for (const Mission &m : set.missions) {
            worlds_host.push_back(m.world);
            NavMission n{};
            std::copy(m.start.begin(), m.start.end(), n.start);
            n.launch_yaw = m.launch_yaw;
            n.goal_offset = uint32_t(goals_host.size() / 4);
            n.goal_count = uint32_t(m.goals.size());
            for (const auto &g : m.goals)
                goals_host.insert(goals_host.end(), {g[0], g[1], g[2], 0});
            require(m.paths.size() == m.goals.size() * NT_PATH * 4, "Mission " + m.name + " has no paths");
            paths_host.insert(paths_host.end(), m.paths.begin(), m.paths.end());
            missions_host.push_back(n);
        }
        count = uint32_t(set.missions.size());
        worlds = gpu.buffer(worlds_host.size() * sizeof(VxWorld), worlds_host.data());
        bits = gpu.buffer(set.bits.size() * 4, set.bits.data());
        missions = gpu.buffer(missions_host.size() * sizeof(NavMission), missions_host.data());
        goals = gpu.buffer(goals_host.size() * 4, goals_host.data());
        paths = gpu.buffer(paths_host.size() * 4, paths_host.data());
    }
};
