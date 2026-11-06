#pragma once

// Loads drone-rl mission files (.task) and voxel maps (.3dmap) into VxWorld form.
// The test missions are for evaluation only. Never train on them.

#include "voxel_world.h"
#include <algorithm>
#include <cmath>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Mission {
    std::string name, portfolio, map_path;
    VxWorld world{};
    std::array<float, 3> start{};
    float launch_yaw = 0;
    std::vector<std::array<float, 3>> goals;
    std::vector<float> paths; // teacher reference paths, see paths.hpp
    // 3D missions (gen3d.hpp): the obstacle between start and first waypoint, for the
    // over/around count, and boxes the teacher's reference path must not know about
    // (clutter hidden behind a blocker). All in metres; empty for drone-rl missions.
    bool has_blocker = false;
    std::array<float, 6> blocker{};
    std::vector<std::array<float, 6>> hidden;
};

// The map the teacher's reference path is planned on: the real map minus hidden boxes, so the
// path cannot route around clutter nobody has seen yet. Returns bits with offset 0.
inline std::vector<uint32_t> mission_plan_bits(const Mission &m, const uint32_t *bits) {
    const uint64_t cells = uint64_t(m.world.nx) * m.world.ny * m.world.nz;
    std::vector<uint32_t> out(bits + m.world.bits_offset, bits + m.world.bits_offset + (cells + 31) / 32);
    const float s = m.world.scale;
    for (const auto &h : m.hidden)
        for (int z = std::max(0, int(h[2] / s)); z < std::min(int(m.world.nz), int(std::ceil(h[5] / s))); z++)
            for (int y = std::max(0, int(h[1] / s)); y < std::min(int(m.world.ny), int(std::ceil(h[4] / s))); y++)
                for (int x = std::max(0, int(h[0] / s)); x < std::min(int(m.world.nx), int(std::ceil(h[3] / s))); x++) {
                    const uint64_t i = (uint64_t(z) * m.world.ny + y) * m.world.nx + x;
                    out[i / 32] &= ~(1u << (i % 32));
                }
    return out;
}

struct MissionSet {
    std::vector<Mission> missions;
    std::vector<uint32_t> bits;
};

inline void load_voxel_map(const std::string &path, VxWorld &world, std::vector<uint32_t> &bits) {
    std::ifstream file(path);
    std::string magic;
    int nx = 0, ny = 0, nz = 0;
    file >> magic >> nx >> ny >> nz;
    if (magic != "voxel" || nx <= 0 || ny <= 0 || nz <= 0 ||
        uint64_t(nx) * ny * nz > 600000000)
        throw std::runtime_error("Bad voxel map " + path);
    world.nx = nx;
    world.ny = ny;
    world.nz = nz;
    world.bits_offset = uint32_t(bits.size());
    const uint64_t cells = uint64_t(nx) * ny * nz;
    bits.resize(bits.size() + (cells + 31) / 32, 0);
    int x, y, z;
    while (file >> x >> y >> z) {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
            throw std::runtime_error("Bad occupied voxel in " + path);
        const uint64_t index = (uint64_t(z) * ny + y) * nx + x;
        bits[world.bits_offset + index / 32] |= 1u << (index % 32);
    }
}

// Reads one .task file. Maps already in the set are shared, not loaded twice.
inline Mission load_mission(const std::filesystem::path &task_path, MissionSet &set,
                            std::map<std::string, VxWorld> &maps) {
    std::ifstream task(task_path);
    Mission m;
    m.name = task_path.stem().string();
    m.portfolio = task_path.parent_path().filename().string();
    float scale;
    int goal_count;
    task >> m.map_path >> scale >> m.start[0] >> m.start[1] >> m.start[2] >> goal_count;
    if (!task || goal_count < 1 || goal_count > 32)
        throw std::runtime_error("Invalid task " + task_path.string());
    m.goals.resize(goal_count);
    for (auto &goal : m.goals)
        task >> goal[0] >> goal[1] >> goal[2];
    if (std::filesystem::path(m.map_path).is_relative())
        m.map_path = (task_path.parent_path() / m.map_path).string();
    auto found = maps.find(m.map_path);
    if (found == maps.end()) {
        VxWorld world{};
        load_voxel_map(m.map_path, world, set.bits);
        found = maps.emplace(m.map_path, world).first;
    }
    m.world = found->second;
    m.world.scale = scale;
    int movers;
    task >> movers;
    if (movers < 0 || movers > VX_MAX_MOVERS)
        throw std::runtime_error("Invalid movers in " + task_path.string());
    m.world.mover_count = movers;
    for (int i = 0; i < movers; i++) {
        VxMover &box = m.world.movers[i];
        int axis;
        task >> box.center[0] >> box.center[1] >> box.center[2] >> box.extent[0] >> box.extent[1] >>
            box.extent[2] >> axis >> box.amplitude >> box.period >> box.phase;
        if (box.period <= 0 || axis < 0 || axis > 2)
            throw std::runtime_error("Invalid mover in " + task_path.string());
        box.axis = uint32_t(axis);
    }
    if (!task)
        throw std::runtime_error("Truncated task " + task_path.string());
    std::string trailing;
    while (task >> trailing) {
        if (trailing == "yaw") {
            if (!(task >> m.launch_yaw) || !std::isfinite(m.launch_yaw))
                throw std::runtime_error("Invalid launch heading in " + task_path.string());
        } else if (trailing == "blocker") {
            m.has_blocker = true;
            for (float &v : m.blocker)
                task >> v;
        } else if (trailing == "hidden") {
            std::array<float, 6> h;
            for (float &v : h)
                task >> v;
            m.hidden.push_back(h);
        } else
            throw std::runtime_error("Unknown entry " + trailing + " in " + task_path.string());
        if (!task)
            throw std::runtime_error("Truncated entry in " + task_path.string());
    }
    return m;
}

// Loads every mission in drone-rl's missions/ folder, in the same order as run_missions.py.
// Other folders (generated 3D sets, one per scenario kind) follow in name order.
inline MissionSet load_missions(const std::filesystem::path &missions_dir) {
    MissionSet set;
    std::map<std::string, VxWorld> maps;
    std::vector<std::string> portfolios;
    for (const char *p : {"source", "public", "fresh"})
        if (std::filesystem::is_directory(missions_dir / p))
            portfolios.push_back(p);
    std::vector<std::string> extra;
    for (const auto &entry : std::filesystem::directory_iterator(missions_dir))
        if (entry.is_directory() && std::find(portfolios.begin(), portfolios.end(), entry.path().filename().string()) == portfolios.end())
            extra.push_back(entry.path().filename().string());
    std::sort(extra.begin(), extra.end());
    portfolios.insert(portfolios.end(), extra.begin(), extra.end());
    for (const auto &portfolio : portfolios) {
        std::vector<std::filesystem::path> tasks;
        for (const auto &entry : std::filesystem::directory_iterator(missions_dir / portfolio))
            if (entry.path().extension() == ".task")
                tasks.push_back(entry.path());
        std::sort(tasks.begin(), tasks.end());
        for (const auto &task : tasks)
            set.missions.push_back(load_mission(task, set, maps));
    }
    if (set.missions.empty())
        throw std::runtime_error("No missions found in " + missions_dir.string());
    return set;
}
