// Grades a trained policy on drone-rl's missions with drone-rl's own simulator: VisibleWorld for
// depth and contact, raptor_motor_command and rl_physics_step for the vehicle. RAPTOR is fed
// the drifting state estimate, as in training. Observations, sensors and scoring come from the
// same nav_task.h the trainer uses.
//
//   rlv2_grade MISSIONS_DIR RAPTOR_BIN CHECKPOINT [teacher|student|route] [SENSOR_LEVEL] [SEEDS] [CSV]
//
// The policy is deterministic (the mean action). Each mission is flown SEEDS times with
// different sensor draws. `route` flies the student's prior alone, with no network.

// RLV2_DEPTH_PRESET=measured: the student's stereo (pooled bins, shield, local map) comes from
// depth_preset.h instead of vx_stereo_ray. Stereo odometry keeps the old model.
#include <cstdio>
#include <cstdint>
static thread_local float g_preset[320];
static thread_local bool g_preset_on = false;
// RLV2_EXT_DEPTH=DIR: stereo depth from an external renderer instead of the preset
// (hyperdrone_eval/ext_depth_server.py, real StereoBM on rendered images). Per stereo frame,
// worker thread t writes 8 floats (position xyz in map metres, orientation wxyz, mission index)
// to the FIFO DIR/req<t> and reads 320 ray ranges (12 = no return) from DIR/resp<t>.
// RLV2_THREADS sets the worker count (one renderer process per FIFO pair).
static thread_local unsigned g_tid = 0;
static thread_local std::FILE *g_ext_req = nullptr, *g_ext_resp = nullptr;
#include <cstdlib>
#include <string>
// RLV2_UNKNOWN_FEATURE=u: student feature for all-hole bins (default: same as far, 1.0).
static const float g_unknown_feature = std::getenv("RLV2_UNKNOWN_FEATURE") ? std::stof(std::getenv("RLV2_UNKNOWN_FEATURE")) : -1.0f;
#define NT_STUDENT_DEPTH(d) (g_unknown_feature >= 0 && (d) >= VX_MAX_RANGE - 1e-3f ? g_unknown_feature : nt_depth_feature(d))
#define NT_STEREO_RAY(rays, k, seed, p) (g_preset_on ? g_preset[k] : vx_stereo_ray(rays[k], k, seed, p))
#include "depth_preset.h"
#include "missions.hpp"
#include "mlp.hpp"
#include "navigation/raptor_model.hpp"
#include "navigation/visible_world.hpp"
#include "local_map.h"
#include "wall_hold.h"
#include "init_scan.h"
#include "paths.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

struct Flight {
    uint32_t mission, seed, goals, arrived, contact, complete, ticks, waypoint_ticks;
    uint32_t scan_ticks = 0, scan_contact = 0;
    float scan_drift = 0;  // estimate error when the scan ends, m
    std::vector<float> errors;  // true distance to the waypoint at each arrival, m
    float tilt_sum = 0, tilt_max = 0, yaw_sum = 0, yaw_max = 0;  // attitude estimate error, rad
    float position_error = 0;  // mean distance between estimated and true position, m
    uint64_t surfaces[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // depth_preset.h rays by DpSurface
    float cap_sum = 0;  // plan-ahead: allowed horizontal speed summed over moving ticks
    uint32_t cap_ticks = 0;
    int contact_kind = -1;  // 0 in view, 1 side (45-135 deg), 2 behind, 3 floor, 4 ceiling, 5 box
    float contact_speed = 0;  // true horizontal speed at contact, m/s
    // Blocker missions: how the drone first passed the blocker's middle plane (x), 0 never,
    // 1 over its top, 2 around it; the most it climbed above the start before that, m.
    int cross = 0;
    float climb = 0, cross_z = 0;
    // RLV2_DIAG=1 (failure breakdown, written to CSV.diag): distances to the current waypoint (from
    // where its leg started, the closest reached, at the end); the last 10 s: true displacement,
    // mean command speed before / after the grade-side guards, nearest true obstacle; at contact:
    // of the rays whose true range was < 0.8 m in the last stereo frame, the share stereo returned
    // within tolerance, and whether the local map held an occupied cell along the velocity (0.6 m).
    float min_clear = 9, leg_dist = 0, min_dist = 1e9f, end_dist = 0, disp10 = 0, pre10 = 0, post10 = 0, clear10 = 9;
    int near_rays = 0, near_seen = 0, map_ahead = -1;
    int near_textureless = 0, near_floor = 0;  // of near_rays: on textureless wall tiles, floor/ceiling
    float path_len = 0;  // true path flown, m
    float mover_gap = 9;  // at contact: distance from the body centre to the nearest moving box's surface, m
};

struct Policy {
    Mlp actor, critic;
    Normalizer norm;
    std::vector<float> log_std;
    bool student = false, route = false;
    void load(const std::string &path) {
        Checkpoint c{{&actor, &critic}, &norm, &log_std};
        if (student)
            c.nets = {&actor};
        c.load(path);
    }
};

// The speed shield is on unless RLV2_SHIELD=0.
static uint32_t nt_grade_shield() {
    const char *v = std::getenv("RLV2_SHIELD");
    return v && std::string(v) == "0" ? 0u : 1u;
}

// Confirmed-free distance along a horizontal direction: march the local map from 0.3 m (the
// drone's own space) until a cell not seen free, on the centre line and 0.2 m either side,
// out to 4 m. Then the fastest speed that still stops inside it: 0.3 m margin, 0.4 s of
// frame age and reaction, 1.5 m/s^2 braking.
static float pa_allowed(const LocalMap &map, VxVec p, float ux, float uy) {
    float free = 4.0f;
    for (float s = .3f; s < 4.0f && free >= 4.0f; s += .1f)
        for (float lateral : {-.2f, 0.0f, .2f}) {
            if (s < .5f && lateral != 0)
                continue;
            const VxVec q = vx_vec(p.x + ux * s - uy * lateral, p.y + uy * s + ux * lateral, p.z);
            const int i = lm_index(map, q);
            if (i < 0 || map.odds[i] >= 0) {
                free = s;
                break;
            }
        }
    const float a = 1.5f, t = .4f, room = std::max(0.0f, free - .3f);
    return a * (-t + std::sqrt(t * t + 2 * room / a));
}

static Flight fly(const Mission &m, const uint32_t *bits, const RaptorWeights &weights, const Policy &policy,
                  float level, uint32_t mission_index, uint32_t seed) {
    VisibleWorld world(m.map_path);
    world.scale = m.world.scale;
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
    auto dynamics = rl_physics_crazyflie_default();
    // RLV2_DYN_RAND=1: per-flight vehicle randomization for a 30-50 g brushed 1S micro quad (assumed ranges):
    // mass x U(1.0, 1.3), thrust (battery sag, worn props) x U(0.85, 1.0), motor spin-up lag x U(1, 2),
    // spin-down lag x U(1, 1.5). RAPTOR's hover estimate below uses the nominal values (it does not know).
    const float nominal_hover_mass = dynamics.mass;
    if (std::getenv("RLV2_DYN_RAND") && std::string(std::getenv("RLV2_DYN_RAND")) == "1") {
        uint32_t ds = vx_seed(mission_index * 104729u + 7, seed ^ 0xd1u);
        dynamics.mass *= 1.0f + .3f * vx_uniform(ds);
        const float kt = .85f + .15f * vx_uniform(ds), up = 1 + vx_uniform(ds), down = 1 + .5f * vx_uniform(ds);
        for (int i = 0; i < 12; i++)
            dynamics.rotor_thrust_coefficients[i] *= kt;
        for (int r = 0; r < 4; r++)
            dynamics.rotor_time_constants_rising[r] *= up, dynamics.rotor_time_constants_falling[r] *= down;
    }
    const auto nominal = rl_physics_crazyflie_default();
    const float c0 = nominal.rotor_thrust_coefficients[0], c1 = nominal.rotor_thrust_coefficients[1],
                c2 = nominal.rotor_thrust_coefficients[2], thrust = nominal_hover_mass * 9.81f / 4;
    const float hover = (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);

    TrainEnv e{};
    e.rng = vx_seed(mission_index * 7919u + 13, seed);
    nt_reset(e, m.start.data(), m.launch_yaw, uint32_t(m.goals.size()), hover, level);
    // RLV2_ATT_EST=0 gives the student and RAPTOR the true attitude (attitude.h); on by default.
    // RLV2_ATT_VO: what stereo odometry corrects. 0 nothing (yaw drifts with the gyro), 1 yaw,
    // 2 the whole attitude.
    static const bool att_est = !(std::getenv("RLV2_ATT_EST") && std::string(std::getenv("RLV2_ATT_EST")) == "0");
    static const uint32_t vo_yaw = std::getenv("RLV2_ATT_VO") ? std::stoul(std::getenv("RLV2_ATT_VO")) : 1;
    if (att_est)
        va_begin(e.attitude, vx_seed(mission_index * 7919u + 0xa77u, seed), level, e.sensor.gyro_noise,
                 e.sensor.accel_noise, e.sensor.vo_pixel, vo_yaw, e.physics.position,
                 e.physics.linear_velocity);
    // RLV2_ATT_COMP=0: the filter takes the accelerometer for gravity alone (attitude.h).
    static const bool att_comp = !(std::getenv("RLV2_ATT_COMP") && std::string(std::getenv("RLV2_ATT_COMP")) == "0");
    e.attitude.compensate = att_est && att_comp ? 1 : 0;
    // RLV2_ATT_KP: the filter's proportional gain (twoKp in sensfusion6, 0.8 there).
    static const float two_kp = std::getenv("RLV2_ATT_KP") ? std::stof(std::getenv("RLV2_ATT_KP")) : VA_TWO_KP;
    e.attitude.two_kp = two_kp;
    // RLV2_ATT_KF_WALK: how fast the velocity filter lets its accelerometer bias move, on top of
    // the true bias walk, to soak up the attitude error (m/s^2 per sqrt(s)).
    static const float kf_walk = std::getenv("RLV2_ATT_KF_WALK") ? std::stof(std::getenv("RLV2_ATT_KF_WALK")) : 0;
    if (att_est)
        e.estimate.extra_walk = kf_walk;
    // RLV2_ARRIVE_EST=1: the drone declares arrival from its own estimate (nt_finish_tick).
    static const bool arrive_est = std::getenv("RLV2_ARRIVE_EST") && std::string(std::getenv("RLV2_ARRIVE_EST")) == "1";
    e.arrive_estimate = arrive_est ? 1 : 0;
    safe_nav::RaptorControlState control;
    control.reference = {m.start[0], m.start[1], m.start[2]};
    control.reference_yaw = m.launch_yaw;
    raptor_reset(weights, control.hidden);
    const NavTaskConfig config{2.0f, .75f, 1.5f, level, 1, 1, 0, nt_grade_shield()};

    // Measured stereo preset and plan-ahead speed cap (docs/DEPTH.md).
    static const bool preset = dp_preset_on();
    static const DepthPreset dp = dp_preset_from_env();
    // RLV2_PLAN_AHEAD=1: cap horizontal speed so the drone can stop inside the stretch of its
    // local map seen free along the commanded direction. RLV2_PLAN_CREEP: speed allowed anyway.
    static const bool plan_ahead = std::getenv("RLV2_PLAN_AHEAD") && std::string(std::getenv("RLV2_PLAN_AHEAD")) == "1";
    static const float creep = std::getenv("RLV2_PLAN_CREEP") ? std::stof(std::getenv("RLV2_PLAN_CREEP")) : 0;
    g_preset_on = preset;
    // RLV2_PRESET_DESPECKLE=1: drop isolated stereo returns before the shield and the map.
    // RLV2_SIDE_GUARD=1: side clearance from occupied map cells (see the loop).
    static const bool side_guard = std::getenv("RLV2_SIDE_GUARD") && std::string(std::getenv("RLV2_SIDE_GUARD")) == "1";
    // RLV2_MEM_GUARD=1: short-term obstacle memory guard in 3D. Obstacles the local map saw in
    // the last RLV2_MEM_FRAMES stereo frames (default 30, ~3 s), including ones now outside
    // the view (beside, behind, above, below), cap the command toward them to what still stops
    // 0.3 m short, and push away inside 0.4 m. Covers floor/ceiling/lintel contacts that the
    // horizontal side guard does not. The floor is left to the height hold (it is not mapped).
    static const bool mem_guard = std::getenv("RLV2_MEM_GUARD") && std::string(std::getenv("RLV2_MEM_GUARD")) == "1";
    static const int mem_frames = std::getenv("RLV2_MEM_FRAMES") ? std::stoi(std::getenv("RLV2_MEM_FRAMES")) : 30;
    // RLV2_LAT_CAP=v: cap on sideways / backward command speed (m/s), 0 = off.
    static const float lat_cap = std::getenv("RLV2_LAT_CAP") ? std::stof(std::getenv("RLV2_LAT_CAP")) : 0;
    static const bool despeckle = std::getenv("RLV2_PRESET_DESPECKLE") && std::string(std::getenv("RLV2_PRESET_DESPECKLE")) == "1";
    const uint32_t texture_salt = vx_seed(mission_index * 2654435761u + 77u, 0x7e47u);
    uint64_t preset_counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float cap_sum = 0;
    uint32_t cap_ticks = 0;
    Mlp actor = policy.actor;
    auto route = std::make_unique<StudentRoute>();
    sr_reset(*route);
    // RLV2_WALL_HOLD=1: hold distance to the walls when slow (wall_hold.h), student modes only.
    static const bool wall_hold = std::getenv("RLV2_WALL_HOLD") && std::string(std::getenv("RLV2_WALL_HOLD")) == "1";
    auto hold = std::make_unique<WallHold>();
    wh_reset(*hold);
    // Start: climb 0.5 m and hover 1 s before the policy takes over (init_scan.h).
    // RLV2_INIT_SCAN=0 sets off at once.
    static const bool init_scan = !(std::getenv("RLV2_INIT_SCAN") && std::string(std::getenv("RLV2_INIT_SCAN")) == "0");
    // RLV2_INIT_SCAN_CLIMB / _TURNS / _SETTLE change the climb (m), turns and final hover (s),
    // to split what each part does.
    static const bool scan_tuned = [] {
        is_turns = 0;
        is_settle = 1;
        if (const char *v = std::getenv("RLV2_INIT_SCAN_CLIMB"))
            is_climb = std::stof(v);
        if (const char *v = std::getenv("RLV2_INIT_SCAN_TURNS"))
            is_turns = std::stof(v);
        if (const char *v = std::getenv("RLV2_INIT_SCAN_SETTLE"))
            is_settle = std::stof(v);
        return true;
    }();
    (void)scan_tuned;
    InitScan scan{};
    scan.phase = 3;
    if (init_scan)
        is_reset(scan, m.start.data(), m.launch_yaw);
    // RLV2_MOVER_WAIT=SECONDS: before setting off, hover while a moving box is within 0.6 m of
    // the first 3 m of the path to the first waypoint, now or up to 2 s ahead, for at most
    // SECONDS. Uses the true box motion, so it is an upper bound on what a box tracker gives.
    static const float mover_wait = std::getenv("RLV2_MOVER_WAIT") ? std::stof(std::getenv("RLV2_MOVER_WAIT")) : 0;
    auto corridor_busy = [&](float time) {
        const float *path = &m.paths[0];
        for (uint32_t i = 0; i < m.world.mover_count; i++) {
            const VxMover &v = m.world.movers[i];
            for (float ahead : {0.0f, 1.0f, 2.0f}) {
                const VxVec c = vx_mover_center(v, time + ahead);
                for (int k = 0; k < NT_PATH && path[3] - path[k * 4 + 3] <= 3.0f; k++) {
                    const float p[3] = {path[k * 4], path[k * 4 + 1], path[k * 4 + 2]}, cc[3] = {c.x, c.y, c.z};
                    float d2 = 0;
                    for (int a = 0; a < 3; a++) {
                        const float gap = std::fabs(p[a] - cc[a]) - v.extent[a];
                        d2 += gap > 0 ? gap * gap : 0;
                    }
                    if (d2 < .36f)
                        return true;
                }
            }
        }
        return false;
    };
    Flight f{mission_index, seed, uint32_t(m.goals.size()), 0, 0, 0, 0, 0};
    uint32_t goal_start = 0;
    static const bool diag = std::getenv("RLV2_DIAG") && std::string(std::getenv("RLV2_DIAG")) == "1";
    std::vector<std::array<float, 6>> hist;
    // Round 7 behaviours (all from the drone's own stereo frame and state estimate, no ground truth):
    // RLV2_BLANK_GUARD=1: when the stereo view toward the command is mostly blank (no return and not
    //   "free to range"), cap speed toward it to RLV2_BLANK_SPEED (0.3 m/s) and dither yaw
    //   (+-RLV2_BLANK_YAW rad/s, 0.5) so edges / texture come into view.
    // RLV2_HOLE_MEM=1: blank rays leave "suspect" points at 0.35-0.7 m along them (unless the local
    //   map has seen that cell free); the memory guard caps motion toward suspect points younger than
    //   RLV2_HOLE_MEM_FRAMES (20) frames, beside and behind too, down to a 0.15 m/s creep.
    // RLV2_PERSIST=1: if the estimated position moved < 0.3 m in 3 s while a waypoint is pending,
    //   try escape moves in turn (climb, slide left, slide right, drop; 1.5 s each, 0.4 m/s) on top of
    //   the policy command, still under every guard.
    static const bool blank_guard = std::getenv("RLV2_BLANK_GUARD") && std::string(std::getenv("RLV2_BLANK_GUARD")) == "1";
    static const bool hole_mem = std::getenv("RLV2_HOLE_MEM") && std::string(std::getenv("RLV2_HOLE_MEM")) == "1";
    static const bool persist = std::getenv("RLV2_PERSIST") && std::string(std::getenv("RLV2_PERSIST")) == "1";
    static const float blank_speed = std::getenv("RLV2_BLANK_SPEED") ? std::stof(std::getenv("RLV2_BLANK_SPEED")) : .3f;
    static const float blank_yaw = std::getenv("RLV2_BLANK_YAW") ? std::stof(std::getenv("RLV2_BLANK_YAW")) : .5f;
    static const float vlm_lat = std::getenv("RLV2_VLM_LAT") ? std::stof(std::getenv("RLV2_VLM_LAT")) : 0.f;
    uint32_t vlm_goal = 0, vlm_until = 0;
    static const float yaw_scan = std::getenv("RLV2_YAW_SCAN") ? std::stof(std::getenv("RLV2_YAW_SCAN")) : 0.f;
    static const float yaw_scan_period = std::getenv("RLV2_YAW_SCAN_PERIOD") ? std::stof(std::getenv("RLV2_YAW_SCAN_PERIOD")) : 4.f;
    static const float blank_frac = std::getenv("RLV2_BLANK_FRAC") ? std::stof(std::getenv("RLV2_BLANK_FRAC")) : .7f;
    static const int hole_frames = std::getenv("RLV2_HOLE_MEM_FRAMES") ? std::stoi(std::getenv("RLV2_HOLE_MEM_FRAMES")) : 20;
    float blank_pending[320], blank_frame[320];
    std::fill(blank_frame, blank_frame + 320, 0.0f);
    uint32_t blank_capture = ~0u, frame_no = 0;
    float blank_q[4] = {1, 0, 0, 0};
    VxVec blank_p = vx_vec(0, 0, 0);
    bool blank_ready = false;
    float blank_sector[5] = {0, 0, 0, 0, 0};  // blank share per 4-column sector, rows 3-12
    std::vector<std::pair<VxVec, uint32_t>> suspects;  // point, frame number
    std::vector<VxVec> est_hist;
    uint32_t escape_tick = 0, escape_phase = 0, blank_ticks = 0;
    float path_len = 0;  // per tick: true x y z, pre speed, post speed, clearance
    uint32_t diag_goal = ~0u;
    float last_cmd[2] = {0, 0};
    while (true) {
        world.time = e.steps * .01f;
        safe_nav::Pose pose;
        pose.position = {e.physics.position[0], e.physics.position[1], e.physics.position[2]};
        raptor_quaternion_matrix(e.physics.orientation_wxyz, pose.rotation.data());
        std::array<float, 320> depth;
        if (g_vx_cam.wide) {  // RLV2_HFOV: render the wide lens ourselves
            for (uint32_t k = 0; k < 320; k++) {
                const VxVec cr = vx_camera_ray(k, safe_nav::sensor_slope);
                depth[k] = world.ray(pose.position, safe_nav::rotate(pose, {cr.x, cr.y, cr.z}));
            }
        } else {
            depth = world.depth(pose);
        }
        const auto &g = m.goals[e.goal];
        float mover[6] = {0, 0, 0, 0, 0, 0};
        float best = 1e9f;
        for (uint32_t i = 0; i < m.world.mover_count; i++) {
            const VxMover &v = m.world.movers[i];
            const VxVec c = vx_mover_center(v, world.time);
            const float d = vx_norm(vx_sub(c, vx_vec(pose.position.x, pose.position.y, pose.position.z)));
            if (d < best) {
                best = d;
                const float w0 = 6.28318530718f / v.period, speed = -v.amplitude * w0 * std::sin(w0 * world.time + v.phase);
                mover[0] = c.x;
                mover[1] = c.y;
                mover[2] = c.z;
                for (uint32_t j = 0; j < 3; j++)
                    mover[3 + j] = v.axis == j ? speed : 0;
            }
        }
        float teacher[NT_TEACHER_OBS], student[NT_STUDENT_OBS];
        float proximity[NT_PROXIMITY];
        nt_proximity(m.world, bits, vx_vec(pose.position.x, pose.position.y, pose.position.z),
                     nt_yaw(e.physics.orientation_wxyz), proximity);
        const float *path = &m.paths[size_t(e.goal) * NT_PATH * 4];
        if (preset && e.tick % e.sensor.depth_period == 0) {
            const VxVec p = vx_vec(pose.position.x, pose.position.y, pose.position.z);
            const uint32_t frame = vx_seed(vx_seed(mission_index * 7919u + 0xd5u, seed), e.tick);
            static const char *ext = std::getenv("RLV2_EXT_DEPTH");
            if (ext && !g_ext_req) {
                g_ext_req = std::fopen((std::string(ext) + "/req" + std::to_string(g_tid)).c_str(), "wb");
                g_ext_resp = std::fopen((std::string(ext) + "/resp" + std::to_string(g_tid)).c_str(), "rb");
            }
            if (ext) {
                // pose, mission, world velocity, body rates (for motion blur / rolling shutter / sync
                // offset in the camera-effects stage), seed, tick
                const float q[16] = {pose.position.x, pose.position.y, pose.position.z, e.physics.orientation_wxyz[0],
                                     e.physics.orientation_wxyz[1], e.physics.orientation_wxyz[2], e.physics.orientation_wxyz[3],
                                     float(mission_index), e.physics.linear_velocity[0], e.physics.linear_velocity[1],
                                     e.physics.linear_velocity[2], e.physics.angular_velocity_body[0],
                                     e.physics.angular_velocity_body[1], e.physics.angular_velocity_body[2], float(seed),
                                     float(e.tick)};
                std::fwrite(q, sizeof(float), 16, g_ext_req);
                std::fflush(g_ext_req);
                if (std::fread(g_preset, sizeof(float), 320, g_ext_resp) != 320)
                    std::abort();
            }
            for (uint32_t k = 0; k < 320 && !ext; k++) {
                const VxVec cr = vx_camera_ray(k, safe_nav::sensor_slope);
                const auto wd = safe_nav::rotate(pose, {cr.x, cr.y, cr.z});
                const VxVec w = vx_vec(wd.x, wd.y, wd.z);
                const float d = depth[k];
                bool mover_hit = false;
                if (d < VX_MAX_RANGE - 1e-3f && m.world.mover_count > 0)
                    mover_hit = vx_ray_static(m.world, bits, p, w, d + .1f) > d + .05f;
                const DpSurface c = dp_classify(m.world, bits, p, w, d, mover_hit, texture_salt, dp.textureless);
                preset_counts[c]++;
                g_preset[k] = dp_dead(k, dp) ? VX_MAX_RANGE : dp_ray(d, c, vx_seed(frame, k), dp, dp.blob > 1 ? dp_blob_seed(frame, k, dp) : 0);
            }
        }
        if (preset && despeckle && e.tick % e.sensor.depth_period == 0)
            dp_despeckle(g_preset);
        if ((blank_guard || hole_mem) && preset) {
            if (e.tick % e.sensor.depth_period == 0) {
                std::copy(g_preset, g_preset + 320, blank_pending);
                blank_capture = e.tick;
                std::copy(nt_attitude(e), nt_attitude(e) + 4, blank_q);
                blank_p = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
            }
            // the frame reaches the guard after the same latency as the policy's
            if (blank_capture != ~0u && e.tick - blank_capture == e.sensor.depth_latency) {
                std::copy(blank_pending, blank_pending + 320, blank_frame);
                blank_ready = true;
                frame_no++;
                for (int sct = 0; sct < 5; sct++) {
                    int n = 0, b = 0;
                    for (int r = 3; r <= 12; r++)
                        for (int c = sct * 4; c < sct * 4 + 4; c++) {
                            const int k = r * 20 + c;
                            if (preset && dp_dead(k, dp))
                                continue;
                            n++;
                            b += blank_frame[k] >= VX_MAX_RANGE;
                        }
                    blank_sector[sct] = n ? float(b) / n : 0;
                }
                if (hole_mem) {
                    float rq[9];
                    vx_rotation(blank_q, rq);
                    for (int k = 0; k < 320; k++) {
                        if (blank_frame[k] < VX_MAX_RANGE || dp_dead(k, dp) || blank_sector[(k % 20) / 4] < blank_frac)
                            continue;
                        const VxVec w = vx_rotate(rq, vx_camera_ray(uint32_t(k), safe_nav::sensor_slope));
                        for (float d : {.35f, .7f}) {
                            const VxVec q = vx_add(blank_p, vx_scale(w, d));
                            if (q.z < .25f)
                                continue;  // the floor is the height hold's
                            const int i = lm_index(route->map, q);
                            if (i >= 0 && route->map.odds[i] < 0)
                                continue;  // seen free
                            suspects.push_back({q, frame_no});
                        }
                    }
                    size_t j = 0;
                    for (size_t i = 0; i < suspects.size(); i++)
                        if (frame_no - suspects[i].second < uint32_t(hole_frames))
                            suspects[j++] = suspects[i];
                    suspects.resize(j);
                    if (suspects.size() > 4000)
                        suspects.erase(suspects.begin(), suspects.begin() + (suspects.size() - 4000));
                }
            }
        }
        nt_observe(e, depth.data(), vx_vec(g[0], g[1], g[2]), mover, m.world.mover_count > 0, proximity, path, teacher,
                   student);
        float prior[NT_ACTIONS] = {0, 0, 0, 0};
        if ((plan_ahead || side_guard) && !policy.student) {  // the map alone, for the caps
            float scratch[NT_STUDENT_OBS];
            sr_update(*route, e, depth.data(), vx_vec(g[0], g[1], g[2]), scratch);
        }
        if (policy.student) {
            sr_update(*route, e, depth.data(), vx_vec(g[0], g[1], g[2]), student);
            nt_student_prior(student, prior);
        } else {
            nt_teacher_prior(teacher, prior);
        }
        float raw[NT_TEACHER_OBS], student_raw[NT_STUDENT_OBS];
        std::copy(teacher, teacher + NT_TEACHER_OBS, raw);
        std::copy(student, student + NT_STUDENT_OBS, student_raw);
        float *obs = policy.student ? student : teacher;
        policy.norm.apply(obs, 1);
        const float *output = actor.forward(obs, 1);
        float action[NT_ACTIONS], command[4];
        if (policy.student && !policy.route)
            nt_student_combine(prior, output, action);
        else
            for (int j = 0; j < NT_ACTIONS; j++)
                action[j] = prior[j] + (policy.route ? 0 : output[j]);
        if (!policy.student)
            nt_teacher_certify(raw, action);
        else
            nt_student_certify(student_raw, action);

        nt_command(e, action, config, command);
        if (persist && scan.phase == 3) {
            const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
            est_hist.push_back(pe);
            if (escape_tick > 0) {
                escape_tick--;
                const VxVec to = vx_sub(vx_vec(g[0], g[1], g[2]), pe);
                const float h = std::max(1e-3f, std::hypot(to.x, to.y)), ux = to.x / h, uy = to.y / h;
                const float v = .4f;
                if (escape_phase % 4 == 0)
                    command[2] += v;
                else if (escape_phase % 4 == 1)
                    command[0] += -uy * v, command[1] += ux * v;
                else if (escape_phase % 4 == 2)
                    command[0] += uy * v, command[1] += -ux * v;
                else
                    command[2] -= v;
                command[0] += ux * .15f, command[1] += uy * .15f;
                if (escape_tick == 0)
                    escape_phase++, est_hist.clear();
            } else if (est_hist.size() >= 60 && vx_norm(vx_sub(pe, est_hist[est_hist.size() - 60])) < .3f &&
                       vx_norm(vx_sub(pe, vx_vec(g[0], g[1], g[2]))) > .4f) {
                escape_tick = 30;
            }
        }
        const float pre_speed = std::sqrt(command[0] * command[0] + command[1] * command[1] + command[2] * command[2]);
        if (plan_ahead) {
            const float speed = std::hypot(command[0], command[1]);
            if (speed > 1e-3f) {
                const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
                const float allowed = std::max(creep, pa_allowed(route->map, pe, command[0] / speed, command[1] / speed));
                if (speed > allowed) {
                    command[0] *= allowed / speed;
                    command[1] *= allowed / speed;
                }
                cap_sum += std::min(speed, allowed);
                cap_ticks++;
            }
        }
        if (lat_cap > 0) {
            // Blind-side cap: the camera sees ~±33 deg. The command's sideways part (and any
            // backward part) is limited to lat_cap m/s; forward motion is untouched.
            const float yaw = nt_yaw(nt_attitude(e)), fx = std::cos(yaw), fy = std::sin(yaw);
            float along = command[0] * fx + command[1] * fy;
            float lx = command[0] - along * fx, ly = command[1] - along * fy;
            const float lat = std::hypot(lx, ly);
            if (lat > lat_cap)
                lx *= lat_cap / lat, ly *= lat_cap / lat;
            if (along < -lat_cap)
                along = -lat_cap;
            command[0] = along * fx + lx;
            command[1] = along * fy + ly;
        }
        if (side_guard) {
            // Side clearance from the map: on 16 headings, the nearest cell the map holds
            // occupied (odds > 0) within 0.9 m. The command component toward it is capped to
            // what still stops in front of it, and inside 0.45 m the drone is pushed away.
            const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
            for (int a = 0; a < 16; a++) {
                const float az = a * 6.28318530718f / 16, ux = std::cos(az), uy = std::sin(az);
                float d = 9;
                for (float s = .2f; s < .9f && d > 8; s += .1f) {
                    const int i = lm_index(route->map, vx_vec(pe.x + ux * s, pe.y + uy * s, pe.z));
                    if (i >= 0 && route->map.odds[i] > 0)
                        d = s;
                }
                if (d > 8)
                    continue;
                const float room = std::max(0.0f, d - .3f), ok = 1.5f * (-.4f + std::sqrt(.16f + 2 * room / 1.5f));
                const float toward = command[0] * ux + command[1] * uy;
                float cut = toward > ok ? toward - ok : 0;
                if (d < .45f)
                    cut += .3f * (.45f - d) / .25f;
                command[0] -= cut * ux;
                command[1] -= cut * uy;
            }
        }
        if (mem_guard) {
            const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
            static std::vector<VxVec> dirs = [] {
                std::vector<VxVec> d;
                for (int a = 0; a < 16; a++)
                    d.push_back(vx_vec(std::cos(a * .392699f), std::sin(a * .392699f), 0));
                for (float el : {-.6f, .6f})
                    for (int a = 0; a < 8; a++)
                        d.push_back(vx_vec(std::cos(el) * std::cos(a * .785398f), std::cos(el) * std::sin(a * .785398f), std::sin(el)));
                d.push_back(vx_vec(0, 0, 1));
                d.push_back(vx_vec(0, 0, -1));
                return d;
            }();
            for (const VxVec &u : dirs) {
                float d = 9;
                for (float s = .2f; s < 1.0f && d > 8; s += .1f) {
                    const VxVec q = vx_add(pe, vx_scale(u, s));
                    const int i = lm_index(route->map, q);
                    if (i >= 0 && route->map.odds[i] >= LM_OCCUPIED && route->map.age[i] < mem_frames)
                        d = std::min(d, s);
                }
                if (d > 8)
                    continue;
                const float room = std::max(0.0f, d - .3f), ok = 1.5f * (-.4f + std::sqrt(.16f + 2 * room / 1.5f));
                const float toward = command[0] * u.x + command[1] * u.y + command[2] * u.z;
                float cut = toward > ok ? toward - ok : 0;
                if (d < .4f)
                    cut += .3f * (.4f - d) / .2f;
                command[0] -= cut * u.x;
                command[1] -= cut * u.y;
                command[2] -= cut * u.z;
            }
        }
        if (blank_guard && blank_ready) {
            // the commanded direction in the camera's horizontal field (estimated yaw)
            const float yaw = nt_yaw(nt_attitude(e)), sp = std::hypot(command[0], command[1]);
            bool blank = false;
            if (sp > blank_speed) {
                const float rel = std::remainder(std::atan2(command[1], command[0]) - yaw, 6.28318530718f);
                const float half = g_vx_cam.half_h;
                if (std::fabs(rel) < half) {
                    // column 0 is the left edge (+y); sectors of 4 columns
                    const int col = std::min(19, std::max(0, int((half - rel) / (2 * half) * 20)));
                    blank = blank_sector[col / 4] >= blank_frac;
                }
            }
            if (blank) {
                command[0] *= blank_speed / sp;
                command[1] *= blank_speed / sp;
                command[3] += (blank_ticks / 30) % 2 ? -blank_yaw : blank_yaw;
                blank_ticks++;
            } else {
                blank_ticks = 0;
            }
        }
        if (hole_mem && !suspects.empty()) {
            const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
            for (int a = 0; a < 16; a++) {
                const VxVec u = vx_vec(std::cos(a * .392699f), std::sin(a * .392699f), 0);
                float d = 9;
                for (const auto &sp : suspects) {
                    const VxVec r = vx_sub(sp.first, pe);
                    const float along = r.x * u.x + r.y * u.y + r.z * u.z;
                    if (along < .1f || along > .8f)
                        continue;
                    const VxVec perp = vx_sub(r, vx_scale(u, along));
                    if (vx_norm(perp) < .2f)
                        d = std::min(d, along);
                }
                if (d > 8)
                    continue;
                const float room = std::max(0.0f, d - .3f);
                const float ok = std::max(.15f, 1.5f * (-.4f + std::sqrt(.16f + 2 * room / 1.5f)));
                const float toward = command[0] * u.x + command[1] * u.y;
                if (toward > ok) {
                    command[0] -= (toward - ok) * u.x;
                    command[1] -= (toward - ok) * u.y;
                }
            }
        }
        if (vlm_lat > 0) {
            // RLV2_VLM_LAT=S: the next waypoint arrives U(0.3, S) s after the last one was reached (off-board
            // VLM over Wi-Fi); until then the drone holds (zero velocity command).
            if (e.goal != vlm_goal) {
                uint32_t vs = vx_seed(e.goal * 31u + 5u, e.rng);
                vlm_goal = e.goal, vlm_until = e.tick + uint32_t((.3f + (vlm_lat - .3f) * vx_uniform(vs)) / .05f);
                if (e.goal == 0)
                    vlm_until = 0;
            }
            if (e.tick < vlm_until)
                command[0] = command[1] = command[2] = command[3] = 0;
        }
        if (yaw_scan > 0 && std::hypot(command[0], command[1]) > .2f) {
            // RLV2_YAW_SCAN=A: slow heading sweep while translating (+-A*P/2pi rad, period P s), so a
            // narrow (65 deg) stereo pair sees the sides it is passing; velocity is in the world frame.
            command[3] += yaw_scan * std::sin(6.28318530718f * e.tick * .05f / yaw_scan_period);
        }
        if (wall_hold && policy.student)
            wh_update(*hold, e, depth.data(), e.tick % e.sensor.depth_period == 0, command);
        bool scanning = false;
        if (mover_wait > 0 && e.tick == f.scan_ticks && e.tick * .05f < mover_wait && corridor_busy(world.time)) {
            scanning = true;
            command[0] = command[1] = command[2] = command[3] = 0;
            f.scan_ticks++;
        } else if (scan.phase != 3) {
            scanning = is_step(scan, e.estimate.position, nt_yaw(nt_attitude(e)), command);
            if (scanning) {
                f.scan_ticks++;
            } else {
                e.goal_tick = 0;
                f.scan_drift = vx_norm(vx_sub(vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]),
                                              vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2])));
            }
        }
        if (diag) {
            const VxVec pt = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
            if (!hist.empty())
                path_len += vx_norm(vx_sub(pt, vx_vec(hist.back()[0], hist.back()[1], hist.back()[2])));
            const float dg = vx_norm(vx_sub(pt, vx_vec(g[0], g[1], g[2])));
            if (e.goal != diag_goal)
                diag_goal = e.goal, f.leg_dist = dg, f.min_dist = dg;
            f.min_dist = std::min(f.min_dist, dg);
            f.end_dist = dg;
            float cl = 9;
            if (e.tick % 4 == 0) {
                for (int a = 0; a < 16; a++) {
                    const float az = a * .392699f;
                    cl = std::min(cl, vx_ray_static(m.world, bits, pt, vx_vec(std::cos(az), std::sin(az), 0), 2));
                }
                cl = std::min(cl, std::min(vx_ray_static(m.world, bits, pt, vx_vec(0, 0, 1), 2), vx_ray_static(m.world, bits, pt, vx_vec(0, 0, -1), 2)));
                f.min_clear = std::min(f.min_clear, cl);
            } else if (!hist.empty()) {
                cl = hist.back()[5];
            }
            hist.push_back({pt.x, pt.y, pt.z, pre_speed, std::sqrt(command[0] * command[0] + command[1] * command[1] + command[2] * command[2]), cl});
        }
        safe_nav::Command cmd;
        cmd.world_velocity = {command[0], command[1], command[2]};
        last_cmd[0] = command[0], last_cmd[1] = command[1];
        cmd.yaw_rate = command[3];
        bool contact = false;
        for (int k = 0; k < 5 && !contact; k++) {
            const RLPhysicsState measured = nt_measure(e);
            if (k == 0)
                f.position_error += vx_norm(vx_sub(vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]),
                                                   vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2])));
            if (e.attitude.on && k == 0) {
                float tilt, yaw_error;
                va_error(e.attitude, e.physics.orientation_wxyz, tilt, yaw_error);
                f.tilt_sum += tilt;
                f.tilt_max = std::max(f.tilt_max, tilt);
                f.yaw_sum += std::fabs(yaw_error);
                f.yaw_max = std::max(f.yaw_max, std::fabs(yaw_error));
            }
            float motors[4];
            if (!safe_nav::raptor_motor_command(weights, control, measured, cmd, .01f, motors)) {
                contact = true;
                break;
            }
            RLPhysicsState next{};
            const float wind[3] = {0, 0, 0};
            rl_physics_step(e.physics, motors, wind, dynamics, next);
            e.physics = next;
            ++e.steps;
            world.time = e.steps * .01f;
            contact = world.contact({e.physics.position[0], e.physics.position[1], e.physics.position[2]});
        }
        const uint32_t goal_before = e.goal;
        nt_finish_tick(e, contact, vx_vec(g[0], g[1], g[2]));
        if (m.has_blocker && f.cross == 0) {
            const float px = e.physics.position[0], pz = e.physics.position[2];
            f.climb = std::max(f.climb, pz - m.start[2]);
            if (px >= .5f * (m.blocker[0] + m.blocker[3])) {
                f.cross = pz > m.blocker[5] ? 1 : 2;
                f.cross_z = pz;
            }
        }
        if (e.goal != goal_before) {
            f.errors.push_back(e.arrive_error);
            f.arrived++;
            f.waypoint_ticks += e.tick - goal_start;
            goal_start = e.tick;
        }
        if (e.done && contact) {
            const VxVec p = vx_vec(e.physics.position[0], e.physics.position[1], e.physics.position[2]);
            f.contact_speed = std::hypot(e.physics.linear_velocity[0], e.physics.linear_velocity[1]);
            const float yaw = nt_yaw(e.physics.orientation_wxyz);
            float best = vx_ray_static(m.world, bits, p, vx_vec(0, 0, -1), 2), az_best = 0;
            int kind = 3;
            const float up = vx_ray_static(m.world, bits, p, vx_vec(0, 0, 1), 2);
            if (up < best)
                best = up, kind = 4;
            for (int a = 0; a < 72; a++) {
                const float az = a * 6.28318530718f / 72;
                const float d = vx_ray_static(m.world, bits, p, vx_vec(std::cos(az), std::sin(az), 0), 2);
                if (d < best)
                    best = d, az_best = az, kind = 0;
            }
            if (kind == 0) {
                float rel = std::fabs(std::remainder(az_best - yaw, 6.28318530718f));
                kind = rel < .785f ? 0 : rel < 2.36f ? 1 : 2;
            }
            if (best > .3f)  // nothing static near: a moving box
                kind = 5;
            f.contact_kind = kind;
            if (diag)
                for (uint32_t i = 0; i < m.world.mover_count; i++) {
                    const VxMover &v = m.world.movers[i];
                    const VxVec c = vx_mover_center(v, world.time);
                    const float q[3] = {p.x - c.x, p.y - c.y, p.z - c.z};
                    float d2 = 0;
                    for (int a = 0; a < 3; a++) {
                        const float gap = std::fabs(q[a]) - v.extent[a];
                        d2 += gap > 0 ? gap * gap : 0;
                    }
                    f.mover_gap = std::min(f.mover_gap, std::sqrt(d2));
                }
            if (diag && preset) {
                for (int k = 0; k < 320; k++)
                    if (depth[k] < .8f) {
                        f.near_rays++;
                        const VxVec cr = vx_camera_ray(uint32_t(k), safe_nav::sensor_slope);
                        const auto wd = safe_nav::rotate(pose, {cr.x, cr.y, cr.z});
                        const VxVec p0 = vx_vec(pose.position.x, pose.position.y, pose.position.z);
                        const DpSurface sc = dp_classify(m.world, bits, p0, vx_vec(wd.x, wd.y, wd.z), depth[k], false, texture_salt, dp.textureless);
                        f.near_textureless += sc == DP_WALL_TEXTURELESS;
                        f.near_floor += sc == DP_FLOOR || sc == DP_CEILING;
                        f.near_seen += std::fabs(g_preset[k] - depth[k]) <= std::max(.1f, .15f * depth[k]);
                    }
                const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
                const float vx = e.physics.linear_velocity[0], vy = e.physics.linear_velocity[1], vz = e.physics.linear_velocity[2];
                const float sp = std::max(1e-3f, std::sqrt(vx * vx + vy * vy + vz * vz));
                f.map_ahead = 0;
                for (float s = 0; s < .6f; s += .05f) {
                    const int i = lm_index(route->map, vx_vec(pe.x + vx / sp * s, pe.y + vy / sp * s, pe.z + vz / sp * s));
                    if (i >= 0 && route->map.odds[i] >= LM_OCCUPIED)
                        f.map_ahead = 1;
                }
            }
            if (std::getenv("RLV2_CONTACT_DEBUG")) {
                const VxVec pe = vx_vec(e.estimate.position[0], e.estimate.position[1], e.estimate.position[2]);
                const float vx = e.physics.linear_velocity[0], vy = e.physics.linear_velocity[1];
                const float sp = std::max(1e-3f, std::hypot(vx, vy));
                std::string line;
                for (float s = 0; s < 1.0f; s += .1f) {
                    const int i = lm_index(route->map, vx_vec(pe.x + vx / sp * s, pe.y + vy / sp * s, pe.z));
                    line += std::to_string(i < 0 ? 99 : int(route->map.odds[i])) + " ";
                }
                std::fprintf(stderr, "contact %s kind=%d speed=%.2f wall=%.2f cmd=(%.2f,%.2f) allowed=%.2f odds-along-v: %s\n",
                             m.name.c_str(), kind, sp, best, last_cmd[0], last_cmd[1],
                             pa_allowed(route->map, pe, vx / sp, vy / sp), line.c_str());
            }
        }
        if (e.done) {
            f.contact = contact;
            f.scan_contact = contact && scanning;
            f.complete = e.goal >= e.goal_count;
            f.ticks = e.tick;
            for (int c = 0; c < 8; c++)
                f.surfaces[c] = preset_counts[c];
            f.cap_sum = cap_sum;
            f.cap_ticks = cap_ticks;
            f.path_len = path_len;
            if (diag && !hist.empty()) {
                const size_t n = hist.size(), a0 = n > 200 ? n - 200 : 0;
                f.disp10 = std::sqrt(std::pow(hist[n - 1][0] - hist[a0][0], 2) + std::pow(hist[n - 1][1] - hist[a0][1], 2) +
                                     std::pow(hist[n - 1][2] - hist[a0][2], 2));
                for (size_t i = a0; i < n; i++)
                    f.pre10 += hist[i][3] / (n - a0), f.post10 += hist[i][4] / (n - a0), f.clear10 = std::min(f.clear10, hist[i][5]);
            }
            return f;
        }
    }
}

int main(int argc, char **argv) {
    try {
        if (argc < 4)
            throw std::runtime_error("rlv2_grade MISSIONS_DIR RAPTOR_BIN CHECKPOINT [teacher|student] [SENSOR_LEVEL] [SEEDS] [CSV]");
        MissionSet set = load_missions(argv[1]);
        for (Mission &m : set.missions) {
            if (m.hidden.empty()) {
                m.paths = path_mission(m.world, set.bits.data(), m.start, m.goals);
            } else {  // the teacher's path does not know about clutter hidden behind a blocker
                VxWorld w = m.world;
                w.bits_offset = 0;
                const auto plan = mission_plan_bits(m, set.bits.data());
                m.paths = path_mission(w, plan.data(), m.start, m.goals);
            }
        }
        RaptorWeights weights;
        if (!raptor_load_weights(argv[2], &weights))
            throw std::runtime_error("Invalid RAPTOR weights");
        safe_nav::sensor_slope = .75f;
        Policy policy;
        const std::string mode = argc > 4 ? argv[4] : "teacher";
        policy.route = mode == "route";
        policy.student = mode == "student" || policy.route;
        if (policy.route)
            policy.actor = Mlp({NT_STUDENT_OBS, 4, NT_ACTIONS}, Activation::relu, 1, 0), policy.norm = Normalizer(NT_STUDENT_OBS);
        else
            policy.load(argv[3]);
        const float level = argc > 5 ? std::stof(argv[5]) : 1;
        const uint32_t seeds = argc > 6 ? std::stoul(argv[6]) : 4;
        // RLV2_SEED_BASE: first seed, for a second disjoint block of seeds.
        const uint32_t seed_base = std::getenv("RLV2_SEED_BASE") ? std::stoul(std::getenv("RLV2_SEED_BASE")) : 0;

        std::vector<Flight> flights(set.missions.size() * seeds);
        std::vector<std::thread> pool;
        const unsigned threads = std::getenv("RLV2_THREADS") ? unsigned(std::stoul(std::getenv("RLV2_THREADS")))
                                                              : std::max(1u, std::thread::hardware_concurrency());
        for (unsigned t = 0; t < threads; t++)
            pool.emplace_back([&, t] {
                g_tid = t;
                for (size_t i = t; i < flights.size(); i += threads)
                    flights[i] = fly(set.missions[i / seeds], set.bits.data(), weights, policy, level, uint32_t(i / seeds),
                                     uint32_t(i % seeds) + seed_base);
            });
        for (auto &p : pool)
            p.join();

        if (argc > 7) {
            std::FILE *csv = std::fopen(argv[7], "w");
            std::fprintf(csv, "mission,portfolio,seed,goals,arrived,complete,contact,seconds,scan_s,scan_contact,scan_drift,"
                              "waypoint_s,position_error,tilt_mean,tilt_max,yaw_mean,yaw_max,errors\n");
            for (const Flight &f : flights) {
                const double deg = 57.2958 / std::max(1u, f.ticks);
                std::fprintf(csv, "%s,%s,%u,%u,%u,%u,%u,%.2f,%.2f,%u,%.3f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f,",
                             set.missions[f.mission].name.c_str(), set.missions[f.mission].portfolio.c_str(), f.seed,
                             f.goals, f.arrived, f.complete, f.contact, f.ticks * .05, f.scan_ticks * .05, f.scan_contact,
                             f.scan_drift, f.waypoint_ticks * .05, f.position_error / std::max(1u, f.ticks), f.tilt_sum * deg, f.tilt_max * 57.2958,
                             f.yaw_sum * deg, f.yaw_max * 57.2958);
                for (size_t k = 0; k < f.errors.size(); k++)
                    std::fprintf(csv, "%s%.3f", k ? " " : "", f.errors[k]);
                std::fprintf(csv, "\n");
            }
            std::fclose(csv);
            if (std::getenv("RLV2_DIAG") && std::string(std::getenv("RLV2_DIAG")) == "1") {
                std::FILE *dg = std::fopen((std::string(argv[7]) + ".diag").c_str(), "w");
                std::fprintf(dg, "mission,seed,end,goal,goals,contact_kind,contact_speed,near_rays,near_seen,map_ahead,leg_dist,"
                                 "min_dist,end_dist,disp10,pre10,post10,clear10,cross,climb,min_clear,mover_gap,near_textureless,near_floor,path_len,seconds\n");
                for (const Flight &f : flights)
                    std::fprintf(dg, "%s,%u,%s,%u,%u,%d,%.2f,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%d,%d,%.2f,%.2f\n",
                                 set.missions[f.mission].name.c_str(), f.seed,
                                 f.complete ? "complete" : f.contact ? "contact" : "timeout", f.arrived, f.goals,
                                 f.contact_kind, f.contact_speed, f.near_rays, f.near_seen, f.map_ahead, f.leg_dist, f.min_dist,
                                 f.end_dist, f.disp10, f.pre10, f.post10, f.clear10, f.cross, f.climb, f.min_clear, f.mover_gap, f.near_textureless, f.near_floor, f.path_len, f.ticks * .05);
                std::fclose(dg);
            }
        }
        std::printf("set,flights,completion,contact_flights,waypoints_reached,seconds_per_waypoint\n");
        std::vector<std::string> portfolios;
        for (const Mission &m : set.missions)
            if (std::find(portfolios.begin(), portfolios.end(), m.portfolio) == portfolios.end())
                portfolios.push_back(m.portfolio);
        portfolios.push_back("all");
        for (const std::string &portfolio : portfolios) {
            uint32_t n = 0, complete = 0, contacts = 0, arrived = 0, goals = 0, wticks = 0;
            for (const Flight &f : flights) {
                if (portfolio != "all" && set.missions[f.mission].portfolio != portfolio)
                    continue;
                n++;
                complete += f.complete;
                contacts += f.contact;
                arrived += f.arrived;
                goals += f.goals;
                wticks += f.waypoint_ticks;
            }
            std::printf("%s,%u,%.3f,%u,%u/%u,%.1f\n", portfolio.c_str(), n, double(complete) / n, contacts, arrived, goals,
                        arrived ? wticks * .05 / arrived : 0.0);
        }
        {
            uint64_t surfaces[8] = {0, 0, 0, 0, 0, 0, 0, 0}, total = 0, movers = 0, mover_contacts = 0;
            double cap = 0;
            uint64_t cap_ticks = 0;
            for (const Flight &f : flights) {
                for (int c = 0; c < 8; c++)
                    surfaces[c] += f.surfaces[c], total += f.surfaces[c];
                cap += f.cap_sum;
                cap_ticks += f.cap_ticks;
                if (set.missions[f.mission].world.mover_count > 0) {
                    movers++;
                    mover_contacts += f.contact;
                }
            }
            for (const std::string &portfolio : portfolios) {
                uint32_t n = 0, over = 0, around = 0, never = 0, over_ok = 0, around_ok = 0, timeouts = 0;
                double climb = 0;
                for (const Flight &f : flights) {
                    const Mission &m = set.missions[f.mission];
                    if (!m.has_blocker || (portfolio != "all" && m.portfolio != portfolio))
                        continue;
                    n++;
                    over += f.cross == 1;
                    around += f.cross == 2;
                    never += f.cross == 0;
                    over_ok += f.cross == 1 && f.complete;
                    around_ok += f.cross == 2 && f.complete;
                    timeouts += !f.complete && !f.contact;
                    climb += f.climb;
                }
                if (n)
                    std::printf("blocker_%s,flights=%u,over=%u(complete %u),around=%u(complete %u),never_passed=%u,"
                                "timeouts=%u,mean_max_climb=%.2f\n", portfolio.c_str(), n, over, over_ok, around,
                                around_ok, never, timeouts, climb / n);
            }
            std::printf("contacts_box_missions,%llu/%llu\n", (unsigned long long)mover_contacts, (unsigned long long)movers);
            if (total) {
                const char *names[7] = {"none", "floor", "ceiling", "wall_textureless", "wall_textured", "thin", "mover"};
                std::printf("preset_surfaces");
                for (int c = 0; c < 7; c++)
                    std::printf(",%s=%.3f", names[c], double(surfaces[c]) / total);
                std::printf("\n");
            }
            {
                const char *kinds[6] = {"in_view", "side", "behind", "floor", "ceiling", "box"};
                uint32_t n[6] = {0, 0, 0, 0, 0, 0};
                double speed[6] = {0, 0, 0, 0, 0, 0};
                for (const Flight &f : flights)
                    if (f.contact && f.contact_kind >= 0)
                        n[f.contact_kind]++, speed[f.contact_kind] += f.contact_speed;
                std::printf("contact_where");
                for (int c = 0; c < 6; c++)
                    std::printf(",%s=%u@%.2f", kinds[c], n[c], n[c] ? speed[c] / n[c] : 0.0);
                std::printf("\n");
            }
            if (cap_ticks)
                std::printf("plan_ahead_mean_allowed_speed,%.2f\n", cap / cap_ticks);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
