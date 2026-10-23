// Flies one mission: startup scan, then visible waypoints issued one at a time, with full
// physics, depth rendering and per-step contact checks. Writes a JSON report and CSV traces.

#include "navigation/measured_route.hpp"
#include "navigation/near_surface_view.hpp"
#include "navigation/observed_route_navigator.hpp"
#include "navigation/raptor_model.hpp"
#include "navigation/surface_watch.hpp"
#include "navigation/visible_world.hpp"
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Visibility {
    bool frustum = false, range = false, clear = false;
    float distance = 0, ray_range = 0;
    Vec3 body;
    bool eligible() const { return frustum && range && clear; }
};
static Visibility visible(const VisibleWorld &world, const safe_nav::Pose &pose, Vec3 goal) {
    Visibility v;
    Vec3 d = goal - pose.position;
    v.distance = safe_nav::norm(d);
    v.body = safe_nav::inverse_rotate(pose, d);
    v.frustum =
        v.body.x > 0 && std::fabs(v.body.y) <= v.body.x && std::fabs(v.body.z) <= .75f * v.body.x;
    v.range = v.distance > .25f && v.distance < 11.9f;
    v.ray_range = world.ray(pose.position, safe_nav::unit(d));
    v.clear = v.ray_range + .001f >= v.distance;
    return v;
}

struct ReacquirePolicy {
    static constexpr int decision_cap = 240;
    static constexpr int leg_decision_cap = 60;
    static constexpr int candidate_count = 7;
    static constexpr int look_cap = 40;
};

struct StartupPolicy {
    static constexpr float envelope =
        safe_nav::Navigator::body_radius + safe_nav::Navigator::margin;
    static constexpr int rotation_ticks = 260;
    static constexpr int move_ticks = 80;
    static constexpr int certify_ticks = 200;
    static constexpr int gate_consecutive = 10;
    static constexpr float rotation_rate = .5f, move_speed = .3f, arrive_tolerance = .06f;
    static constexpr float move_reach = .75f, gate_speed = .15f, arrive_deceleration = .5f;
};

static std::vector<Vec3> startup_candidates() {
    std::vector<Vec3> list;
    const float step = safe_nav::Navigator::resolution;
    for (int dz = -1; dz <= 1; dz++)
        for (int iy = -2; iy <= 2; iy++)
            for (int ix = -2; ix <= 2; ix++) {
                if (ix == 0 && iy == 0 && dz == 0)
                    continue;
                if (dz != 0 && std::max(std::abs(ix), std::abs(iy)) > 1)
                    continue;
                list.push_back({ix * step, iy * step, dz * step});
            }
    std::stable_sort(list.begin(), list.end(), [](Vec3 a, Vec3 b) {
        float na = safe_nav::norm(a), nb = safe_nav::norm(b);
        if (std::fabs(na - nb) > 1e-6f)
            return na < nb;
        if (a.z != b.z)
            return a.z < b.z;
        if (a.y != b.y)
            return a.y < b.y;
        return a.x < b.x;
    });
    return list;
}

int main(int argc, char **argv) {
    try {
        std::vector<std::string> positional;
        int sweep_argument = -1, watch_argument = -1, legs_argument = -1, reacquire_argument = -1;
        for (int i = 1; i < argc; i++) {
            std::string argument = argv[i];
            if (argument == "--sweep") {
                if (i + 1 >= argc)
                    throw std::runtime_error("--sweep needs 0 or 1");
                sweep_argument = std::stoi(argv[++i]);
            } else if (argument == "--surface-watch") {
                if (i + 1 >= argc)
                    throw std::runtime_error("--surface-watch needs a mode");
                watch_argument = std::stoi(argv[++i]);
            } else if (argument == "--reacquire") {
                if (i + 1 >= argc)
                    throw std::runtime_error("--reacquire needs 0 or 1");
                reacquire_argument = std::stoi(argv[++i]);
            } else if (argument == "--standoff-legs") {
                if (i + 1 >= argc)
                    throw std::runtime_error("--standoff-legs needs 1 or 2");
                legs_argument = std::stoi(argv[++i]);
            } else if (argument == "--require-launch-contract") {

            } else if (argument == "--sensing-latency" || argument == "--depth-dropout") {
                i++;
            } else if (argument == "--launch-contract" || argument == "--launch-contract-bound") {
                i++;
            } else if (argument == "--attention-window") {
                i++;
            } else if (argument == "--attention-floor") {
                i++;
            } else if (argument == "--attention-motion") {

            } else if (argument == "--ring-extra") {
                i++;
            } else if (argument == "--ring-retreat") {

            } else if (argument == "--watch-after-handover") {

            } else if (argument == "--ray-slope") {
                i++;
            } else if (argument == "--camera-pitch") {
                i++;
            } else if (argument == "--reacquire-debug") {
                i++;
            } else if (argument.rfind("--", 0) == 0) {
                throw std::runtime_error("Unknown option: " + argument);
            } else {
                positional.push_back(argument);
            }
        }
        if (positional.size() < 6 || positional.size() > 7)
            throw std::runtime_error("benchmark TASK RAPTOR OUTPUT REFINED SWEEP MEASURED_ROUTE "
                                     "[SURFACE_WATCH] [--sweep 0|1] [--surface-watch N]");
        const std::string task_path = positional[0], weights_path = positional[1],
                          output = positional[2];
        const int refined = std::stoi(positional[3]);
        const int sweep = sweep_argument >= 0 ? sweep_argument : std::stoi(positional[4]);
        const int measured_route_enabled = std::stoi(positional[5]);
        const int watch_mode = watch_argument >= 0
                                   ? watch_argument
                                   : (positional.size() > 6 ? std::stoi(positional[6]) : 0);
        if (sweep != 0 && sweep != 1)
            throw std::runtime_error("--sweep must be 0 (executed) or 1 (synthetic sweep)");
        const bool synthetic_sweep = sweep == 1;
        const int standoff_legs = legs_argument >= 0 ? legs_argument : 2;

        const bool reacquire_enabled = reacquire_argument == 1;
        if (reacquire_argument > 1)
            throw std::runtime_error("--reacquire must be 0 or 1");
        float ray_slope = 0.75f;
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--ray-slope")
                ray_slope = std::stof(argv[i + 1]);

        float contract_translation_bound = -1.f;
        std::string contract_sha;
        bool require_contract = false;
        for (int i = 1; i < argc; i++)
            if (std::string(argv[i]) == "--require-launch-contract")
                require_contract = true;
        for (int i = 1; i + 1 < argc; i++) {
            std::string a = argv[i];
            if (a == "--launch-contract")
                contract_sha = argv[i + 1];
            else if (a == "--launch-contract-bound")
                contract_translation_bound = std::stof(argv[i + 1]);
        }

        int sensing_latency = 0;
        float depth_dropout = 0.f;
        for (int i = 1; i + 1 < argc; i++) {
            std::string a = argv[i];
            if (a == "--sensing-latency")
                sensing_latency = std::stoi(argv[i + 1]);
            else if (a == "--depth-dropout")
                depth_dropout = std::stof(argv[i + 1]);
        }
        int attention_motion = 0;
        float attention_floor = .05f;
        float attention_window = .7598f;
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--attention-window")
                attention_window = std::stof(argv[i + 1]);
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--attention-floor")
                attention_floor = std::stof(argv[i + 1]);
        for (int i = 1; i < argc; i++)
            if (std::string(argv[i]) == "--attention-motion")
                attention_motion = 1;
        int ring_retreat = 0;
        float ring_extra = .125f + .02f;
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--ring-extra")
                ring_extra = std::stof(argv[i + 1]);
        for (int i = 1; i < argc; i++)
            if (std::string(argv[i]) == "--ring-retreat")
                ring_retreat = 1;
        int watch_after_handover = 0;
        for (int i = 1; i < argc; i++)
            if (std::string(argv[i]) == "--watch-after-handover")
                watch_after_handover = 1;
        float camera_pitch_deg = 0;
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--camera-pitch")
                camera_pitch_deg = std::stof(argv[i + 1]);
        std::string reacquire_debug;
        for (int i = 1; i + 1 < argc; i++)
            if (std::string(argv[i]) == "--reacquire-debug")
                reacquire_debug = argv[i + 1];
        if (standoff_legs != 1 && standoff_legs != 2)
            throw std::runtime_error("--standoff-legs must be 1 (ablation) or 2 "
                                     "(declared executed startup)");

        std::ifstream config(task_path);
        std::string map_path;
        float scale;
        Vec3 start;
        int goals_count;
        config >> map_path >> scale >> start.x >> start.y >> start.z >> goals_count;
        if (!config || goals_count < 1 || goals_count > 32)
            throw std::runtime_error("Invalid task");
        std::vector<Vec3> goals(goals_count);
        for (auto &goal : goals)
            config >> goal.x >> goal.y >> goal.z;
        if (std::filesystem::path(map_path).is_relative())
            map_path = (std::filesystem::path(task_path).parent_path() / map_path).string();
        VisibleWorld world(map_path);
        world.scale = scale;
        int movers;
        config >> movers;
        if (movers < 0 || movers > 16)
            throw std::runtime_error("Invalid movers");
        for (int i = 0; i < movers; i++) {
            MovingBox box;
            config >> box.center.x >> box.center.y >> box.center.z >> box.half.x >> box.half.y >>
                box.half.z >> box.axis >> box.amplitude >> box.period >> box.phase;
            if (box.period <= 0 || box.axis < 0 || box.axis > 2)
                throw std::runtime_error("Invalid mover");
            world.movers.push_back(box);
        }
        if (!config)
            throw std::runtime_error("Truncated task");
        float launch_yaw = 0;
        std::string trailing;
        if (config >> trailing) {
            if (trailing != "yaw" || !(config >> launch_yaw) || !std::isfinite(launch_yaw))
                throw std::runtime_error("Invalid launch heading");
            if (config >> trailing)
                throw std::runtime_error("Unknown task trailer");
        }

        safe_nav::sensor_slope = ray_slope;
        RaptorWeights weights;
        if (!raptor_load_weights(weights_path.c_str(), &weights))
            throw std::runtime_error("Invalid RAPTOR weights");
        auto dynamics = rl_physics_crazyflie_default();
        RLPhysicsState physics{};
        physics.orientation_wxyz[0] = std::cos(launch_yaw / 2);
        physics.orientation_wxyz[3] = std::sin(launch_yaw / 2);
        physics.position[0] = start.x;
        physics.position[1] = start.y;
        physics.position[2] = start.z;
        float thrust = dynamics.mass * 9.81f / 4, c0 = dynamics.rotor_thrust_coefficients[0],
              c1 = dynamics.rotor_thrust_coefficients[1],
              c2 = dynamics.rotor_thrust_coefficients[2];
        float hover = (-c1 + std::sqrt(c1 * c1 - 4 * c2 * (c0 - thrust))) / (2 * c2);
        std::fill(physics.rpm, physics.rpm + 4, hover);
        safe_nav::RaptorControlState control;
        control.reference = start;
        control.reference_yaw = launch_yaw;
        raptor_reset(weights, control.hidden);
        safe_nav::RaptorVehicleModel actual(weights, dynamics, physics, control),
            workspace = actual;
        safe_nav::ObservedRouteNavigator navigator(true, refined != 0);
        safe_nav::SurfaceWatch surface_watch;
        const bool watch_enabled = watch_mode > 0 && watch_mode < 4,
                   watch_view_only = watch_mode == 2;
        surface_watch.require_measured_motion(watch_mode >= 2);
        safe_nav::NearSurfaceView near_surface_view;
        near_surface_view.use_range_barrier(watch_mode == 5 || watch_mode == 7, watch_mode == 7);
        near_surface_view.use_change_priority(watch_mode == 6);
        near_surface_view.use_motion_coordination(attention_motion == 1);
        near_surface_view.set_motion_floor(attention_floor);
        near_surface_view.set_forward_window(attention_window);
        navigator.use_near_surface_view(watch_mode >= 4 ? &near_surface_view : nullptr);
        safe_nav::MeasuredRoute route_workspace;
        route_workspace.use_continuous_arrival(watch_mode >= 8);
        navigator.use_joint_arrival_view(watch_mode == 9);
        navigator.reset(start);
        navigator.use_measured_route(measured_route_enabled != 0 ? &route_workspace : nullptr);
        navigator.use_surface_watch(watch_enabled ? &surface_watch : nullptr, watch_view_only);
        actual.moving_surface_guard = watch_enabled && !watch_view_only ? &surface_watch : nullptr;

        int synthetic_scan_poses = 0;
        if (synthetic_sweep) {
            std::ofstream scan(output + ".preflight.f32", std::ios::binary);
            for (int elevation = -1; elevation <= 1; elevation++)
                for (int heading = 0; heading < 8; heading++) {
                    float yaw = heading * 6.28318530718f / 8, pitch = elevation * 1.0471975512f;
                    float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch),
                          sp = std::sin(pitch);
                    safe_nav::Pose capture{
                        start, {cy * cp, -sy, cy * sp, sy * cp, cy, sy * sp, -sp, 0, cp}};
                    auto depth = world.depth(capture);
                    navigator.observe(capture, depth);
                    scan.write(reinterpret_cast<const char *>(&capture), sizeof(capture));
                    scan.write(reinterpret_cast<const char *>(depth.data()), sizeof(depth));
                    ++synthetic_scan_poses;
                }
        }
        if (watch_enabled)
            surface_watch.reset();
        near_surface_view.reset();

        if (require_contract && !synthetic_sweep && contract_sha.empty()) {
            std::ofstream report(output);
            report << "{\"outcome\":\"launch_contract_absent\",\"launch_contract_sha256\":\"\""
                   << ",\"launch_contract_bound_m\":" << contract_translation_bound
                   << ",\"require_launch_contract\":1}\n";
            std::cout << "launch_contract_absent\n";
            return 3;
        }
        const std::vector<Vec3> candidates = startup_candidates();
        bool guarded = false, collision = world.contact(start);
        int leg = 0, hold = 0, leg_ticks = 0, steps = 0, model_steps = 0, startup_ticks = 0,
            inspection_ticks = 0;
        int startup_model_steps = 0, gate_streak = 0, handover_tick = -1, startup_contacts = 0,
            startup_contact_tick = -1;
        bool watch_activated = false, contract_exceeded = false;
        std::deque<std::pair<safe_nav::Pose, std::array<float, 320>>> capture_history;
        int ring_retreat_decisions = 0;
        int reacquire_decisions = 0, leg_reacquire_decisions = 0, reacquire_events = 0,
            reacquire_candidate = -1, reacquire_looks = 0;
        int rotation_used = 0, move_used = 0, certify_used = 0, candidate_index = -1;
        float handover_distance = -1, handover_heading = 0, startup_contact_x = 0,
              startup_contact_y = 0, startup_contact_z = 0, previous_capture_time = -1;
        std::string startup_phase = synthetic_sweep ? "synthetic_sweep" : "rotate",
                    startup_reason = "none";
        bool startup_capture_pose_ok = true, startup_clock_monotone = true;
        Vec3 candidate_offset{}, move_target{};
        bool target_locked = false, arrived = false, pending_goal = false;
        float startup_peak_speed = 0;
        int stage = 1, rotation_total = 0, move_total = 0;
        Vec3 standoff_position{}, target_position{};
        float path = 0;
        std::string outcome = collision ? "initial_contact" : "startup_unavailable";

        std::ofstream trace(output + ".trace.csv"), issuance(output + ".visibility.csv"),
            handovers(output + ".handovers.csv");
        trace << std::setprecision(9)
              << "tick,time_s,guarded,leg,x,y,z,vx,vy,vz,goal_x,goal_y,goal_z,"
                 "request_x,request_y,request_z,yaw_rate,"
                 "collision,model_steps,planner_us,route_x,route_y,route_z,route_"
                 "yaw,route_expanded,route_complete,"
                 "target_x,target_y,target_z,focus_x,focus_y,focus_z,has_focus,"
                 "startup_ticks,handover_tick,handover_distance_m,handover_heading_"
                 "rad,startup_contacts,startup_phase\n";
        issuance << std::setprecision(9)
                 << "leg,time_s,camera_x,camera_y,camera_z,goal_x,goal_y,goal_z,"
                    "body_x,body_y,body_z,distance_m,ray_range_m,"
                    "frustum,range,unoccluded,eligible\n";
        handovers << std::setprecision(9) << "leg,time_s,distance_m,speed_mps\n";
        std::ofstream captures(output + ".depth-pose.bin", std::ios::binary),
            objects(output + ".objects.csv");
        std::ofstream delivered_log(output + ".delivered.csv");
        delivered_log << std::setprecision(9)
                      << "tick,capture_x,capture_y,capture_z,delivered_x,delivered_y,"
                         "delivered_z,delivered,declared_latency_ticks,dropped_rays\n";
        std::ofstream attention(output + ".attention.csv");
        attention << std::setprecision(9)
                  << "time_s,active,focus_x,focus_y,focus_z,range_m,retreating,"
                     "target_x,target_y,target_z,target_supported,"
                     "verified,selected_x,selected_y,selected_z,selected_yaw\n";
        std::ofstream decisions(output + ".motion.csv"), applied_commands(output + ".commands.csv");
        decisions << std::setprecision(9)
                  << "tick,time_s,event,trial,requested_x,requested_y,requested_z,requested_yaw,"
                     "verified,selected_x,selected_y,selected_z,selected_yaw\n";
        applied_commands << std::setprecision(9)
                         << "physics_step,time_s,tick,substep,guarded,vx,vy,vz,yaw_rate,motor0,"
                            "motor1,motor2,motor3\n";
        objects << std::setprecision(9)
                << "time_s,object,camera_body_x,camera_body_y,camera_body_z,body_"
                   "surface_distance_m,raw_hit_rays\n";
        std::ofstream startup(output + ".startup.csv");
        startup << std::setprecision(9)
                << "tick,time_s,measured_sim_time_s,phase,x,y,z,yaw,speed,distance_"
                   "from_launch,body_free,body_unknown,"
                   "body_occupied,body_outside,body_known_free,handover_verified,"
                   "handover_status,handover_model_steps,"
                   "gate_streak,phase_ticks,candidate_index,candidate_dx,candidate_"
                   "dy,candidate_dz,candidate_known_free,"
                   "path_measured_clear,command_x,command_y,command_z,command_yaw_"
                   "rate,contact,capture_pose_is_measured,"
                   "clock_advanced_between_captures,handover_has_focus,handover_"
                   "focus_x,handover_focus_y,handover_focus_z,reference_x,"
                   "reference_y,reference_z,reference_gap_m\n";
        std::ofstream changes(output + ".change-truth.csv");
        changes << "time_s,changed_returns,moving_returns,static_returns\n";

        auto yaw_of = [](const safe_nav::Pose &pose) {
            return std::atan2(pose.rotation[3], pose.rotation[0]);
        };
        auto issue = [&]() {
            auto pose = actual.pose();
            auto view = visible(world, pose, goals[leg]);
            issuance << leg << ',' << steps * .01 << ',' << pose.position.x << ','
                     << pose.position.y << ',' << pose.position.z << ',' << goals[leg].x << ','
                     << goals[leg].y << ',' << goals[leg].z << ',' << view.body.x << ','
                     << view.body.y << ',' << view.body.z << ',' << view.distance << ','
                     << view.ray_range << ',' << view.frustum << ',' << view.range << ','
                     << view.clear << ',' << view.eligible() << '\n';
            issuance.flush();
            return view.eligible();
        };
        if (!collision && !issue())
            outcome = "ineligible_initial_goal";

        auto began = std::chrono::steady_clock::now();
        double planning_total = 0, planning_max = 0;
        int plans = 0;
        bool stale_snapshot_saved = false;

        for (int tick = 0; tick < 6400 && !collision && outcome != "ineligible_initial_goal";
             tick++) {
            auto pose = actual.pose();

            safe_nav::Pose camera_pose = pose;
            if (camera_pitch_deg != 0) {
                const float pitch = camera_pitch_deg * 3.14159265358979f / 180.f;
                const float cp = std::cos(pitch), sp = std::sin(pitch);

                std::array<float, 9> mount{cp, 0, -sp, 0, 1, 0, sp, 0, cp};
                std::array<float, 9> composed{};
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        composed[r * 3 + c] = pose.rotation[r * 3 + 0] * mount[0 * 3 + c] +
                                              pose.rotation[r * 3 + 1] * mount[1 * 3 + c] +
                                              pose.rotation[r * 3 + 2] * mount[2 * 3 + c];
                camera_pose.rotation = composed;
            }
            auto depth_frame = world.depth(camera_pose);

            safe_nav::Pose capture_pose = pose;
            safe_nav::Pose plant_pose_after_render = actual.pose();
            bool capture_pose_is_measured =
                std::memcmp(&capture_pose, &plant_pose_after_render, sizeof(safe_nav::Pose)) == 0;
            float measured_sim_time = steps * .01f;
            world.time = measured_sim_time;
            bool clock_advanced = measured_sim_time > previous_capture_time;
            if (!capture_pose_is_measured)
                startup_capture_pose_ok = false;
            if (previous_capture_time >= 0 && !clock_advanced)
                startup_clock_monotone = false;
            previous_capture_time = measured_sim_time;

            std::array<float, 320> observed_depth = depth_frame;
            safe_nav::Pose observed_pose = camera_pose;
            int dropped_rays = 0;
            if (depth_dropout > 0.f) {
                const int period = std::max(2, int(std::round(1.f / depth_dropout)));
                for (int ray = 0; ray < 320; ray++)
                    if ((ray + tick) % period == 0) {

                        observed_depth[ray] = std::numeric_limits<float>::quiet_NaN();
                        ++dropped_rays;
                    }
            }
            bool delivered = true;
            if (sensing_latency > 0) {
                capture_history.push_back({camera_pose, observed_depth});
                if (int(capture_history.size()) > sensing_latency) {
                    const auto &delayed = capture_history.front();
                    observed_pose = delayed.first;
                    observed_depth = delayed.second;
                    capture_history.pop_front();
                } else {

                    delivered = false;
                }
            }
            if (delivered)
                navigator.observe(observed_pose, observed_depth);
            if (delivered_log.good()) {
                delivered_log << tick << ',' << camera_pose.position.x << ','
                              << camera_pose.position.y << ',' << camera_pose.position.z << ','
                              << observed_pose.position.x << ',' << observed_pose.position.y << ','
                              << observed_pose.position.z << ',' << delivered << ','
                              << sensing_latency << ',' << dropped_rays << '\n';
            }
            safe_nav::Command command;
            double elapsed = 0;
            Vec3 stale_focus;
            if (watch_enabled && !stale_snapshot_saved &&
                surface_watch.stale_focus(pose.position, stale_focus)) {
                std::ofstream state(output + ".first-stale.bin", std::ios::binary);
                const auto &current_map = navigator.map();
                state.write(reinterpret_cast<const char *>(&current_map), sizeof(current_map));
                state.write(reinterpret_cast<const char *>(&actual.physics),
                            sizeof(actual.physics));
                state.write(reinterpret_cast<const char *>(&actual.control),
                            sizeof(actual.control));
                std::ofstream tracked(output + ".first-stale-watch.bin", std::ios::binary);
                tracked.write(reinterpret_cast<const char *>(&surface_watch),
                              sizeof(surface_watch));
                stale_snapshot_saved = true;
            }
            float capture_time = steps * .01f;
            captures.write(reinterpret_cast<const char *>(&capture_time), sizeof(capture_time));
            captures.write(reinterpret_cast<const char *>(&pose), sizeof(pose));
            captures.write(reinterpret_cast<const char *>(depth_frame.data()), sizeof(depth_frame));

            safe_nav::BackupDecision ready;
            safe_nav::BodyCoverage coverage;
            if (!guarded) {
                coverage = safe_nav::body_coverage(navigator.map(), pose.position,
                                                   StartupPolicy::envelope);
                ready = navigator.check_handover(actual, workspace);
                model_steps += ready.model_steps;
                startup_model_steps += ready.model_steps;

                const bool ready_now = synthetic_sweep ? ready.verified
                                                       : ready.verified &&
                                                             safe_nav::norm(actual.velocity()) <=
                                                                 StartupPolicy::gate_speed &&
                                                             !collision;
                gate_streak = ready_now ? gate_streak + 1 : 0;
                if (gate_streak >= (synthetic_sweep ? 1 : StartupPolicy::gate_consecutive)) {
                    guarded = true;
                    handover_tick = tick;
                    handover_distance = safe_nav::norm(pose.position - start);
                    handover_heading = yaw_of(pose);
                    startup_phase = "handed_over";
                    if (watch_after_handover) {

                        surface_watch.reset();
                        navigator.use_surface_watch(&surface_watch, false);
                        actual.moving_surface_guard = &surface_watch;
                        watch_activated = true;
                    }
                    if (synthetic_sweep)
                        startup_reason = "synthetic_sweep_verified";
                    else if (target_locked)
                        startup_reason = "verified_at_certificate_target";
                    else
                        startup_reason = "verified_at_launch_pose";
                    outcome = "waypoint_timeout";
                }
            }
            if (!guarded) {
                if (synthetic_sweep) {

                    if (!ready.verified) {
                        outcome = "preflight_not_ready";
                        startup_reason = "synthetic_sweep_not_verified";
                        break;
                    }
                } else {

                    if (arrived && stage == 1 && standoff_legs >= 2) {
                        stage = 2;
                        arrived = false;
                        target_locked = false;
                        rotation_used = 0;
                        move_used = 0;
                        standoff_position = pose.position;
                    }
                    const Vec3 gap = actual.reference_position() - pose.position;
                    if (!target_locked) {
                        for (size_t i = 0; i < candidates.size(); i++) {
                            if (safe_nav::norm(candidates[i]) > StartupPolicy::move_reach)
                                continue;
                            Vec3 target = pose.position + candidates[i];
                            Vec3 reference = target + gap;
                            if (!safe_nav::body_coverage(navigator.map(), target,
                                                         StartupPolicy::envelope)
                                     .known_free())
                                continue;
                            if (stage == 2) {
                                if (!safe_nav::body_coverage(navigator.map(), reference,
                                                             StartupPolicy::envelope)
                                         .known_free())
                                    continue;
                                if (!safe_nav::swept_body_known_free(navigator.map(), target,
                                                                     reference,
                                                                     StartupPolicy::envelope))
                                    continue;
                            }
                            bool clear = true;
                            for (int sample = 1; sample <= 8 && clear; sample++)
                                if (!navigator.map().measured_clear(
                                        pose.position + candidates[i] * (sample / 8.0f)))
                                    clear = false;
                            if (!clear)
                                continue;
                            candidate_index = int(i);
                            candidate_offset = candidates[i];
                            move_target = target;
                            target_locked = true;
                            break;
                        }
                        if (rotation_used > 0 && rotation_used >= StartupPolicy::rotation_ticks) {
                            outcome = "startup_unavailable";
                            startup_reason = stage == 1 ? "no_observed_envelope_standoff"
                                                        : "no_observed_certificate_target";
                            break;
                        }
                    }
                    if (target_locked) {
                        Vec3 delta = move_target - pose.position;
                        float length = safe_nav::norm(delta);
                        if (length <= StartupPolicy::arrive_tolerance) {
                            arrived = true;
                            if (stage == 1)
                                standoff_position = pose.position;
                            else
                                target_position = pose.position;
                        }
                        if (arrived) {

                            startup_phase = "certify";
                            ++certify_used;
                            command = {};
                            if (certify_used > StartupPolicy::certify_ticks) {
                                outcome = "startup_unavailable";
                                startup_reason = "handover_not_verified_at_certificate_target";
                                break;
                            }
                        } else {
                            startup_phase = stage == 1 ? "acquire_move_1" : "acquire_move_2";
                            ++move_used;
                            ++move_total;

                            float speed = std::min(
                                StartupPolicy::move_speed,
                                std::sqrt(2 * StartupPolicy::arrive_deceleration *
                                          std::max(length - StartupPolicy::arrive_tolerance, 0.f)));
                            command.world_velocity = safe_nav::unit(delta) * speed;
                            command.yaw_rate = std::clamp(
                                2 * safe_nav::wrap(std::atan2(delta.y, delta.x) - yaw_of(pose)),
                                -.5f, .5f);
                            if (move_used > StartupPolicy::move_ticks) {
                                outcome = "startup_unavailable";
                                startup_reason = stage == 1 ? "standoff_move_did_not_arrive"
                                                            : "target_move_did_not_arrive";
                                break;
                            }
                        }
                    } else {
                        startup_phase = stage == 1 ? "rotate_launch" : "rotate_standoff";

                        command.yaw_rate = StartupPolicy::rotation_rate;
                        ++rotation_used;
                        ++rotation_total;
                    }
                }
            }
            if (!guarded && startup_phase != "handed_over") {
                if (!synthetic_sweep && contract_translation_bound >= 0.f) {

                    const float travelled = safe_nav::norm(actual.pose().position - start);
                    if (travelled > contract_translation_bound + 1e-4f) {

                        outcome = "launch_contract_exceeded";
                        startup_reason = "unprotected_translation_beyond_declared_bound";
                        contract_exceeded = true;
                        break;
                    }
                }
                startup_peak_speed =
                    std::max(startup_peak_speed, safe_nav::norm(actual.velocity()));
                bool candidate_known_free =
                    target_locked &&
                    safe_nav::body_coverage(navigator.map(), move_target, StartupPolicy::envelope)
                        .known_free();
                bool path_clear = true;
                for (int sample = 1; sample <= 8 && target_locked && path_clear; sample++)
                    if (!navigator.map().measured_clear(
                            pose.position + (move_target - pose.position) * (sample / 8.0f)))
                        path_clear = false;
                startup << tick << ',' << measured_sim_time << ',' << world.time << ','
                        << startup_phase << ',' << pose.position.x << ',' << pose.position.y << ','
                        << pose.position.z << ',' << yaw_of(pose) << ','
                        << safe_nav::norm(actual.velocity()) << ','
                        << safe_nav::norm(pose.position - start) << ',' << coverage.free_cells
                        << ',' << coverage.unknown_cells << ',' << coverage.occupied_cells << ','
                        << coverage.outside_cells << ',' << coverage.known_free() << ','
                        << ready.verified << ',' << int(ready.status) << ',' << ready.model_steps
                        << ',' << gate_streak << ','
                        << (startup_phase == "rotate"         ? rotation_used
                            : startup_phase == "acquire_move" ? move_used
                                                              : certify_used)
                        << ',' << candidate_index << ',' << candidate_offset.x << ','
                        << candidate_offset.y << ',' << candidate_offset.z << ','
                        << candidate_known_free << ',' << path_clear << ','
                        << command.world_velocity.x << ',' << command.world_velocity.y << ','
                        << command.world_velocity.z << ',' << command.yaw_rate << ',' << collision
                        << ',' << capture_pose_is_measured << ',' << clock_advanced << ','
                        << ready.has_inspection_focus << ',' << ready.inspection_focus.x << ','
                        << ready.inspection_focus.y << ',' << ready.inspection_focus.z << ','
                        << actual.reference_position().x << ',' << actual.reference_position().y
                        << ',' << actual.reference_position().z << ','
                        << safe_nav::norm(actual.reference_position() - pose.position) << '\n';
                startup.flush();
                ++startup_ticks;
            }

            if (guarded) {
                if (leg_ticks >= 2000) {
                    outcome = "waypoint_timeout";
                    break;
                }
                if (model_steps + navigator.next_plan_model_step_limit() > 1800000) {
                    outcome = "model_budget_stop";
                    break;
                }
                auto before = std::chrono::steady_clock::now();
                safe_nav::BackupDecision choice;
                if (pending_goal) {
                    auto view = visible(world, actual.pose(), goals[leg]);
                    if (view.eligible()) {
                        if (!issue())
                            throw std::runtime_error("Visibility changed without physics");
                        pending_goal = false;
                        inspection_ticks = 0;
                    }
                    if (pending_goal) {
                        if (++inspection_ticks > 400) {
                            outcome = "view_acquisition_timeout";
                            break;
                        }
                        safe_nav::Command look;
                        Vec3 d = goals[leg] - actual.pose().position;
                        float yaw = yaw_of(actual.pose());
                        look.yaw_rate =
                            std::clamp(2 * safe_nav::wrap(std::atan2(d.y, d.x) - yaw), -.5f, .5f);
                        choice = navigator.inspect(actual, look, workspace);
                    }
                }
                if (!pending_goal)
                    choice = navigator.plan(actual, goals[leg], workspace);
                elapsed = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - before)
                              .count();
                planning_total += elapsed;
                planning_max = std::max(planning_max, elapsed);
                ++plans;
                model_steps += choice.model_steps;
                auto focus = navigator.attention_focus(), target = navigator.attention_target();
                attention << capture_time << ',' << navigator.attention_active() << ',' << focus.x
                          << ',' << focus.y << ',' << focus.z << ','
                          << safe_nav::norm(focus - pose.position) << ','
                          << navigator.attention_retreating() << ',' << target.x << ',' << target.y
                          << ',' << target.z << ','
                          << safe_nav::body_coverage(navigator.map(), target, .26f).known_free()
                          << ',' << choice.verified << ',' << choice.command.world_velocity.x << ','
                          << choice.command.world_velocity.y << ','
                          << choice.command.world_velocity.z << ',' << choice.command.yaw_rate
                          << '\n';
                if (!choice.verified) {
                    outcome = "requires_supervisor";
                    break;
                }
                command = choice.command;
                auto record_decision = [&](const char *event, int trial,
                                           safe_nav::Command requested,
                                           const safe_nav::BackupDecision &selected) {
                    decisions << tick << ',' << capture_time << ',' << event << ',' << trial << ','
                              << requested.world_velocity.x << ',' << requested.world_velocity.y
                              << ',' << requested.world_velocity.z << ',' << requested.yaw_rate
                              << ',' << selected.verified << ','
                              << selected.command.world_velocity.x << ','
                              << selected.command.world_velocity.y << ','
                              << selected.command.world_velocity.z << ','
                              << selected.command.yaw_rate << '\n';
                };
                record_decision("primary", -1, choice.command, choice);
                if (ring_retreat) {

                    auto ring =
                        safe_nav::body_coverage(navigator.map(), pose.position, .26f + ring_extra);
                    if (ring.unknown_cells > 0 &&
                        ring_retreat_decisions < ReacquirePolicy::decision_cap) {
                        Vec3 measured = actual.velocity();
                        Vec3 horizontal{measured.x, measured.y, 0};
                        Vec3 goal_delta = goals[leg] - pose.position;
                        Vec3 heading = safe_nav::norm(horizontal) > .05f
                                           ? safe_nav::unit(horizontal)
                                           : safe_nav::unit(goal_delta);
                        safe_nav::Command retreats[2];
                        retreats[0].world_velocity = heading * -.20f;
                        retreats[1].world_velocity = heading * -.40f;
                        bool retreated = false;
                        for (auto &retreat : retreats) {
                            retreat.yaw_rate = std::clamp(
                                2 * safe_nav::wrap(std::atan2(goal_delta.y, goal_delta.x) -
                                                   yaw_of(pose)),
                                -.5f, .5f);
                            auto attempt = navigator.try_aligned_motion(actual, retreat, workspace);
                            model_steps += attempt.model_steps;
                            record_decision("retreat_trial", int(&retreat - retreats), retreat,
                                            attempt);
                            if (!attempt.verified)
                                continue;
                            command = attempt.command;
                            retreated = true;
                            ++ring_retreat_decisions;
                            break;
                        }
                        if (retreated && ring_retreat_decisions == 1)
                            ++reacquire_events;
                    }
                }
                if (reacquire_enabled) {

                    auto certified_hold_at = [&](const safe_nav::RaptorVehicleModel &model) {
                        Vec3 d = goals[leg] - model.pose().position;
                        safe_nav::Command look;
                        look.yaw_rate = std::clamp(
                            2 * safe_nav::wrap(std::atan2(d.y, d.x) - yaw_of(model.pose())), -.5f,
                            .5f);
                        safe_nav::BackupDecision held = navigator.inspect(model, look, workspace);
                        model_steps += held.model_steps;
                        return held.verified &&
                               safe_nav::body_coverage(navigator.map(), model.pose().position, .26f)
                                   .known_free();
                    };
                    auto certified_hold_at_focus = [&](const safe_nav::RaptorVehicleModel &model) {
                        Vec3 d = goals[leg] - model.pose().position;
                        safe_nav::Command look;
                        look.yaw_rate = std::clamp(
                            2 * safe_nav::wrap(std::atan2(d.y, d.x) - yaw_of(model.pose())), -.5f,
                            .5f);
                        safe_nav::BackupDecision held = navigator.inspect(model, look, workspace);
                        model_steps += held.model_steps;
                        return held.has_inspection_focus ? held.inspection_focus
                                                         : model.pose().position;
                    };
                    auto successor_is_stoppable = [&](safe_nav::Command candidate) {
                        safe_nav::RaptorVehicleModel trial = actual;
                        for (int k = 0; k < 5; k++)
                            if (!trial.substep(candidate))
                                return false;
                        return certified_hold_at(trial);
                    };
                    if (!successor_is_stoppable(command)) {

                        bool observed = false;
                        Vec3 goal_delta_look = goals[leg] - pose.position;
                        Vec3 focus_hint = certified_hold_at_focus(actual);
                        const Vec3 aim = safe_nav::norm(focus_hint - pose.position) > .05f
                                             ? focus_hint - pose.position
                                             : goal_delta_look;
                        if (reacquire_looks < ReacquirePolicy::look_cap) {
                            safe_nav::Command look;
                            look.yaw_rate = std::clamp(
                                2 * safe_nav::wrap(std::atan2(aim.y, aim.x) - yaw_of(pose)), -.5f,
                                .5f);
                            safe_nav::BackupDecision looked =
                                navigator.inspect(actual, look, workspace);
                            model_steps += looked.model_steps;
                            if (looked.verified) {
                                command = looked.command;
                                ++reacquire_looks;
                                ++reacquire_decisions;
                                observed = true;
                            }
                        }
                        if (!observed) {

                            Vec3 measured = actual.velocity();
                            Vec3 horizontal{measured.x, measured.y, 0};
                            Vec3 goal_delta = goals[leg] - pose.position;
                            Vec3 heading = safe_nav::norm(horizontal) > .05f
                                               ? safe_nav::unit(horizontal)
                                               : safe_nav::unit(goal_delta);
                            Vec3 right{heading.y, -heading.x, 0};
                            safe_nav::Command escapes[6];
                            escapes[0].world_velocity = Vec3{};
                            escapes[1].world_velocity = heading * -.20f;
                            escapes[2].world_velocity = right * .25f;
                            escapes[3].world_velocity = right * -.25f;
                            escapes[4].world_velocity = Vec3{0, 0, .18f};
                            escapes[5].world_velocity = Vec3{0, 0, -.18f};
                            bool escaped = false;
                            for (auto &escape : escapes) {
                                escape.yaw_rate = std::clamp(
                                    2 * safe_nav::wrap(std::atan2(goal_delta.y, goal_delta.x) -
                                                       yaw_of(pose)),
                                    -.5f, .5f);
                                auto attempt =
                                    navigator.plan_proposal(actual, goals[leg], escape, workspace);
                                model_steps += attempt.model_steps;
                                if (!attempt.verified)
                                    continue;
                                safe_nav::Vec3 a = safe_nav::unit(attempt.command.world_velocity),
                                               b = safe_nav::unit(escape.world_velocity);
                                const bool hold_escape =
                                    safe_nav::norm(escape.world_velocity) <= .01f;
                                const float alignment = a.x * b.x + a.y * b.y + a.z * b.z;
                                const bool follows =
                                    hold_escape
                                        ? safe_nav::norm(attempt.command.world_velocity) <= .05f
                                        : alignment > .5f;
                                if (!follows)
                                    continue;
                                if (!successor_is_stoppable(attempt.command))
                                    continue;
                                command = attempt.command;
                                escaped = true;
                                reacquire_candidate = int(&escape - escapes);
                                break;
                            }
                            if (!escaped) {

                                if (++reacquire_decisions > ReacquirePolicy::decision_cap) {
                                    outcome = "requires_supervisor";
                                    break;
                                }
                                command = safe_nav::Command{};
                                command.yaw_rate = std::clamp(
                                    2 * safe_nav::wrap(std::atan2(goal_delta.y, goal_delta.x) -
                                                       yaw_of(pose)),
                                    -.5f, .5f);
                            } else {
                                ++reacquire_decisions;
                                ++leg_reacquire_decisions;
                                if (leg_reacquire_decisions == 1)
                                    ++reacquire_events;
                            }
                        }
                    }
                }
            }
            for (int k = 0; k < 5; k++) {
                if (guarded && !navigator.command_for_next_tick(command, actual.velocity())) {
                    outcome = "execution_fault";
                    break;
                }
                auto before = actual.pose().position;
                if (!actual.substep(command)) {
                    outcome = "execution_fault";
                    break;
                }
                applied_commands << steps << ',' << (steps + 1) * .01f << ',' << tick << ',' << k
                                 << ',' << guarded << ',' << command.world_velocity.x << ','
                                 << command.world_velocity.y << ',' << command.world_velocity.z
                                 << ',' << command.yaw_rate;
                for (float motor : actual.control.previous_motor)
                    applied_commands << ',' << motor;
                applied_commands << '\n';
                ++steps;
                world.time = steps * .01f;
                path += safe_nav::norm(actual.pose().position - before);
                if (guarded)
                    navigator.advance_executed_tick();
                if (world.contact(actual.pose().position)) {
                    collision = true;
                    outcome = guarded ? "guarded_contact" : "startup_contact";
                    if (!guarded) {
                        ++startup_contacts;
                        startup_contact_tick = tick;
                        startup_contact_x = actual.pose().position.x;
                        startup_contact_y = actual.pose().position.y;
                        startup_contact_z = actual.pose().position.z;
                        startup_reason = "startup_contact_in_phase_" + startup_phase;
                    }
                    break;
                }
            }
            auto p = actual.pose().position, v = actual.velocity();
            auto g = goals[leg];
            trace << tick << ',' << steps * .01 << ',' << guarded << ',' << leg << ',' << p.x << ','
                  << p.y << ',' << p.z << ',' << v.x << ',' << v.y << ',' << v.z << ',' << g.x
                  << ',' << g.y << ',' << g.z << ',' << command.world_velocity.x << ','
                  << command.world_velocity.y << ',' << command.world_velocity.z << ','
                  << command.yaw_rate << ',' << collision << ',' << model_steps << ',' << elapsed
                  << ',' << navigator.diagnostic_route.world_velocity.x << ','
                  << navigator.diagnostic_route.world_velocity.y << ','
                  << navigator.diagnostic_route.world_velocity.z << ','
                  << navigator.diagnostic_route.yaw_rate << ','
                  << route_workspace.diagnostic_expanded << ','
                  << route_workspace.diagnostic_complete << ','
                  << route_workspace.diagnostic_target().x << ','
                  << route_workspace.diagnostic_target().y << ','
                  << route_workspace.diagnostic_target().z << ','
                  << navigator.diagnostic_backup.inspection_focus.x << ','
                  << navigator.diagnostic_backup.inspection_focus.y << ','
                  << navigator.diagnostic_backup.inspection_focus.z << ','
                  << navigator.diagnostic_backup.has_inspection_focus << ',' << startup_ticks << ','
                  << handover_tick << ',' << handover_distance << ',' << handover_heading << ','
                  << startup_contacts << ',' << startup_phase << '\n';
            if (collision || outcome == "execution_fault")
                break;

            if (handover_tick == tick) {

                bool candidate_known_free =
                    target_locked &&
                    safe_nav::body_coverage(navigator.map(), move_target, StartupPolicy::envelope)
                        .known_free();
                bool path_clear = true;
                for (int sample = 1; sample <= 8 && target_locked && path_clear; sample++)
                    if (!navigator.map().measured_clear(
                            pose.position + (move_target - pose.position) * (sample / 8.0f)))
                        path_clear = false;
                startup << tick << ',' << steps * .01f << ',' << world.time << ',' << startup_phase
                        << ',' << p.x << ',' << p.y << ',' << p.z << ',' << yaw_of(actual.pose())
                        << ',' << safe_nav::norm(v) << ',' << safe_nav::norm(p - start) << ','
                        << coverage.free_cells << ',' << coverage.unknown_cells << ','
                        << coverage.occupied_cells << ',' << coverage.outside_cells << ','
                        << coverage.known_free() << ',' << ready.verified << ','
                        << int(ready.status) << ',' << ready.model_steps << ',' << gate_streak
                        << ',' << 0 << ',' << candidate_index << ',' << candidate_offset.x << ','
                        << candidate_offset.y << ',' << candidate_offset.z << ','
                        << candidate_known_free << ',' << path_clear << ','
                        << command.world_velocity.x << ',' << command.world_velocity.y << ','
                        << command.world_velocity.z << ',' << command.yaw_rate << ',' << collision
                        << ',' << true << ',' << clock_advanced << ',' << ready.has_inspection_focus
                        << ',' << ready.inspection_focus.x << ',' << ready.inspection_focus.y << ','
                        << ready.inspection_focus.z << ',' << actual.reference_position().x << ','
                        << actual.reference_position().y << ',' << actual.reference_position().z
                        << ',' << safe_nav::norm(actual.reference_position() - pose.position)
                        << '\n';
                startup.flush();
            }
            if (guarded && !pending_goal) {
                ++leg_ticks;
                hold = safe_nav::norm(p - g) < .25f && safe_nav::norm(v) < .2f ? hold + 1 : 0;
                if (hold >= 6) {
                    handovers << leg << ',' << steps * .01 << ',' << safe_nav::norm(p - g) << ','
                              << safe_nav::norm(v) << '\n';
                    ++leg;
                    hold = 0;
                    leg_ticks = 0;
                    leg_reacquire_decisions = 0;
                    navigator.next_goal();
                    if (leg == goals_count) {
                        outcome = "success";
                        break;
                    }
                    pending_goal = true;
                }
            }
        }

        {
            const safe_nav::Navigator &map = navigator.map();
            std::ofstream window(output + ".map-window.txt");
            window << std::setprecision(9)
                   << "cell_state_mask_hex around measured pose and around reference\n";
            for (int which = 0; which < 2; which++) {
                Vec3 around = which == 0 ? actual.pose().position : actual.reference_position();
                window << "around=" << (which == 0 ? "measured" : "reference") << ' ' << around.x
                       << ',' << around.y << ',' << around.z << '\n';
                Vec3 first = map.map_cell_center(0) - Vec3{.125f, .125f, .125f};
                int base[3] = {int(std::round(first.x / safe_nav::Navigator::resolution)),
                               int(std::round(first.y / safe_nav::Navigator::resolution)),
                               int(std::round(first.z / safe_nav::Navigator::resolution))};
                int centre[3] = {
                    int(std::floor(around.x / safe_nav::Navigator::resolution)) - base[0],
                    int(std::floor(around.y / safe_nav::Navigator::resolution)) - base[1],
                    int(std::floor(around.z / safe_nav::Navigator::resolution)) - base[2]};
                for (int dz = -2; dz <= 2; dz++) {
                    for (int dy = -2; dy <= 2; dy++) {
                        window << "z" << dz << " y" << dy << " :";
                        for (int dx = -2; dx <= 2; dx++) {
                            int x = centre[0] + dx, y = centre[1] + dy, z = centre[2] + dz;
                            if (x < 0 || x >= safe_nav::Navigator::nx || y < 0 ||
                                y >= safe_nav::Navigator::ny || z < 0 ||
                                z >= safe_nav::Navigator::nz) {
                                window << " out";
                                continue;
                            }
                            int index =
                                (z * safe_nav::Navigator::ny + y) * safe_nav::Navigator::nx + x;
                            char buffer[24];
                            std::snprintf(buffer, sizeof(buffer), " %d/%02x",
                                          map.map_cell_state(index),
                                          int(map.map_cell_free_mask(index)));
                            window << buffer;
                        }
                        window << '\n';
                    }
                }
            }
        }
        std::ofstream snapshot(output + ".state.bin", std::ios::binary);
        std::ofstream watch_state(output + ".watch-state.bin", std::ios::binary);
        watch_state.write(reinterpret_cast<const char *>(&surface_watch), sizeof(surface_watch));
        auto captured_map = navigator.map();
        snapshot.write(reinterpret_cast<const char *>(&captured_map), sizeof(captured_map));
        snapshot.write(reinterpret_cast<const char *>(&actual.physics), sizeof(actual.physics));
        snapshot.write(reinterpret_cast<const char *>(&actual.control), sizeof(actual.control));
        double wall =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        std::ofstream report(output);
        report << std::setprecision(12) << "{\"outcome\":\"" << outcome
               << "\",\"goals_completed\":" << leg << ",\"goals_requested\":" << goals_count
               << ",\"contact\":" << collision << ",\"guarded\":" << guarded
               << ",\"startup_s\":" << startup_ticks * .05 << ",\"time_s\":" << steps * .01
               << ",\"path_m\":" << path << ",\"model_steps\":" << model_steps
               << ",\"startup_model_steps\":" << startup_model_steps
               << ",\"physics_steps\":" << steps << ",\"wall_s\":" << wall
               << ",\"planner_mean_us\":" << planning_total / std::max(1, plans)
               << ",\"planner_max_us\":" << planning_max << ",\"navigator_bytes\":"
               << (sizeof(navigator) + (measured_route_enabled != 0 ? sizeof(route_workspace) : 0))
               << ",\"surface_changed_frames\":" << surface_watch.changed_frames
               << ",\"surface_view_overrides\":" << surface_watch.view_overrides
               << ",\"startup_protected\":false,\"operator_preflight_sweep\":"
               << (synthetic_sweep ? 1 : 0) << ",\"startup_mode\":\""
               << (synthetic_sweep ? "synthetic_sweep" : "executed") << "\""
               << ",\"synthetic_scan_poses\":" << synthetic_scan_poses
               << ",\"startup_ticks\":" << startup_ticks
               << ",\"startup_rotation_ticks\":" << rotation_used
               << ",\"startup_move_ticks\":" << move_used
               << ",\"startup_certify_ticks\":" << certify_used
               << ",\"startup_gate_consecutive\":" << StartupPolicy::gate_consecutive
               << ",\"startup_gate_speed\":" << StartupPolicy::gate_speed
               << ",\"handover_tick\":" << handover_tick
               << ",\"handover_distance_m\":" << handover_distance
               << ",\"handover_heading_rad\":" << handover_heading
               << ",\"startup_contacts\":" << startup_contacts
               << ",\"startup_contact_tick\":" << startup_contact_tick
               << ",\"startup_contact_position\":[" << startup_contact_x << ',' << startup_contact_y
               << ',' << startup_contact_z << ']' << ",\"startup_phase\":\"" << startup_phase
               << "\",\"startup_reason\":\"" << startup_reason << "\""
               << ",\"startup_candidate_index\":" << candidate_index
               << ",\"startup_candidate_offset\":[" << candidate_offset.x << ','
               << candidate_offset.y << ',' << candidate_offset.z << ']'
               << ",\"startup_capture_pose_is_measured\":"
               << (startup_capture_pose_ok ? "true" : "false")
               << ",\"startup_clock_monotone\":" << (startup_clock_monotone ? "true" : "false")
               << ",\"startup_peak_measured_speed_mps\":" << startup_peak_speed
               << ",\"startup_stage\":" << stage << ",\"startup_standoff_legs\":" << standoff_legs
               << ",\"startup_rotation_total\":" << rotation_total
               << ",\"startup_move_total\":" << move_total << ",\"startup_standoff_position\":["
               << standoff_position.x << ',' << standoff_position.y << ',' << standoff_position.z
               << ']' << ",\"startup_target_position\":[" << target_position.x << ','
               << target_position.y << ',' << target_position.z << ']'
               << ",\"startup_arrived\":" << (arrived ? "true" : "false")
               << ",\"startup_arrival_position\":[" << (target_locked ? move_target.x : start.x)
               << ',' << (target_locked ? move_target.y : start.y) << ','
               << (target_locked ? move_target.z : start.z) << ']' << ",\"startup_state_at_end\":["
               << actual.pose().position.x << ',' << actual.pose().position.y << ','
               << actual.pose().position.z << ']' << ",\"startup_reference_at_end\":["
               << actual.reference_position().x << ',' << actual.reference_position().y << ','
               << actual.reference_position().z << ']' << ",\"startup_setpoint_gap_at_end_m\":"
               << safe_nav::norm(actual.reference_position() - actual.pose().position)
               << ",\"launch_yaw\":" << launch_yaw
               << ",\"reacquire_enabled\":" << (reacquire_enabled ? 1 : 0)
               << ",\"reacquire_decisions\":" << reacquire_decisions
               << ",\"reacquire_events\":" << reacquire_events
               << ",\"reacquire_candidate\":" << reacquire_candidate
               << ",\"reacquire_looks\":" << reacquire_looks
               << ",\"watch_after_handover\":" << (watch_activated ? 1 : 0)
               << ",\"ring_retreat_decisions\":" << ring_retreat_decisions
               << ",\"launch_contract_sha256\":\"" << contract_sha << "\""
               << ",\"launch_contract_bound_m\":" << contract_translation_bound
               << ",\"launch_contract_exceeded\":" << (contract_exceeded ? 1 : 0)
               << ",\"require_launch_contract\":" << (require_contract ? 1 : 0)
               << ",\"sensing_latency\":" << sensing_latency
               << ",\"depth_dropout\":" << depth_dropout
               << ",\"attention_motion\":" << attention_motion
               << ",\"attention_window\":" << attention_window
               << ",\"camera_pitch_deg\":" << camera_pitch_deg << ",\"ray_slope\":" << ray_slope
               << "}\n";
        std::cout << outcome << " goals=" << leg << " contact=" << collision
                  << " startup_ticks=" << startup_ticks << " handover_tick=" << handover_tick
                  << " startup_contacts=" << startup_contacts << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
