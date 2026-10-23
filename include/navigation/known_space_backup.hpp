#pragma once

// Forward-simulates candidate commands through the controller and motor model and accepts
// them only if the whole body, including a braking tail, stays inside observed free space.

#include <cstdio>
#include <cstdlib>
#include "navigator.hpp"
#include <type_traits>
#include <cstring>
#include <utility>

namespace safe_nav {
template <class Model, class = void> struct HasSurfaceForecast : std::false_type {};
template <class Model>
struct HasSurfaceForecast<
    Model, std::void_t<decltype(std::declval<const Model &>().forecast_clear(Vec3{}, 0.f))>>
    : std::true_type {};
template <class Model>
inline bool surface_forecast_clear(const Model &actual, Vec3 point, float seconds) {
    if constexpr (HasSurfaceForecast<Model>::value)
        return actual.forecast_clear(point, seconds);
    return true;
}

struct BodyCoverage {
    int free_cells = 0, unknown_cells = 0, occupied_cells = 0, outside_cells = 0;
    Vec3 unknown_position_sum{};
    bool known_free() const {
        return unknown_cells == 0 && occupied_cells == 0 && outside_cells == 0;
    }
};

inline BodyCoverage body_coverage(const Navigator &map, Vec3 position, float radius) {
    BodyCoverage result;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
        !std::isfinite(radius) || radius <= 0) {
        result.outside_cells = 1;
        return result;
    }
    const float size = Navigator::resolution;
    Vec3 first = map.map_cell_center(0) - Vec3{size * .5f, size * .5f, size * .5f};
    if (!std::isfinite(first.x) || !std::isfinite(first.y) || !std::isfinite(first.z) ||
        position.x - radius <= first.x || position.y - radius <= first.y ||
        position.z - radius <= first.z || position.x + radius >= first.x + Navigator::nx * size ||
        position.y + radius >= first.y + Navigator::ny * size ||
        position.z + radius >= first.z + Navigator::nz * size) {
        result.outside_cells = 1;
        return result;
    }
    int low[3], high[3];
    float point[3] = {position.x, position.y, position.z};
    float origin[3] = {first.x, first.y, first.z};
    for (int i = 0; i < 3; i++) {
        low[i] = int(std::floor((point[i] - radius - origin[i]) / size)) - 1;
        high[i] = int(std::floor((point[i] + radius - origin[i]) / size));
    }
    for (int z = low[2]; z <= high[2]; z++)
        for (int y = low[1]; y <= high[1]; y++)
            for (int x = low[0]; x <= high[0]; x++) {
                int cell[3] = {x, y, z};
                float distance2 = 0;
                for (int i = 0; i < 3; i++) {
                    float lo = origin[i] + cell[i] * size;
                    float d = point[i] - std::clamp(point[i], lo, lo + size);
                    distance2 += d * d;
                }
                if (distance2 > radius * radius)
                    continue;
                if (x < 0 || x >= Navigator::nx || y < 0 || y >= Navigator::ny || z < 0 ||
                    z >= Navigator::nz) {
                    ++result.outside_cells;
                    continue;
                }
                int state = map.map_cell_state((z * Navigator::ny + y) * Navigator::nx + x);
                int coarse = (z * Navigator::ny + y) * Navigator::nx + x;
                uint8_t mask = map.map_cell_free_mask(coarse);
                if (state < 0)
                    ++result.free_cells;
                else if (state > 0)
                    ++result.occupied_cells;
                else if (mask == 0) {
                    ++result.unknown_cells;
                    result.unknown_position_sum =
                        result.unknown_position_sum + map.map_cell_center(coarse);
                } else
                    for (int part = 0; part < 8; part++) {
                        Vec3 center = map.map_subcell_center(coarse, part),
                             lo = center - Vec3{.0625f, .0625f, .0625f};
                        Vec3 d = {position.x - std::clamp(position.x, lo.x, lo.x + .125f),
                                  position.y - std::clamp(position.y, lo.y, lo.y + .125f),
                                  position.z - std::clamp(position.z, lo.z, lo.z + .125f)};
                        if (d.x * d.x + d.y * d.y + d.z * d.z > radius * radius)
                            continue;
                        if (mask & (1u << part))
                            ++result.free_cells;
                        else {
                            ++result.unknown_cells;
                            result.unknown_position_sum = result.unknown_position_sum + center;
                        }
                    }
            }
    return result;
}

inline float segment_box_distance_squared(Vec3 from, Vec3 to, Vec3 lower, float size) {
    float start[3] = {from.x - lower.x, from.y - lower.y, from.z - lower.z},
          direction[3] = {to.x - from.x, to.y - from.y, to.z - from.z};
    float lo[3] = {0, 0, 0}, cuts[8] = {0, 1};
    int count = 2;
    for (int axis = 0; axis < 3; axis++) {
        if (std::fabs(direction[axis]) < 1e-12f)
            continue;
        for (float face : {lo[axis], lo[axis] + size}) {
            float t = (face - start[axis]) / direction[axis];
            if (t > 0 && t < 1)
                cuts[count++] = t;
        }
    }
    std::sort(cuts, cuts + count);
    float best = std::numeric_limits<float>::infinity();
    auto distance_at = [&](float t) {
        float distance = 0;
        for (int axis = 0; axis < 3; axis++) {
            float point = start[axis] + direction[axis] * t;
            float offset = point - std::clamp(point, lo[axis], lo[axis] + size);
            distance += offset * offset;
        }
        return distance;
    };
    for (int interval = 0; interval < count - 1; interval++) {
        float a = cuts[interval], b = cuts[interval + 1], mid = (a + b) * .5f;
        float quadratic = 0, linear = 0;
        for (int axis = 0; axis < 3; axis++) {
            float point = start[axis] + direction[axis] * mid;
            if (point >= lo[axis] && point <= lo[axis] + size)
                continue;
            float face = point < lo[axis] ? lo[axis] : lo[axis] + size;
            quadratic += direction[axis] * direction[axis];
            linear += (start[axis] - face) * direction[axis];
        }
        best = std::min(best, std::min(distance_at(a), distance_at(b)));
        if (quadratic > 0)
            best = std::min(best, distance_at(std::clamp(-linear / quadratic, a, b)));
    }
    return best;
}

inline bool swept_body_known_free(const Navigator &map, Vec3 from, Vec3 to, float radius,
                                  BodyCoverage *failure = nullptr) {
    if (failure)
        *failure = {};
    if (!std::isfinite(from.x) || !std::isfinite(from.y) || !std::isfinite(from.z) ||
        !std::isfinite(to.x) || !std::isfinite(to.y) || !std::isfinite(to.z) ||
        !std::isfinite(radius) || radius <= 0) {
        if (failure)
            failure->outside_cells = 1;
        return false;
    }
    constexpr float size = Navigator::resolution;
    Vec3 origin = map.map_cell_center(0) - Vec3{size * .5f, size * .5f, size * .5f};
    float first[3] = {from.x, from.y, from.z}, last[3] = {to.x, to.y, to.z};
    float base[3] = {origin.x, origin.y, origin.z};
    int dimensions[3] = {Navigator::nx, Navigator::ny, Navigator::nz}, low[3], high[3];
    for (int axis = 0; axis < 3; axis++) {
        float lo = std::min(first[axis], last[axis]) - radius,
              hi = std::max(first[axis], last[axis]) + radius;
        if (lo <= base[axis] || hi >= base[axis] + dimensions[axis] * size) {
            if (failure)
                failure->outside_cells = 1;
            return false;
        }
        low[axis] = std::max(0, int(std::floor((lo - base[axis]) / size)) - 1);
        high[axis] = std::min(dimensions[axis] - 1, int(std::floor((hi - base[axis]) / size)));
    }
    BodyCoverage unsupported;
    for (int z = low[2]; z <= high[2]; z++)
        for (int y = low[1]; y <= high[1]; y++)
            for (int x = low[0]; x <= high[0]; x++) {
                int cell = (z * Navigator::ny + y) * Navigator::nx + x,
                    state = map.map_cell_state(cell);
                if (state < 0)
                    continue;
                Vec3 lower = origin + Vec3{x * size, y * size, z * size};

                if (segment_box_distance_squared(from, to, lower, size) > radius * radius + 1e-6f)
                    continue;
                uint8_t mask = map.map_cell_free_mask(cell);
                if (state > 0)
                    ++unsupported.occupied_cells;
                else if (mask == 0) {
                    ++unsupported.unknown_cells;
                    unsupported.unknown_position_sum =
                        unsupported.unknown_position_sum + map.map_cell_center(cell);
                } else
                    for (int part = 0; part < 8; part++) {
                        if (mask & (1u << part))
                            continue;
                        Vec3 center = map.map_subcell_center(cell, part),
                             lo = center - Vec3{.0625f, .0625f, .0625f};
                        if (segment_box_distance_squared(from, to, lo, .125f) >
                            radius * radius + 1e-6f)
                            continue;
                        ++unsupported.unknown_cells;
                        unsupported.unknown_position_sum =
                            unsupported.unknown_position_sum + center;
                    }
            }
    if (failure)
        *failure = unsupported;
    return unsupported.known_free();
}

enum class BackupStatus { accepted, continuing_backup, not_ready, invalidated };
struct BackupDecision {
    Command command{};
    BackupStatus status = BackupStatus::not_ready;
    bool verified = false;
    int model_steps = 0;
    Vec3 inspection_focus{};
    bool has_inspection_focus = false;
};

class KnownSpaceBackup {
  public:
    static constexpr int horizon_steps = 80, prefix_steps = 5;
    explicit KnownSpaceBackup(bool inspect = false, bool allow_braking = false)
        : inspect_unknown(inspect), allow_feedback_braking(allow_braking) {}
    template <class Model>
    BackupDecision choose(const Navigator &map, const Model &vehicle, Command desired,
                          Model &workspace) {
        BackupDecision result;
        if (!body_coverage(map, vehicle.pose().position, Navigator::body_radius + Navigator::margin)
                 .known_free()) {
            result.status = active ? BackupStatus::invalidated : BackupStatus::not_ready;
            active = false;
            return result;
        }
        const bool finite_request =
            std::isfinite(desired.world_velocity.x) && std::isfinite(desired.world_velocity.y) &&
            std::isfinite(desired.world_velocity.z) && std::isfinite(desired.yaw_rate);
        bool have_unknown_hint = false;
        Vec3 unknown_hint{};
        struct Trial {
            float fraction;
            bool feedback;
        };
        const Trial trials[] = {{1, false},  {.5f, false}, {1, true},
                                {.5f, true}, {0, false},   {0, true}};
        struct Checked {
            Command command;
            bool feedback;
        };
        Checked checked[6];
        int checked_count = 0;
        for (Trial trial : trials) {
            if (trial.feedback && !allow_feedback_braking)
                continue;
            float fraction = trial.fraction;
            if (!finite_request)
                break;
            Command candidate = desired;
            candidate.world_velocity = candidate.world_velocity * fraction;

            if (fraction == 0 && inspect_unknown && have_unknown_hint) {
                Vec3 direction = unknown_hint - vehicle.pose().position;
                if (std::hypot(direction.x, direction.y) > .05f) {
                    auto pose = vehicle.pose();
                    float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
                    candidate.yaw_rate =
                        std::clamp(2 * wrap(std::atan2(direction.y, direction.x) - yaw), -.5f, .5f);
                }
            }
            bool duplicate = false;
            static_assert(sizeof(Vec3) == 3 * sizeof(float),
                          "Velocity cache requires packed float coordinates");
            for (int i = 0; i < checked_count; i++) {
                const auto &earlier = checked[i];
                if (earlier.feedback == trial.feedback &&
                    std::memcmp(&earlier.command.world_velocity, &candidate.world_velocity,
                                sizeof(Vec3)) == 0 &&
                    std::memcmp(&earlier.command.yaw_rate, &candidate.yaw_rate, sizeof(float)) ==
                        0) {
                    duplicate = true;
                    break;
                }
            }

            if (duplicate)
                continue;
            checked[checked_count++] = {candidate, trial.feedback};
            workspace = vehicle;
            bool valid = true;
            for (int step = 0; step < horizon_steps; step++) {
                Command command = step < prefix_steps ? candidate : Command{};
                Vec3 before = workspace.pose().position;
                ++result.model_steps;
                if (!advance_model(workspace, command, trial.feedback) ||
                    !surface_forecast_clear(vehicle, workspace.pose().position,
                                            (step + 1) * .01f)) {
                    valid = false;
                    break;
                }
                BodyCoverage unsupported;
                if (!swept_body_known_free(map, before, workspace.pose().position,
                                           Navigator::body_radius + Navigator::margin,
                                           &unsupported)) {
                    if (std::getenv("OBS_DEBUG_SWEEP"))
                        std::fprintf(
                            stderr,
                            "SWEEP_FAIL step=%d cand=(%.3f,%.3f,%.3f) yaw=%.3f "
                            "before=(%.3f,%.3f,%.3f) after=(%.3f,%.3f,%.3f) unsupported_free=%d "
                            "unknown=%d occupied=%d out=%d hint=(%.3f,%.3f,%.3f)\n",
                            step, candidate.world_velocity.x, candidate.world_velocity.y,
                            candidate.world_velocity.z, candidate.yaw_rate, before.x, before.y,
                            before.z, workspace.pose().position.x, workspace.pose().position.y,
                            workspace.pose().position.z, unsupported.free_cells,
                            unsupported.unknown_cells, unsupported.occupied_cells,
                            unsupported.outside_cells,
                            unsupported.unknown_cells > 0
                                ? unsupported.unknown_position_sum.x / unsupported.unknown_cells
                                : 0.f,
                            unsupported.unknown_cells > 0
                                ? unsupported.unknown_position_sum.y / unsupported.unknown_cells
                                : 0.f,
                            unsupported.unknown_cells > 0
                                ? unsupported.unknown_position_sum.z / unsupported.unknown_cells
                                : 0.f);
                    if (!have_unknown_hint && unsupported.unknown_cells > 0) {
                        unknown_hint =
                            unsupported.unknown_position_sum * (1.0f / unsupported.unknown_cells);
                        have_unknown_hint = true;
                        result.inspection_focus = unknown_hint;
                        result.has_inspection_focus = true;
                    }
                    valid = false;
                    break;
                }
            }
            BodyCoverage terminal_unsupported;
            bool captured =
                valid && swept_body_known_free(
                             map, workspace.pose().position, workspace.reference_position(),
                             Navigator::body_radius + Navigator::margin, &terminal_unsupported);
            if (!captured && !have_unknown_hint && terminal_unsupported.unknown_cells > 0) {
                unknown_hint = terminal_unsupported.unknown_position_sum *
                               (1.0f / terminal_unsupported.unknown_cells);
                have_unknown_hint = true;
                result.inspection_focus = unknown_hint;
                result.has_inspection_focus = true;
            }
            if (captured && norm(workspace.velocity()) < .12f) {
                for (int step = 0; step < horizon_steps; step++)
                    commands[step] = step < prefix_steps ? candidate : Command{};
                cursor = 0;
                active = true;
                feedback_tail = trial.feedback;
                result.command = candidate;
                result.status = BackupStatus::accepted;
                result.verified = true;
                return result;
            }
        }

        if (active && cursor < horizon_steps) {
            workspace = vehicle;
            bool valid = true;
            for (int step = cursor; step < horizon_steps; step++) {
                Vec3 before = workspace.pose().position;
                ++result.model_steps;
                if (!advance_model(workspace, commands[step], feedback_tail) ||
                    !surface_forecast_clear(vehicle, workspace.pose().position,
                                            (step - cursor + 1) * .01f)) {
                    valid = false;
                    break;
                }
                if (!swept_body_known_free(map, before, workspace.pose().position,
                                           Navigator::body_radius + Navigator::margin)) {
                    valid = false;
                    break;
                }
            }
            BodyCoverage terminal_unsupported;
            bool captured =
                valid && swept_body_known_free(
                             map, workspace.pose().position, workspace.reference_position(),
                             Navigator::body_radius + Navigator::margin, &terminal_unsupported);
            if (!captured && !have_unknown_hint && terminal_unsupported.unknown_cells > 0) {
                unknown_hint = terminal_unsupported.unknown_position_sum *
                               (1.0f / terminal_unsupported.unknown_cells);
                have_unknown_hint = true;
                result.inspection_focus = unknown_hint;
                result.has_inspection_focus = true;
            }
            if (captured && norm(workspace.velocity()) < .12f) {
                result.command = commands[cursor];
                result.status = BackupStatus::continuing_backup;
                result.verified = true;
                return result;
            }
            result.status = BackupStatus::invalidated;
        }
        active = false;
        return result;
    }
    bool command_for_next_tick(Command &command) const {
        if (!active || cursor >= horizon_steps || feedback_tail)
            return false;
        command = commands[cursor];
        return true;
    }
    bool command_for_next_tick(Command &command, Vec3 measured_velocity) const {
        if (!active || cursor >= horizon_steps)
            return false;
        if (!std::isfinite(measured_velocity.x) || !std::isfinite(measured_velocity.y) ||
            !std::isfinite(measured_velocity.z) || !std::isfinite(norm(measured_velocity)))
            return false;
        command = effective_command(commands[cursor], measured_velocity, feedback_tail);
        return true;
    }
    bool using_feedback_braking() const { return active && feedback_tail; }
    void invalidate() { active = false; }
    void advance_physics_step() {
        if (active && ++cursor >= horizon_steps)
            active = false;
    }
    void reset() {
        active = false;
        cursor = 0;
    }

  private:
    static Command effective_command(Command command, Vec3 velocity, bool feedback) {
        if (feedback && norm(command.world_velocity) == 0) {
            float speed = norm(velocity);
            command.world_velocity =
                velocity * (-std::min(1.0f, Navigator::max_speed / std::max(speed, 1e-8f)));
        }
        return command;
    }
    template <class Model>
    static bool advance_model(Model &model, Command command, bool feedback = false) {
        command = effective_command(command, model.velocity(), feedback);
        if constexpr (std::is_same<decltype(model.substep(command)), bool>::value)
            return model.substep(command);
        else {
            model.substep(command);
            return true;
        }
    }
    std::array<Command, horizon_steps> commands{};
    int cursor = 0;
    bool active = false;
    bool inspect_unknown = false;
    bool allow_feedback_braking = false, feedback_tail = false;
};
} // namespace safe_nav
