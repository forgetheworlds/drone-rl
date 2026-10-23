#pragma once

// Whole-body route search that only executes the prefix supported by measured free cells.

#include "known_space_backup.hpp"
#include "surface_watch.hpp"

namespace safe_nav {

class MeasuredRoute {
  public:
    const SurfaceWatch *surface_guard = nullptr;
    int diagnostic_expanded = 0;
    bool diagnostic_complete = false;
    Vec3 diagnostic_hypothesis{};
    Vec3 diagnostic_target() const { return target; }
    void use_continuous_arrival(bool enabled) { continuous_arrival = enabled; }
    void observe_goal_progress(Vec3 position, Vec3 goal) {
        if (!continuous_arrival)
            return;
        float distance = norm(goal - position);
        if (!arrival_progress_initialized || norm(goal - arrival_progress_goal) > .25f) {
            arrival_progress_initialized = true;
            arrival_progress_goal = goal;
            arrival_best_distance = distance;
            arrival_no_gain = 0;
            arrival_recovery_active = false;
        }
        if (distance >= arrival_sampling_range) {
            arrival_best_distance = distance;
            arrival_no_gain = 0;
        } else if (distance < arrival_best_distance - Navigator::resolution * .25f) {
            arrival_best_distance = distance;
            arrival_no_gain = 0;
        } else if (++arrival_no_gain >= 80)
            arrival_recovery_active = true;
    }
    bool needs_arrival_view() const { return arrival_needs_view; }
    Vec3 arrival_view_focus() const { return arrival_focus; }
    Vec3 planned_turn() const { return turn_direction; }

    void use_body_evidence_recovery() {
        if (!weighted) {
            weighted = true;
            active = false;
        }
    }
    bool evidence_recovery_active() const { return weighted; }

    Command propose(Navigator &map, const Pose &pose, Vec3 goal) {
        planning_goal = goal;
        arrival_needs_view = false;
        if (norm(goal - last_goal) > .25f) {
            active = false;
            arrival_recovery_active = false;
            last_goal = goal;
        }
        Vec3 first = map.map_cell_center(0) - Vec3{.125f, .125f, .125f};
        Vec3 search_goal{std::clamp(goal.x, first.x + .5f,
                                    first.x + Navigator::nx * Navigator::resolution - .5f),
                         std::clamp(goal.y, first.y + .5f,
                                    first.y + Navigator::ny * Navigator::resolution - .5f),
                         std::clamp(goal.z, first.z + .5f,
                                    first.z + Navigator::nz * Navigator::resolution - .5f)};
        if (active &&
            (norm(target - pose.position) < .15f || !segment_clear(map, pose.position, target)))
            active = false;
        if (segment_clear(map, pose.position, goal)) {
            target = goal;
            active = true;
            turn_direction = {};
        }
        if (continuous_arrival && arrival_recovery_active &&
            norm(goal - pose.position) < arrival_sampling_range) {
            float best = std::numeric_limits<float>::infinity();
            Vec3 arrival{};
            bool found = false;

            for (int z = -1; z <= 1; z++)
                for (int y = -1; y <= 1; y++)
                    for (int x = -1; x <= 1; x++) {
                        Vec3 offset{x * .075f, y * .075f, z * .075f}, candidate = goal + offset;
                        float rank = norm(offset) + .01f * norm(candidate - pose.position);
                        if (rank < best && segment_clear(map, pose.position, candidate)) {
                            best = rank;
                            arrival = candidate;
                            found = true;
                        }
                    }
            if (found) {
                target = arrival;
                active = true;
                turn_direction = {};
            } else {
                auto support = body_coverage(map, goal, Navigator::body_radius + Navigator::margin);
                if (support.unknown_cells > 0 && support.occupied_cells == 0 &&
                    support.outside_cells == 0) {
                    arrival_focus = support.unknown_position_sum * (1.f / support.unknown_cells);
                    arrival_needs_view = true;
                }
            }
        }
        if (!active)
            search(map, pose, search_goal);
        Command command;
        if (!active)
            return command;
        Vec3 difference = target - pose.position;
        float distance = norm(difference);
        command.world_velocity =
            difference * std::min(1.f, Navigator::max_speed / std::max(distance, 1e-6f));
        command.path_found = true;
        if (std::hypot(difference.x, difference.y) > .1f) {
            float yaw = std::atan2(pose.rotation[3], pose.rotation[0]);
            command.yaw_rate =
                std::clamp(2 * wrap(std::atan2(difference.y, difference.x) - yaw), -.5f, .5f);
        }
        return command;
    }
    void reset() {
        active = false;
        weighted = false;
        arrival_recovery_active = false;
        arrival_progress_initialized = false;
        arrival_no_gain = 0;
        last_goal = {};
        turn_direction = {};
    }

  private:
    static constexpr float arrival_sampling_range = 1.f;
    class Frontier {
      public:
        explicit Frontier(Navigator::RouteSearchStorage data) : data(data) {
            std::fill(data.costs, data.costs + Navigator::cells, 65535);
            std::fill(data.positions, data.positions + Navigator::cells, -1);
            std::fill(data.parent_directions, data.parent_directions + Navigator::cells, 255);
        }
        bool empty() const { return size == 0; }
        void push(int cell) {
            int pos = data.positions[cell];
            if (pos < 0) {
                pos = size++;
                data.frontier[pos] = uint16_t(cell);
                data.positions[cell] = int16_t(pos);
            }
            while (pos && earlier(data.frontier[pos], data.frontier[(pos - 1) / 2])) {
                exchange(pos, (pos - 1) / 2);
                pos = (pos - 1) / 2;
            }
        }
        int pop() {
            int cell = data.frontier[0];
            data.positions[cell] = -2;
            --size;
            if (size) {
                data.frontier[0] = data.frontier[size];
                data.positions[data.frontier[0]] = 0;
                int pos = 0;
                while (pos * 2 + 1 < size) {
                    int child = pos * 2 + 1;
                    if (child + 1 < size && earlier(data.frontier[child + 1], data.frontier[child]))
                        ++child;
                    if (!earlier(data.frontier[child], data.frontier[pos]))
                        break;
                    exchange(pos, child);
                    pos = child;
                }
            }
            return cell;
        }

      private:
        Navigator::RouteSearchStorage data;
        int size = 0;
        bool earlier(int a, int b) const {
            return data.costs[a] == data.costs[b] ? a < b : data.costs[a] < data.costs[b];
        }
        void exchange(int a, int b) {
            std::swap(data.frontier[a], data.frontier[b]);
            data.positions[data.frontier[a]] = int16_t(a);
            data.positions[data.frontier[b]] = int16_t(b);
        }
    };
    std::array<int8_t, Navigator::cells> clear_cache{};
    Vec3 target{}, last_goal{}, turn_direction{}, planning_goal{};
    bool active = false, weighted = false, continuous_arrival = false, arrival_needs_view = false,
         arrival_recovery_active = false;
    Vec3 arrival_focus{}, arrival_progress_goal{};
    float arrival_best_distance = 0;
    int arrival_no_gain = 0;
    bool arrival_progress_initialized = false;

    static int previous(int cell, const uint8_t *parent) {
        int direction = parent[cell];
        if (direction >= 6)
            return -1;
        int stride = direction / 2 == 0   ? 1
                     : direction / 2 == 1 ? Navigator::nx
                                          : Navigator::nx * Navigator::ny;
        return cell - stride * (direction % 2 ? 1 : -1);
    }
    bool segment_clear(const Navigator &map, Vec3 from, Vec3 to) const {
        return (!surface_guard || surface_guard->segment_forecast_clear(from, to, .8f)) &&
               swept_body_known_free(map, from, to, Navigator::body_radius + Navigator::margin);
    }
    bool center_clear(const Navigator &map, int cell) {
        if (surface_guard && !surface_guard->forecast_clear(map.map_cell_center(cell), .8f))
            return false;
        if (!clear_cache[cell]) {

            Vec3 center = map.map_cell_center(cell);
            float padding = norm(center - planning_goal) > .5f ? Navigator::resolution * .5f : 0;
            auto coverage =
                body_coverage(map, center, Navigator::body_radius + Navigator::margin + padding);
            clear_cache[cell] = coverage.occupied_cells || coverage.outside_cells
                                    ? -1
                                    : (coverage.known_free() ? 1 : 2);
        }
        return clear_cache[cell] > 0;
    }
    void search(Navigator &map, const Pose &pose, Vec3 goal) {
        diagnostic_expanded = 0;
        diagnostic_complete = false;
        diagnostic_hypothesis = goal;
        turn_direction = {};
        clear_cache.fill(0);
        int seed = -1;
        float nearest = 1e20f;
        for (int i = 0; i < Navigator::cells; i++) {
            float distance = norm(map.map_cell_center(i) - pose.position);
            if (distance >= nearest || distance > .5f || !center_clear(map, i))
                continue;
            if (segment_clear(map, pose.position, map.map_cell_center(i))) {
                nearest = distance;
                seed = i;
            }
        }
        if (seed < 0)
            return;
        auto scratch = map.route_search_storage();
        Frontier frontier(scratch);
        scratch.costs[seed] = 0;
        scratch.parent_directions[seed] = 254;
        int best = -1;
        int head = 0, tail = 0;
        if (weighted)
            frontier.push(seed);
        else
            scratch.frontier[tail++] = uint16_t(seed);
        while (weighted ? !frontier.empty() : head < tail) {
            int cell = weighted ? frontier.pop() : scratch.frontier[head++];
            ++diagnostic_expanded;
            if (norm(map.map_cell_center(cell) - goal) < .25f) {
                best = cell;
                diagnostic_complete = true;
                break;
            }
            int x = cell % Navigator::nx, y = (cell / Navigator::nx) % Navigator::ny,
                z = cell / (Navigator::nx * Navigator::ny);
            for (int axis = 0; axis < 3; axis++)
                for (int sign : {-1, 1}) {
                    int xx = x + (axis == 0 ? sign : 0), yy = y + (axis == 1 ? sign : 0),
                        zz = z + (axis == 2 ? sign : 0);
                    if (xx < 0 || xx >= Navigator::nx || yy < 0 || yy >= Navigator::ny || zz < 0 ||
                        zz >= Navigator::nz)
                        continue;
                    int next = (zz * Navigator::ny + yy) * Navigator::nx + xx;
                    if (scratch.positions[next] == -2 || !center_clear(map, next))
                        continue;
                    if (!weighted && scratch.parent_directions[next] != 255)
                        continue;
                    uint16_t cost =
                        uint16_t(scratch.costs[cell] + (weighted ? clear_cache[next] : 1));
                    if (weighted && cost >= scratch.costs[next])
                        continue;
                    scratch.costs[next] = cost;
                    scratch.parent_directions[next] = uint8_t(axis * 2 + (sign > 0 ? 1 : 0));
                    if (weighted)
                        frontier.push(next);
                    else
                        scratch.frontier[tail++] = uint16_t(next);
                }
        }
        if (best < 0)
            return;
        int toward = best, current = previous(best, scratch.parent_directions);
        while (current >= 0) {
            int before = previous(current, scratch.parent_directions);
            if (before < 0)
                break;
            Vec3 incoming = map.map_cell_center(current) - map.map_cell_center(before),
                 outgoing = map.map_cell_center(toward) - map.map_cell_center(current);
            if (norm(incoming - outgoing) > .1f)
                turn_direction = unit(outgoing);
            toward = current;
            current = before;
        }
        int next = best;
        while (previous(next, scratch.parent_directions) >= 0 &&
               !segment_clear(map, pose.position, map.map_cell_center(next)))
            next = previous(next, scratch.parent_directions);

        if (next == seed && (!weighted || norm(map.map_cell_center(seed) - pose.position) <=
                                              Navigator::resolution * .25f)) {
            next = best;
            while (previous(next, scratch.parent_directions) >= 0 &&
                   previous(previous(next, scratch.parent_directions), scratch.parent_directions) >=
                       0)
                next = previous(next, scratch.parent_directions);
        }
        target = map.map_cell_center(next);
        active = true;
    }
};
} // namespace safe_nav
