#pragma once

// Top-level navigator: combines mapping, route search, view planning and certified motion.

#include "known_space_views.hpp"
#include "measured_route.hpp"
#include "surface_watch.hpp"
#include "moving_escape.hpp"
#include "near_surface_view.hpp"

namespace safe_nav {

class ObservedRouteNavigator {
  public:
    bool attention_active() const { return near_surface_view && near_surface_view->has_focus(); }
    Vec3 attention_focus() const {
        return near_surface_view ? near_surface_view->focus_position() : Vec3{};
    }
    bool attention_retreating() const { return clearing_surface; }
    Vec3 attention_target() const { return surface_clearance_target; }
    Command diagnostic_route{};
    BackupDecision diagnostic_backup{};
    static constexpr int max_model_steps_per_plan = 66 * KnownSpaceBackup::horizon_steps;
    explicit ObservedRouteNavigator(bool allow_braking = false, bool allow_recovery = false)
        : backup(true, allow_braking), recovery_allowed(allow_recovery) {
        if (allow_recovery)
            observed_map.track_subcells(true);
    }
    void use_joint_arrival_view(bool enabled) { joint_arrival_view = enabled; }
    void use_near_surface_view(NearSurfaceView *view) {
        near_surface_view = view;
        if (view)
            view->reset();
    }
    void use_surface_watch(SurfaceWatch *watch, bool view_only = false) {
        surface_watch = watch;
        surface_view_only = view_only;
        if (watch)
            watch->reset();
        if (measured_route)
            measured_route->surface_guard = view_only ? nullptr : watch;
    }
    void use_measured_route(MeasuredRoute *route) {
        measured_route = route;
        if (route) {
            route->reset();
            route->surface_guard = surface_view_only ? nullptr : surface_watch;
        }
    }
    void allow_refinement_recovery(bool allowed) {
        recovery_allowed = allowed;
        if (allowed)
            observed_map.track_subcells(true);
    }
    int next_plan_model_step_limit() const {
        return max_model_steps_per_plan * (recovery_allowed && !refinement_active ? 2 : 1);
    }
    void reset(Vec3 position) {
        observed_map.reset(position);
        observed_map.enable_refined_free_space(false);
        observed_map.enable_fine_body_checks(false);
        observed_map.track_subcells(recovery_allowed);
        refinement_active = false;
        views.reset();
        backup.reset();
        escape.reset();
        escape_active = false;
        viewing_surface = false;
        progress_initialized = false;
        proposal_initialized = false;
        clearing_surface = false;
        if (surface_watch)
            surface_watch->reset();
        if (near_surface_view)
            near_surface_view->reset();
        if (measured_route)
            measured_route->reset();
    }
    void observe(const Pose &capture_pose, const std::array<float, 320> &depth,
                 std::array<uint8_t, 40> *changed_output = nullptr,
                 const std::array<float, 320> *tile_minimum = nullptr) {
        if (surface_watch)
            surface_watch->observe(observed_map, capture_pose, depth, changed_output, tile_minimum);
        if (near_surface_view)
            near_surface_view->observe(observed_map, capture_pose, depth);
        observed_map.observe(capture_pose, depth);
    }
    void enable_fine_body_checks(bool enabled) { observed_map.enable_fine_body_checks(enabled); }
    void enable_refined_free_space(bool enabled) {
        observed_map.enable_refined_free_space(enabled);
    }
    const Navigator &map() const { return observed_map; }
    void handover_map(const Navigator &acquired_map) {
        observed_map = acquired_map;
        observed_map.enable_refined_free_space(false);
        observed_map.enable_fine_body_checks(false);
        if (recovery_allowed)
            observed_map.track_subcells(true);
        refinement_active = false;
        views.reset();
        backup.reset();
        escape.reset();
        escape_active = false;
        viewing_surface = false;
        progress_initialized = false;
        proposal_initialized = false;
        clearing_surface = false;
        if (surface_watch)
            surface_watch->reset();
        if (near_surface_view)
            near_surface_view->reset();
        if (measured_route)
            measured_route->reset();
    }
    template <class Model>
    BackupDecision check_handover(const Model &actual, Model &workspace) const {
        KnownSpaceBackup probe;
        return probe.choose(observed_map, actual, {}, workspace);
    }
    template <class Model> BackupDecision plan(const Model &actual, Vec3 goal, Model &workspace) {
        Vec3 position = actual.pose().position;
        if (!progress_initialized || norm(goal - progress_goal) > .25f) {
            progress_initialized = true;
            progress_goal = goal;
            progress_anchor = position;
            progress_ticks = 0;
            low_progress = false;
            surface_clearance_height = position.z;
            cell_progress_anchor = position;
            cell_progress_ticks = 0;
        }
        if (++progress_ticks >= 20) {
            low_progress = norm(position - progress_anchor) < Navigator::resolution * .25f;
            progress_anchor = position;
            progress_ticks = 0;
        }
        if (near_surface_view) {

            if (norm(position - cell_progress_anchor) >= Navigator::resolution) {
                cell_progress_anchor = position;
                cell_progress_ticks = 0;
            } else
                ++cell_progress_ticks;
            low_progress = low_progress || cell_progress_ticks >= 80;
        }
        if (measured_route)
            measured_route->observe_goal_progress(position, goal);
        int observation_steps = 0;
        Vec3 surface_focus;
        bool continue_view =
            viewing_surface && views.diagnostic_active() &&
            norm(views.diagnostic_target() - position) > Navigator::resolution * .25f;
        bool stale_surface = surface_watch && surface_watch->stale_focus(position, surface_focus);
        if (surface_watch && !surface_view_only && (stale_surface || continue_view)) {
            if (!continue_view) {
                views.reset();
                viewing_surface = true;
                latched_surface_focus = surface_focus;
            }
            Command view =
                views.propose(observed_map, actual.pose(), goal, latched_surface_focus, .5f);
            view = surface_watch->keep_changed_surface_visible(actual.pose(), view);
            auto trial = backup;
            auto observed = trial.choose(observed_map, actual, view, workspace);
            observation_steps = observed.model_steps;

            if (observed.verified && view.path_found) {
                backup = trial;
                escape_active = false;
                return observed;
            }
        } else if (viewing_surface) {
            views.reset();
            viewing_surface = false;
        }
        auto primary = plan_current_map(actual, goal, workspace);
        primary.model_steps += observation_steps;
        if (primary.verified || !recovery_allowed || refinement_active)
            return finish(actual, primary, workspace);
        int preceding_steps = primary.model_steps;
        observed_map.enable_refined_free_space(true);
        views.reset();
        auto recovered = plan_current_map(actual, goal, workspace);
        recovered.model_steps += preceding_steps;
        if (recovered.verified) {
            refinement_active = true;
            return finish(actual, recovered, workspace);
        }
        observed_map.enable_refined_free_space(false);
        primary.model_steps = recovered.model_steps;
        return finish(actual, primary, workspace);
    }
    template <class Model>
    BackupDecision plan_proposal(const Model &actual, Vec3 goal, Command proposal,
                                 Model &workspace) {
        float distance = norm(goal - actual.pose().position);
        if (!proposal_initialized || norm(goal - proposal_goal) > .25f) {
            proposal_initialized = true;
            proposal_goal = goal;
            proposal_best = distance;
            proposal_without_gain = 0;
            proposal_abandoned = false;
        }
        if (distance < proposal_best - Navigator::resolution * .25f) {
            proposal_best = distance;
            proposal_without_gain = 0;
        } else
            ++proposal_without_gain;

        constexpr int patience =
            int((Navigator::nx * Navigator::resolution * .5f) / Navigator::max_speed / .05f);
        if (proposal_without_gain >= patience)
            proposal_abandoned = true;
        int spent = 0;
        if (!proposal_abandoned) {
            auto trial = backup;
            auto chosen = trial.choose(observed_map, actual, proposal, workspace);
            spent = chosen.model_steps;
            if (chosen.verified &&
                (norm(chosen.command.world_velocity) > .01f || distance < .25f)) {
                backup = trial;
                escape_active = false;
                return chosen;
            }
        }
        auto fallback = plan(actual, goal, workspace);
        fallback.model_steps += spent;
        return fallback;
    }
    Vec3 local_waypoint() const {
        return measured_route ? measured_route->diagnostic_target() : Vec3{};
    }
    template <class Model>
    BackupDecision try_aligned_motion(const Model &actual, Command proposal, Model &workspace) {
        BackupDecision rejected;
        const float requested_speed = norm(proposal.world_velocity);
        if (!std::isfinite(requested_speed) || requested_speed <= .01f ||
            !std::isfinite(proposal.yaw_rate))
            return rejected;

        auto trial = backup;
        auto chosen = trial.choose(observed_map, actual, proposal, workspace);
        const float chosen_speed = norm(chosen.command.world_velocity);
        const float alignment = chosen.command.world_velocity.x * proposal.world_velocity.x +
                                chosen.command.world_velocity.y * proposal.world_velocity.y +
                                chosen.command.world_velocity.z * proposal.world_velocity.z;
        if (!chosen.verified || chosen_speed <= .01f ||
            alignment <= .5f * requested_speed * chosen_speed) {
            chosen.verified = false;
            chosen.status = BackupStatus::not_ready;
            return chosen;
        }
        backup = trial;
        escape_active = false;
        return chosen;
    }
    template <class Model>
    BackupDecision refine_motion(const Model &actual, BackupDecision route, Command proposal,
                                 Model &workspace) {
        auto trial = backup;
        auto candidate = trial.choose(observed_map, actual, proposal, workspace);
        int steps = route.model_steps + candidate.model_steps;
        if (candidate.verified && norm(candidate.command.world_velocity) > .01f) {
            backup = trial;
            route = candidate;
            escape_active = false;
        }
        route.model_steps = steps;
        return route;
    }
    template <class Model>
    BackupDecision plan_current_map(const Model &actual, Vec3 goal, Model &workspace) {
        auto pose = actual.pose();
        Command route = observed_map.plan(pose, actual.velocity(), goal);
        if (measured_route) {
            auto measured = measured_route->propose(observed_map, pose, goal);
            if (measured.path_found)
                route = measured;
        }
        bool arrival_view = false;
        if (measured_route && measured_route->needs_arrival_view()) {
            if (clearing_surface) {
                clearing_surface = false;
                views.reset();
            }
            Vec3 retained = near_surface_view ? near_surface_view->focus_position() : Vec3{};
            bool joint = joint_arrival_view && near_surface_view && near_surface_view->has_focus();
            auto inspection =
                views.propose(observed_map, pose, goal, measured_route->arrival_view_focus(), 0,
                              2 * Navigator::resolution, std::numeric_limits<float>::quiet_NaN(),
                              joint ? &retained : nullptr);
            if (inspection.path_found) {
                route = inspection;
                arrival_view = true;
            }
        }
        if (surface_watch)
            route = surface_watch->keep_changed_surface_visible(pose, route);
        if (near_surface_view)
            route = near_surface_view->constrain(pose, route);
        if (!arrival_view && near_surface_view && near_surface_view->has_focus()) {
            if (!clearing_surface && low_progress && norm(goal - pose.position) > .5f) {
                surface_clearance_min_range =
                    norm(near_surface_view->focus_position() - pose.position) +
                    Navigator::resolution;
                surface_clearance_height = pose.position.z;
                surface_clearance_target = pose.position;
                views.reset();
                clearing_surface = true;
            }
            if (clearing_surface) {
                if (norm(surface_clearance_target - pose.position) < Navigator::resolution * .25f &&
                    norm(near_surface_view->focus_position() - pose.position) >=
                        surface_clearance_min_range) {
                    clearing_surface = false;
                    views.reset();
                } else {

                    auto clearance =
                        views.propose(observed_map, pose, goal, near_surface_view->focus_position(),
                                      .5f, surface_clearance_min_range, surface_clearance_height);
                    if (clearance.path_found) {
                        surface_clearance_target = views.diagnostic_target();
                        route = near_surface_view->constrain(pose, clearance);
                    } else {
                        clearing_surface = false;
                        views.reset();
                    }
                }
            }
        } else
            clearing_surface = false;
        diagnostic_route = route;
        auto selected = backup.choose(observed_map, actual, route, workspace);
        diagnostic_backup = selected;
        if (clearing_surface || arrival_view) {

            if (selected.verified && norm(selected.command.world_velocity) < .01f) {
                Vec3 focus = arrival_view ? measured_route->arrival_view_focus()
                                          : near_surface_view->focus_position();
                auto edge = views.next_path_edge(pose, focus);
                if (near_surface_view)
                    edge = near_surface_view->constrain(pose, edge);
                auto trial = backup;
                auto edge_choice = trial.choose(observed_map, actual, edge, workspace);
                int steps = selected.model_steps + edge_choice.model_steps;
                if (edge_choice.verified && norm(edge_choice.command.world_velocity) > .01f) {
                    backup = trial;
                    selected = edge_choice;
                }
                selected.model_steps = steps;
            }
            return selected;
        }

        if (measured_route && selected.verified &&
            (norm(selected.command.world_velocity) < .01f ||
             (low_progress && norm(goal - pose.position) > .5f)) &&
            norm(route.world_velocity) > .05f) {
            Command horizontal = route;
            horizontal.world_velocity.z = 0;
            Command vertical;
            vertical.yaw_rate = route.yaw_rate;
            float height = measured_route->diagnostic_target().z - pose.position.z;
            if (std::fabs(height) > Navigator::resolution * .25f)
                vertical.world_velocity.z =
                    std::clamp(2 * height, -Navigator::max_speed * .5f, Navigator::max_speed * .5f);
            auto overhead =
                body_coverage(observed_map, pose.position + Vec3{0, 0, Navigator::resolution},
                              Navigator::body_radius + Navigator::margin);
            bool blocked_climb = route.world_velocity.z > 0 && overhead.occupied_cells > 0;
            Command turn = route;
            turn.world_velocity = measured_route->planned_turn() * (Navigator::max_speed * .5f);
            Command alternate_turn = turn;
            alternate_turn.world_velocity = alternate_turn.world_velocity * (-1);
            int component_index = 0;
            for (Command component : {horizontal, vertical, turn, alternate_turn}) {
                bool horizontal_component = component_index++ == 0;

                if (horizontal_component && !blocked_climb)
                    continue;
                if (norm(component.world_velocity) < .01f)
                    continue;
                auto trial = backup;
                auto candidate = trial.choose(observed_map, actual, component, workspace);
                int steps = selected.model_steps + candidate.model_steps;
                if (candidate.verified && norm(candidate.command.world_velocity) > .01f) {
                    backup = trial;
                    selected = candidate;
                    selected.model_steps = steps;
                    break;
                }
                selected.model_steps = steps;
            }
        }
        if (selected.verified && norm(selected.command.world_velocity) < .01f &&
            norm(route.world_velocity) > .05f && selected.has_inspection_focus) {
            Command view = views.propose(observed_map, pose, goal, selected.inspection_focus);
            int total_steps = selected.model_steps;
            auto trial = backup;
            auto alternative = trial.choose(observed_map, actual, view, workspace);
            total_steps += alternative.model_steps;
            if (alternative.verified && norm(alternative.command.world_velocity) > .01f) {
                backup = trial;
                selected = alternative;
            } else {
                Command edge = views.next_path_edge(pose, selected.inspection_focus);
                auto edge_trial = backup;
                auto edge_choice = edge_trial.choose(observed_map, actual, edge, workspace);
                total_steps += edge_choice.model_steps;
                if (edge_choice.verified && norm(edge_choice.command.world_velocity) > .01f) {
                    backup = edge_trial;
                    selected = edge_choice;
                    views.reset();
                } else if (alternative.verified) {
                    backup = trial;
                    selected = alternative;
                }
            }
            selected.model_steps = total_steps;
        }
        return selected;
    }
    template <class Model>
    BackupDecision inspect(const Model &actual, Command look, Model &workspace) {
        look.world_velocity = {};
        return finish(actual, backup.choose(observed_map, actual, look, workspace), workspace);
    }
    bool command_for_next_tick(Command &command) const {
        return escape_active ? escape.next(command) : backup.command_for_next_tick(command);
    }
    bool command_for_next_tick(Command &command, Vec3 measured_velocity) const {
        return escape_active ? escape.next(command)
                             : backup.command_for_next_tick(command, measured_velocity);
    }
    bool using_feedback_braking() const { return backup.using_feedback_braking(); }
    void advance_executed_tick() {
        if (escape_active)
            escape.advance();
        else
            backup.advance_physics_step();
    }
    void invalidate() {
        backup.invalidate();
        escape.reset();
        escape_active = false;
    }
    void next_goal() {
        proposal_initialized = false;
        views.reset();
        if (measured_route)
            measured_route->reset();
    }

  private:
    template <class Model>
    BackupDecision finish(const Model &actual, BackupDecision selected, Model &workspace) {
        escape_active = false;
        if (surface_watch && !surface_view_only && !selected.verified) {
            auto alternative = escape.choose(observed_map, actual, workspace, *surface_watch);
            int steps = selected.model_steps + alternative.model_steps;
            if (alternative.verified) {
                selected = alternative;
                escape_active = true;
            }
            selected.model_steps = steps;
        }
        return selected;
    }
    MovingEscape escape;
    Vec3 latched_surface_focus{};
    bool escape_active = false, viewing_surface = false;
    Vec3 proposal_goal{};
    float proposal_best = 0;
    int proposal_without_gain = 0;
    bool proposal_initialized = false, proposal_abandoned = false;
    Navigator observed_map;
    KnownSpaceViews views;
    Vec3 cell_progress_anchor{};
    int cell_progress_ticks = 0;
    float surface_clearance_min_range = 0, surface_clearance_height = 0;
    Vec3 surface_clearance_target{};
    bool clearing_surface = false, joint_arrival_view = false;
    NearSurfaceView *near_surface_view = nullptr;
    SurfaceWatch *surface_watch = nullptr;
    bool surface_view_only = false;
    MeasuredRoute *measured_route = nullptr;
    KnownSpaceBackup backup{true};
    Vec3 progress_anchor{}, progress_goal{};
    int progress_ticks = 0;
    bool progress_initialized = false, low_progress = false;
    bool recovery_allowed = false, refinement_active = false;
};
} // namespace safe_nav
