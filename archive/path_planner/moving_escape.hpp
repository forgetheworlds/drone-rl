#pragma once

// Short evasive manoeuvre away from an approaching surface, checked over its full duration.

#include "known_space_backup.hpp"
#include "surface_watch.hpp"

namespace safe_nav {

class MovingEscape {
  public:
    template <class Model>
    BackupDecision choose(const Navigator &map, const Model &actual, Model &workspace,
                          SurfaceWatch &watch) {
        BackupDecision result;
        Vec3 away = watch.away_direction(actual.pose().position);
        if (norm(away) < .1f ||
            !body_coverage(map, actual.pose().position, Navigator::body_radius + Navigator::margin)
                 .known_free())
            return result;
        const Vec3 directions[] = {away, {-away.y, away.x, 0}, {away.y, -away.x, 0}};
        for (Vec3 direction : directions) {
            Command candidate;
            candidate.world_velocity = direction * Navigator::max_speed;
            candidate.path_found = true;
            candidate = watch.keep_changed_surface_visible(actual.pose(), candidate);
            workspace = actual;
            bool valid = true;
            for (int step = 0; step < KnownSpaceBackup::horizon_steps; step++) {
                Vec3 before = workspace.pose().position;
                ++result.model_steps;
                if (!workspace.substep(candidate) ||
                    !surface_forecast_clear(actual, workspace.pose().position, (step + 1) * .01f) ||
                    !swept_body_known_free(map, before, workspace.pose().position,
                                           Navigator::body_radius + Navigator::margin)) {
                    valid = false;
                    break;
                }
            }
            if (valid && watch.separating(workspace.pose().position, workspace.velocity(), .8f)) {
                command = candidate;
                remaining = KnownSpaceBackup::horizon_steps;
                result.command = candidate;
                result.verified = true;
                result.status = BackupStatus::accepted;
                return result;
            }
        }
        remaining = 0;
        return result;
    }
    bool next(Command &output) const {
        if (remaining <= 0)
            return false;
        output = command;
        return true;
    }
    void advance() {
        if (remaining > 0)
            --remaining;
    }
    void reset() { remaining = 0; }

  private:
    Command command{};
    int remaining = 0;
};
} // namespace safe_nav
