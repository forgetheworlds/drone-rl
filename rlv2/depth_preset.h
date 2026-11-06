#pragma once

// The "measured" stereo preset: per-ray depth errors taken from the HyperDrone depth benchmark
// (rlv2/depth_bench, bm160_temporal_r3 on OV2640-like images), applied to the grader's 20 x 16
// rays by the surface each ray hits. CPU only (grade.cpp); the Metal trainer keeps
// vx_stereo_ray. See docs/DEPTH.md.
//
//   range 3 m: a textured surface past it gives "free to 3 m" (VX_FAR_FREE), anything else nothing
//   noise sigma = k z^2, k = 0.026
//   holes: textureless walls 95%, floor / ceiling 90%, textured surfaces 35% (ray level)
//   phantoms: 3% of rays return a false near hit at 0.3-2.3 m
//   false-free: 1.5% of solid rays report 1.3-2.5x too far; thin (one-voxel) obstacles are seen
//   49% of the time, look through to free space 6%, give nothing otherwise
//
// Which wall tiles are textureless is fixed per mission: each 2 m tile of each wall plane is
// textureless with probability `textureless` (0.5 unless RLV2_TEXTURELESS is set). Moving boxes
// are textured.

#include "sensors.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>

#define VX_FAR_FREE 11.0f  // stereo matched, but past the range cap: free up to the cap
#define LM_FAR_FREE_TO 2.75f  // local_map.h carves such rays to the 3 m cap less its 0.25 m margin

struct DepthPreset {
    float range = 3.0f, k = .026f;
    float textureless = .5f, hole_textureless = .95f, hole_floor = .90f, hole_textured = .35f;
    float phantom = .03f, false_free = .015f, thin_seen = .49f, thin_through = .06f;
    // Per-class overrides (< 0: use phantom / false_free) and phantom shape. blob = rays per phantom
    // along a row (same decision and depth, so dp_despeckle cannot remove it); nearer = phantoms only
    // in front of the true surface (0.3 m .. min(2.3 m, 0.85 d)).
    float phantom_textureless = -1, phantom_floor = -1, ff_textureless = -1, ff_floor = -1;
    int blob = 1;
    bool nearer = false;
    // Rays that never return: the leftmost dead_cols columns (StereoBM's dead band, u < numDisparities)
    // and dead_rows rows at the top and bottom (outside the rectilinear render of a wide lens).
    int dead_cols = 0, dead_rows = 0;
};

inline bool dp_dead(uint32_t k, const DepthPreset &dp) {
    const int r = int(k / 20), c = int(k % 20);
    return c < dp.dead_cols || r < dp.dead_rows || r >= 16 - dp.dead_rows;
}

// RLV2_DEPTH_PRESET=measured: the round-1 preset (bm160_temporal_r3 on the flat-shaded renders).
// RLV2_DEPTH_PRESET=realistic: sgbm320 on the world-anchored retextured nav renders (amp 0.3),
//   ray level, rlv2/depth_bench/calibrate_preset.py -> results/preset_calibration.json:
//   holes ~0 on every class (SGBM returns everywhere), k 0.023, phantoms 1.4% textured /
//   27% textureless / 0.8% floor, false-free 1.3% / 4.8% / 2.1%, phantom blobs ~2 rays,
//   textureless wall tiles 8% (real indoor photos: 2.5-6% by the BM gate, see texture_study.json).
// Env overrides: RLV2_TEXTURELESS, RLV2_HOLE_TEXTURED, RLV2_PRESET_PHANTOM, RLV2_PRESET_BLOB, and the
// per-class RLV2_PRESET_K / _PHANTOM_TEXTURELESS / _PHANTOM_FLOOR / _FF / _FF_TEXTURELESS / _FF_FLOOR,
// RLV2_HOLE_TEXTURELESS / RLV2_HOLE_FLOOR.
inline bool dp_preset_on() {
    const char *v = std::getenv("RLV2_DEPTH_PRESET");
    if (!v)
        return false;
    const std::string n(v);
    return n == "measured" || n == "realistic" || n == "bm160_photo" || n.rfind("hd_", 0) == 0;
}

inline DepthPreset dp_preset_from_env() {
    DepthPreset d;
    const char *name = std::getenv("RLV2_DEPTH_PRESET");
    if (name && std::string(name) == "realistic") {
        d.k = .023f;
        d.textureless = .08f;
        d.hole_textureless = .01f, d.hole_floor = 0, d.hole_textured = 0;
        d.phantom = .014f, d.phantom_textureless = .27f, d.phantom_floor = .008f;
        d.false_free = .013f, d.ff_textureless = .048f, d.ff_floor = .021f;
        d.blob = 2;
        d.nearer = true;
    }
    // RLV2_DEPTH_PRESET=bm160_photo: StereoBM 160x120 + LR check (bm160_lrc) at the rates measured
    // on real indoor photos (ETH3D + Middlebury, results/preset_calibration.json real_method_sweep):
    // holes 33% textured / 69% textureless, phantoms 0.7% / 2.6%, false-free 6.6% of returns, k 0.031.
    // The recommended training preset (round 4: t9_a .989 / 32 contacts, s4 .886 / 240 over 3 blocks).
    if (name && std::string(name) == "bm160_photo") {
        d = DepthPreset();
        d.k = .031f;
        d.textureless = .08f;
        d.hole_textured = .334f, d.hole_textureless = .686f, d.hole_floor = .334f;
        d.phantom = .007f, d.phantom_textureless = .026f, d.phantom_floor = .007f;
        d.false_free = .066f, d.ff_textureless = .066f, d.ff_floor = .066f;
        d.blob = 1;
        d.nearer = true;
    }
    // RLV2_DEPTH_PRESET=hd_bm160_90 / hd_bm320_90 / hd_bm320_120: StereoBM + LR check on HyperLux renders of
    // ProcTHOR-Train-1 through the "best" OV2640 camera-effects profile (synced, rectified to <= 0.2 px,
    // matched AE, mild JPEG, light noise, blur / rolling shutter from the drone's motion), ray level, at the
    // poses s4 flies (rlv2/hyperdrone_eval/calibrate_hd_preset.py -> results/hd_calibration/). Range is not
    // capped (the real pipeline reports far returns as hits); the StereoBM dead band blinds the left
    // columns. See docs/HYPERDRONE.md section 6.
    if (name && std::string(name).rfind("hd_", 0) == 0) {
        const std::string n(name);
        d = DepthPreset();
        d.range = 11.f;
        d.blob = 1;
        d.nearer = true;
        if (n == "hd_bm160_90") {
            d.k = .044f, d.textureless = .509f, d.dead_cols = 4;
            d.hole_floor = .753f, d.hole_textureless = .948f, d.hole_textured = .627f;
            d.phantom_floor = .0123f, d.phantom_textureless = .0115f, d.phantom = .0334f;
            d.ff_floor = .0855f, d.ff_textureless = .0803f, d.false_free = .0959f;
        } else if (n == "hd_bm320_90") {
            d.k = .030f, d.textureless = .619f, d.dead_cols = 4;
            d.hole_floor = .544f, d.hole_textureless = .932f, d.hole_textured = .511f;
            d.phantom_floor = .0067f, d.phantom_textureless = .0109f, d.phantom = .019f;
            d.ff_floor = .0291f, d.ff_textureless = .0588f, d.false_free = .0771f;
        } else if (n == "hd_bm320_120") {
            d.k = .0381f, d.textureless = .542f, d.dead_cols = 2;
            d.hole_floor = .457f, d.hole_textureless = .910f, d.hole_textured = .444f;
            d.phantom_floor = .0066f, d.phantom_textureless = .016f, d.phantom = .0339f;
            d.ff_floor = .0277f, d.ff_textureless = .099f, d.false_free = .1168f;
        } else if (n == "hd_bm160_65_ov2640") {
            // round 8: 65 deg (equiangular grid), 160x120 StereoBM, camera_effects.py ov2640_dvp (raw DVP, no
            // JPEG, 0-8 ms unsynced), fitted on ProcTHOR-1 closed-loop rays (results/r8/calibration). Use RLV2_HFOV=65.
            d.k = .0372f, d.textureless = .546f, d.dead_cols = 4;
            d.hole_floor = .749f, d.hole_textureless = .936f, d.hole_textured = .546f;
            d.phantom_floor = .0247f, d.phantom_textureless = .0096f, d.phantom = .026f;
            d.ff_floor = .0907f, d.ff_textureless = .0789f, d.false_free = .0799f;
        } else if (n == "hd_bm160_65_ov5647") {
            // round 8: as above with camera_effects.py ov5647 (MIPI raw, ~30 ms rolling readout, 0-16 ms unsynced).
            d.k = .0423f, d.textureless = .575f, d.dead_cols = 4;
            d.hole_floor = .83f, d.hole_textureless = .947f, d.hole_textured = .566f;
            d.phantom_floor = .0252f, d.phantom_textureless = .007f, d.phantom = .0337f;
            d.ff_floor = .1169f, d.ff_textureless = .0706f, d.false_free = .0794f;
        } else {
            std::abort();  // unknown hd_ preset
        }
    }
    if (const char *v = std::getenv("RLV2_TEXTURELESS"))
        d.textureless = std::stof(v);
    if (const char *v = std::getenv("RLV2_HOLE_TEXTURED"))
        d.hole_textured = std::stof(v);
    if (const char *v = std::getenv("RLV2_PRESET_PHANTOM")) {  // scales every class
        const float f = d.phantom > 0 ? std::stof(v) / d.phantom : 0;
        d.phantom = std::stof(v);
        if (d.phantom_textureless >= 0)
            d.phantom_textureless *= f;
        if (d.phantom_floor >= 0)
            d.phantom_floor *= f;
    }
    // Per-class overrides for sensitivity runs (e.g. rates measured on real photos).
    auto env = [](const char *name, float &x) {
        if (const char *v = std::getenv(name))
            x = std::stof(v);
    };
    env("RLV2_PRESET_K", d.k);
    env("RLV2_HOLE_TEXTURELESS", d.hole_textureless);
    env("RLV2_HOLE_FLOOR", d.hole_floor);
    env("RLV2_PRESET_PHANTOM_TEXTURELESS", d.phantom_textureless);
    env("RLV2_PRESET_PHANTOM_FLOOR", d.phantom_floor);
    env("RLV2_PRESET_FF", d.false_free);
    env("RLV2_PRESET_FF_TEXTURELESS", d.ff_textureless);
    env("RLV2_PRESET_FF_FLOOR", d.ff_floor);
    if (const char *v = std::getenv("RLV2_PRESET_BLOB"))
        d.blob = std::max(1, std::stoi(v));
    return d;
}

// Seed shared by the rays of one phantom blob (row-wise groups of dp.blob rays of the 20 x 16 grid).
inline uint32_t dp_blob_seed(uint32_t frame, uint32_t k, const DepthPreset &dp) {
    return vx_seed(frame, 0x51ab0000u + (k / 20) * 20 + (k % 20) / uint32_t(dp.blob));
}

enum DpSurface : uint32_t { DP_NONE, DP_FLOOR, DP_CEILING, DP_WALL_TEXTURELESS, DP_WALL_TEXTURED, DP_THIN, DP_MOVER };

inline bool dp_solid(const VxWorld &w, const uint32_t *bits, int x, int y, int z) { return vx_solid(w, bits, x, y, z); }

// Surface class at a hit point. `dir` is the unit ray, `d` its true range.
inline DpSurface dp_classify(const VxWorld &w, const uint32_t *bits, VxVec p, VxVec dir, float d, bool mover_hit,
                             uint32_t mission_salt, float textureless) {
    if (d >= VX_MAX_RANGE - 1e-3f)
        return DP_NONE;
    if (mover_hit)
        return DP_MOVER;
    const float s = w.scale;
    const VxVec h = vx_add(p, vx_scale(dir, d));
    // the face: the coordinate closest to a cell boundary
    const float c[3] = {h.x / s, h.y / s, h.z / s};
    int axis = 0;
    float best = 1e9f;
    for (int a = 0; a < 3; a++) {
        const float f = std::fabs(c[a] - std::round(c[a]));
        if (f < best)
            best = f, axis = a;
    }
    if (axis == 2)
        return dir.z < 0 ? DP_FLOOR : DP_CEILING;
    // the solid cell just behind the hit
    const VxVec in = vx_add(h, vx_scale(dir, .02f * s));
    const int cx = int(std::floor(in.x / s)), cy = int(std::floor(in.y / s)), cz = int(std::floor(in.z / s));
    // thin: one cell wide across the view (pole) or one cell tall (bar), only on fine maps
    if (s < .2f) {
        const int ox = axis == 0 ? 0 : 1, oy = axis == 0 ? 1 : 0;
        const bool pole = !dp_solid(w, bits, cx + ox, cy + oy, cz) && !dp_solid(w, bits, cx - ox, cy - oy, cz);
        const bool bar = !dp_solid(w, bits, cx, cy, cz + 1) && !dp_solid(w, bits, cx, cy, cz - 1);
        if (pole || bar)
            return DP_THIN;
    }
    // wall tile: plane (axis, face index) and a 2 m tile along it
    const int face = int(std::round(c[axis]));
    const float u = axis == 0 ? h.y : h.x;
    const uint32_t tile = uint32_t(int(std::floor(u / 2.0f)) + 4096) * 64u + uint32_t(int(std::floor(h.z / 2.0f)) + 32);
    uint32_t hs = vx_seed(mission_salt ^ (uint32_t(axis) * 0x51ed27u) ^ (uint32_t(face + 4096) * 0x9e37u), tile);
    return vx_uniform(hs) < textureless ? DP_WALL_TEXTURELESS : DP_WALL_TEXTURED;
}

// Stereo range for one ray. Returns VX_MAX_RANGE for nothing, VX_FAR_FREE for "free to range".
inline float dp_ray(float d, DpSurface surface, uint32_t seed, const DepthPreset &dp, uint32_t blob_seed = 0) {
    uint32_t s = seed;
    const bool tl = surface == DP_WALL_TEXTURELESS, fc = surface == DP_FLOOR || surface == DP_CEILING;
    const float phantom = tl && dp.phantom_textureless >= 0 ? dp.phantom_textureless
                          : fc && dp.phantom_floor >= 0  ? dp.phantom_floor
                                                         : dp.phantom;
    const float false_free = tl && dp.ff_textureless >= 0 ? dp.ff_textureless
                             : fc && dp.ff_floor >= 0    ? dp.ff_floor
                                                         : dp.false_free;
    // Phantom draw: from the blob's shared seed if any (so the whole blob agrees), else from the
    // ray's own stream (bit-identical to the round-1 preset).
    uint32_t bs = blob_seed;
    uint32_t &ps = blob_seed ? bs : s;
    if (vx_uniform(ps) < phantom) {
        const float pz = vx_uniform(ps);
        if (!dp.nearer)
            return .3f + 2 * pz;
        const float top = std::min(2.3f, .85f * d);
        if (top > .3f)
            return .3f + (top - .3f) * pz;
    }
    if (surface == DP_NONE)  // nothing within 12 m: treat as a far textured background
        return vx_uniform(s) < dp.hole_textured ? VX_MAX_RANGE : VX_FAR_FREE;
    float hole = dp.hole_textured;
    if (surface == DP_WALL_TEXTURELESS)
        hole = dp.hole_textureless;
    else if (surface == DP_FLOOR || surface == DP_CEILING)
        hole = dp.hole_floor;
    float z = d;
    if (surface == DP_THIN) {
        const float r = vx_uniform(s);
        if (r < dp.thin_through)
            return VX_FAR_FREE;
        if (r > dp.thin_seen + dp.thin_through)
            return VX_MAX_RANGE;
    } else {
        if (vx_uniform(s) < hole)
            return VX_MAX_RANGE;
        if (vx_uniform(s) < false_free)
            z = d * (1.3f + 1.2f * vx_uniform(s));
    }
    z += dp.k * z * z * vx_normal(s);
    if (z <= 0)
        return VX_MAX_RANGE;
    if (z > dp.range) {
        const bool textured = surface == DP_WALL_TEXTURED || surface == DP_MOVER || surface == DP_THIN;
        return textured ? VX_FAR_FREE : VX_MAX_RANGE;
    }
    return z;
}

// Drop isolated returns (RLV2_PRESET_DESPECKLE=1): a ray nearer than the far-free mark with no
// 8-neighbour within max(0.3 m, 15 %) of it becomes unknown. A cheap on-board filter that removes
// i.i.d. phantoms (and lone true hits). Real phantoms clump at edges; this is the optimistic case.
inline void dp_despeckle(float *rays /* 16x20 */) {
    float kept[320];
    for (int k = 0; k < 320; k++) {
        kept[k] = rays[k];
        const float d = rays[k];
        if (d >= VX_FAR_FREE)
            continue;
        int support = 0;
        for (int dr = -1; dr <= 1; dr++)
            for (int dc = -1; dc <= 1; dc++) {
                const int r = k / 20 + dr, c = k % 20 + dc;
                if ((dr || dc) && r >= 0 && r < 16 && c >= 0 && c < 20 &&
                    std::fabs(rays[r * 20 + c] - d) < std::max(.3f, .15f * d))
                    support++;
            }
        if (!support)
            kept[k] = VX_MAX_RANGE;
    }
    for (int k = 0; k < 320; k++)
        rays[k] = kept[k];
}
