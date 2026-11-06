#pragma once

// Compiles the Metal kernels as plain C++ so the training loop can run on any machine.
// The Metal address space keywords are defined away while the kernels are read, then
// restored.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace metal {}
#define kernel
#define device
#define constant
#define thread
using std::atan2;
using std::cos;
using std::exp;
using std::fabs;
using std::floor;
using std::sin;
using std::sqrt;
using std::tanh;
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }
inline uint32_t min(uint32_t a, uint32_t b) { return a < b ? a : b; }
inline float clamp(float x, float a, float b) { return std::fmin(std::fmax(x, a), b); }
struct float4 {
    float x, y, z, w;
    float4() = default;
    float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

#include "../voxel_world.h"
#include "../sensors.h"
#include "../../physics.metal"
#include "../../raptor.metal"
#include "../nav_task.h"
#include "../nav_env.metal"
#include "../train_env.metal"

#undef kernel
#undef device
#undef constant
#undef thread
