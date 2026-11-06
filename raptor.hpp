#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

// Frozen RAPTOR checkpoint: input Dense(22,16)-ReLU, GRU(16), output Dense(16,4).
// All arrays are row-major. This order is the on-disk and Metal ABI.
struct RaptorWeights {
    float input_weights[16 * 22];
    float input_bias[16];
    float gru_input_weights[48 * 16];
    float gru_hidden_weights[48 * 16];
    float gru_input_bias[48];
    float gru_hidden_bias[48];
    float initial_hidden_state[16];
    float output_weights[4 * 16];
    float output_bias[4];
};

static_assert(sizeof(RaptorWeights) == 2084 * sizeof(float), "RAPTOR weight ABI changed");

inline bool raptor_load_weights(const char* path, RaptorWeights* weights) {
    if (!path || !weights) return false;
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    unsigned char header[16];
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    static constexpr unsigned char magic[8] = {'R','A','P','T','O','R','1',0};
    if (!file || std::memcmp(header, magic, sizeof(magic)) != 0) return false;
    const uint32_t version = uint32_t(header[8]) | (uint32_t(header[9]) << 8) |
                             (uint32_t(header[10]) << 16) | (uint32_t(header[11]) << 24);
    const uint32_t count = uint32_t(header[12]) | (uint32_t(header[13]) << 8) |
                           (uint32_t(header[14]) << 16) | (uint32_t(header[15]) << 24);
    if (version != 1 || count != 2084) return false;
    file.read(reinterpret_cast<char*>(weights), sizeof(*weights));
    return bool(file);
}

inline void raptor_reset(const RaptorWeights& weights, float hidden[16]) {
    std::copy(weights.initial_hidden_state, weights.initial_hidden_state + 16, hidden);
}

inline float raptor_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// One native policy update at 100 Hz. State persists across calls until reset.
// This returns the raw final Dense output, as RLtools evaluate_step does.
inline void raptor_forward(const RaptorWeights& w, const float observation[22],
                           float hidden[16], float action[4]) {
    float x[16];
    for (int row = 0; row < 16; ++row) {
        float value = w.input_bias[row];
        const float* weights = w.input_weights + row * 22;
        for (int col = 0; col < 22; ++col) value += weights[col] * observation[col];
        x[row] = std::max(value, 0.0f);
    }

    float input_gates[48];
    float hidden_gates[48];
    for (int row = 0; row < 48; ++row) {
        float input_value = w.gru_input_bias[row];
        float hidden_value = w.gru_hidden_bias[row];
        const float* input_weights = w.gru_input_weights + row * 16;
        const float* hidden_weights = w.gru_hidden_weights + row * 16;
        for (int col = 0; col < 16; ++col) {
            input_value += input_weights[col] * x[col];
            hidden_value += hidden_weights[col] * hidden[col];
        }
        input_gates[row] = input_value;
        hidden_gates[row] = hidden_value;
    }

    float next_hidden[16];
    for (int i = 0; i < 16; ++i) {
        const float reset = raptor_sigmoid(input_gates[i] + hidden_gates[i]);
        const float update = raptor_sigmoid(input_gates[16 + i] + hidden_gates[16 + i]);
        const float candidate = std::tanh(input_gates[32 + i] + reset * hidden_gates[32 + i]);
        next_hidden[i] = (1.0f - update) * candidate + update * hidden[i];
    }
    std::copy(next_hidden, next_hidden + 16, hidden);

    for (int row = 0; row < 4; ++row) {
        float value = w.output_bias[row];
        const float* weights = w.output_weights + row * 16;
        for (int col = 0; col < 16; ++col) value += weights[col] * hidden[col];
        action[row] = value;
    }
}

// The official L2F executor applies this clamp after the network forward pass.
inline void raptor_clip_action(float action[4]) {
    for (int i=0; i<4; ++i) action[i] = std::max(-1.0f, std::min(1.0f, action[i]));
}

inline float raptor_clip_error(float x, float limit) {
    return std::max(-limit, std::min(limit, x));
}

inline void raptor_quaternion_multiply(const float a[4], const float b[4], float out[4]) {
    const float w = a[0], x = a[1], y = a[2], z = a[3];
    const float bw = b[0], bx = b[1], by = b[2], bz = b[3];
    out[0] = w*bw - x*bx - y*by - z*bz;
    out[1] = w*bx + x*bw + y*bz - z*by;
    out[2] = w*by - x*bz + y*bw + z*bx;
    out[3] = w*bz + x*by - y*bx + z*bw;
}

inline void raptor_quaternion_matrix(const float q[4], float r[9]) {
    const float w=q[0], x=q[1], y=q[2], z=q[3];
    r[0]=1-2*y*y-2*z*z; r[1]=2*x*y-2*w*z;   r[2]=2*x*z+2*w*y;
    r[3]=2*x*y+2*w*z;   r[4]=1-2*x*x-2*z*z; r[5]=2*y*z-2*w*x;
    r[6]=2*x*z-2*w*y;   r[7]=2*y*z+2*w*x;   r[8]=1-2*x*x-2*y*y;
}

// Match the PX4 adapter: body state errors in the target frame, then the
// relative attitude matrix, body angular rate, and the previous motor action.
inline void raptor_pack_observation(const float position[3], const float orientation[4],
                                   const float world_velocity[3], const float body_rates[3],
                                   const float target_position[3], const float target_orientation[4],
                                   const float target_velocity[3], const float previous_action[4],
                                   float observation[22]) {
    const float target_inverse[4] = {target_orientation[0], -target_orientation[1],
                                     -target_orientation[2], -target_orientation[3]};
    float target_inverse_rotation[9];
    raptor_quaternion_matrix(target_inverse, target_inverse_rotation);
    float position_error[3], velocity_error[3];
    for (int i=0; i<3; ++i) {
        position_error[i] = position[i] - target_position[i];
        velocity_error[i] = world_velocity[i] - target_velocity[i];
    }
    for (int row=0; row<3; ++row) {
        float p=0, v=0;
        for (int col=0; col<3; ++col) {
            p += target_inverse_rotation[row*3+col] * position_error[col];
            v += target_inverse_rotation[row*3+col] * velocity_error[col];
        }
        observation[row] = raptor_clip_error(p, 0.5f);
        observation[12+row] = raptor_clip_error(v, 1.0f);
    }
    float relative_orientation[4];
    raptor_quaternion_multiply(target_inverse, orientation, relative_orientation);
    raptor_quaternion_matrix(relative_orientation, observation+3);
    for (int i=0; i<3; ++i) observation[15+i] = body_rates[i];
    for (int i=0; i<4; ++i) observation[18+i] = previous_action[i];
}
