// The offsets match RaptorWeights in raptor.hpp and assets/raptor.bin.
// Per-environment helper. Call once per native 10 ms step. The caller owns
// persistent hidden[16] state and resets it on episode/policy reset.
inline void raptor_reset(device const float* weights, thread float* hidden) {
    for (uint i = 0; i < 16; ++i) hidden[i] = weights[2000 + i];
}

inline float raptor_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Match the clamp in the official L2F executor after raw policy evaluation.
inline void raptor_clip_action(thread float* action) {
    for (uint i = 0; i < 4; ++i) action[i] = clamp(action[i], -1.0f, 1.0f);
}

inline void raptor_forward(device const float* weights,
                           thread const float* observation,
                           thread float* hidden,
                           thread float* action) {
    thread float x[16];
    for (uint row = 0; row < 16; ++row) {
        float value = weights[352 + row];
        for (uint col = 0; col < 22; ++col)
            value += weights[row * 22 + col] * observation[col];
        x[row] = max(value, 0.0f);
    }

    thread float input_gates[48];
    thread float hidden_gates[48];
    for (uint row = 0; row < 48; ++row) {
        float input_value = weights[1904 + row];
        float hidden_value = weights[1952 + row];
        for (uint col = 0; col < 16; ++col) {
            input_value += weights[368 + row * 16 + col] * x[col];
            hidden_value += weights[1136 + row * 16 + col] * hidden[col];
        }
        input_gates[row] = input_value;
        hidden_gates[row] = hidden_value;
    }

    thread float next_hidden[16];
    for (uint i = 0; i < 16; ++i) {
        float reset = raptor_sigmoid(input_gates[i] + hidden_gates[i]);
        float update = raptor_sigmoid(input_gates[16 + i] + hidden_gates[16 + i]);
        float candidate = tanh(input_gates[32 + i] + reset * hidden_gates[32 + i]);
        next_hidden[i] = (1.0f - update) * candidate + update * hidden[i];
    }
    for (uint i = 0; i < 16; ++i) hidden[i] = next_hidden[i];

    for (uint row = 0; row < 4; ++row) {
        float value = weights[2080 + row];
        for (uint col = 0; col < 16; ++col)
            value += weights[2016 + row * 16 + col] * hidden[col];
        action[row] = value;
    }
}
