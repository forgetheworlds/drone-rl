#pragma once

// Small fully connected networks for PPO and imitation on the CPU, with Adam and checkpoints.
// Matrix products go to Accelerate on macOS and to the system BLAS elsewhere.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
extern "C" void sgemm_(const char *, const char *, const int *, const int *, const int *, const float *,
                       const float *, const int *, const float *, const int *, const float *, float *, const int *);
#endif

// Row-major C = alpha * op(A) * op(B) + beta * C, with op(A) M x K and op(B) K x N.
inline void gemm(bool ta, bool tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                 float beta, float *C, int ldc) {
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda,
                B, ldb, beta, C, ldc);
#else
    const char a = ta ? 'T' : 'N', b = tb ? 'T' : 'N';
    sgemm_(&b, &a, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
#endif
}

inline void tanh_inplace(float *x, size_t n) {
#ifdef __APPLE__
    const int count = int(n);
    vvtanhf(x, x, &count);
#else
    for (size_t i = 0; i < n; i++)
        x[i] = std::tanh(x[i]);
#endif
}

enum class Activation : uint32_t { tanh = 0, relu = 1 };

// Layers sizes[0] -> ... -> sizes.back(). Hidden layers use `activation`, the output is linear.
// Weights per layer: W (out x in, row-major) then b (out).
struct Mlp {
    std::vector<int> sizes;
    Activation activation = Activation::tanh;
    std::vector<float> params, grads;
    std::vector<size_t> offsets;
    std::vector<std::vector<float>> acts; // acts[0] is the input, acts[l] the output of layer l

    Mlp() = default;
    Mlp(std::vector<int> s, Activation a, uint32_t seed, float output_gain) : sizes(std::move(s)), activation(a) {
        size_t n = 0;
        for (size_t l = 0; l + 1 < sizes.size(); l++) {
            offsets.push_back(n);
            n += size_t(sizes[l]) * sizes[l + 1] + sizes[l + 1];
        }
        params.assign(n, 0);
        grads.assign(n, 0);
        std::mt19937 rng(seed);
        for (size_t l = 0; l + 1 < sizes.size(); l++) {
            const bool last = l + 2 == sizes.size();
            const float gain = last ? output_gain : (a == Activation::relu ? std::sqrt(2.0f) : 1.0f);
            std::normal_distribution<float> d(0, gain / std::sqrt(float(sizes[l])));
            float *w = params.data() + offsets[l];
            for (int i = 0; i < sizes[l] * sizes[l + 1]; i++)
                w[i] = d(rng);
        }
    }
    int inputs() const { return sizes.front(); }
    int outputs() const { return sizes.back(); }
    const float *weight(size_t l) const { return params.data() + offsets[l]; }
    const float *bias(size_t l) const { return weight(l) + size_t(sizes[l]) * sizes[l + 1]; }

    // Keeps the activations for backward().
    const float *forward(const float *x, int batch) {
        acts.resize(sizes.size());
        acts[0].assign(x, x + size_t(batch) * sizes[0]);
        for (size_t l = 0; l + 1 < sizes.size(); l++) {
            const int in = sizes[l], out = sizes[l + 1];
            auto &y = acts[l + 1];
            y.resize(size_t(batch) * out);
            for (int b = 0; b < batch; b++)
                std::memcpy(&y[size_t(b) * out], bias(l), out * 4);
            gemm(false, true, batch, out, in, 1, acts[l].data(), in, weight(l), in, 1, y.data(), out);
            if (l + 2 < sizes.size()) {
                if (activation == Activation::tanh)
                    tanh_inplace(y.data(), y.size());
                else
                    for (auto &v : y)
                        v = v > 0 ? v : 0;
            }
        }
        return acts.back().data();
    }

    // Adds the gradient for d(loss)/d(output) into grads.
    void backward(const float *dy_in, int batch) {
        std::vector<float> dy(dy_in, dy_in + size_t(batch) * outputs()), dx;
        for (size_t l = sizes.size() - 1; l-- > 0;) {
            const int in = sizes[l], out = sizes[l + 1];
            float *gw = grads.data() + offsets[l], *gb = gw + size_t(in) * out;
            gemm(true, false, out, in, batch, 1, dy.data(), out, acts[l].data(), in, 1, gw, in);
            for (int b = 0; b < batch; b++)
                for (int j = 0; j < out; j++)
                    gb[j] += dy[size_t(b) * out + j];
            if (l == 0)
                break;
            dx.assign(size_t(batch) * in, 0);
            gemm(false, false, batch, in, out, 1, dy.data(), out, weight(l), in, 0, dx.data(), in);
            const auto &a = acts[l];
            if (activation == Activation::tanh)
                for (size_t i = 0; i < dx.size(); i++)
                    dx[i] *= 1 - a[i] * a[i];
            else
                for (size_t i = 0; i < dx.size(); i++)
                    dx[i] *= a[i] > 0 ? 1 : 0;
            dy.swap(dx);
        }
    }
};

struct Adam {
    std::vector<float> m, v;
    uint64_t t = 0;
    void step(std::vector<float> &params, std::vector<float> &grads, float lr, float b1 = .9f, float b2 = .999f,
              float eps = 1e-8f) {
        if (m.size() != params.size()) {
            m.assign(params.size(), 0);
            v.assign(params.size(), 0);
        }
        ++t;
        const float c1 = 1 - std::pow(b1, float(t)), c2 = 1 - std::pow(b2, float(t));
        for (size_t i = 0; i < params.size(); i++) {
            m[i] = b1 * m[i] + (1 - b1) * grads[i];
            v[i] = b2 * v[i] + (1 - b2) * grads[i] * grads[i];
            params[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
        }
        std::fill(grads.begin(), grads.end(), 0.0f);
    }
};

inline float clip_grad_norm(std::vector<float> &grads, float limit) {
    double s = 0;
    for (float g : grads)
        s += double(g) * g;
    const float norm = float(std::sqrt(s));
    if (norm > limit)
        for (auto &g : grads)
            g *= limit / norm;
    return norm;
}

// Running mean and variance of observations, frozen when `update` is false.
struct Normalizer {
    std::vector<double> mean, var;
    double count = 1e-4;
    explicit Normalizer(int n = 0) : mean(n, 0), var(n, 1) {}
    void update(const float *x, int batch) {
        const int n = int(mean.size());
        std::vector<double> bm(n, 0), bv(n, 0);
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++)
                bm[i] += x[size_t(b) * n + i];
        for (int i = 0; i < n; i++)
            bm[i] /= batch;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++) {
                const double d = x[size_t(b) * n + i] - bm[i];
                bv[i] += d * d;
            }
        const double total = count + batch;
        for (int i = 0; i < n; i++) {
            bv[i] /= batch;
            const double delta = bm[i] - mean[i];
            mean[i] += delta * batch / total;
            var[i] = (var[i] * count + bv[i] * batch + delta * delta * count * batch / total) / total;
        }
        count = total;
    }
    void apply(float *x, int batch) const {
        const int n = int(mean.size());
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < n; i++) {
                float &v = x[size_t(b) * n + i];
                v = float((v - mean[i]) / std::sqrt(var[i] + 1e-8));
                v = std::fmin(5.0f, std::fmax(-5.0f, v));
            }
    }
};

// Checkpoint: magic, then for each network its sizes, activation and parameters, then the
// normalizer and the policy log std.
struct Checkpoint {
    std::vector<Mlp *> nets;
    Normalizer *norm = nullptr;
    std::vector<float> *log_std = nullptr;

    void save(const std::string &path) const {
        std::ofstream f(path + ".tmp", std::ios::binary);
        const uint32_t magic = 0x32564c52, count = uint32_t(nets.size());
        f.write((const char *)&magic, 4);
        f.write((const char *)&count, 4);
        for (const Mlp *n : nets) {
            const uint32_t layers = uint32_t(n->sizes.size()), act = uint32_t(n->activation);
            f.write((const char *)&layers, 4);
            f.write((const char *)n->sizes.data(), layers * 4);
            f.write((const char *)&act, 4);
            f.write((const char *)n->params.data(), n->params.size() * 4);
        }
        const uint32_t dims = uint32_t(norm->mean.size());
        f.write((const char *)&dims, 4);
        f.write((const char *)norm->mean.data(), dims * 8);
        f.write((const char *)norm->var.data(), dims * 8);
        f.write((const char *)&norm->count, 8);
        const uint32_t ls = uint32_t(log_std->size());
        f.write((const char *)&ls, 4);
        f.write((const char *)log_std->data(), ls * 4);
        f.close();
        std::rename((path + ".tmp").c_str(), path.c_str());
    }
    void load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot read " + path);
        uint32_t magic, count;
        f.read((char *)&magic, 4);
        f.read((char *)&count, 4);
        if (magic != 0x32564c52 || count != nets.size())
            throw std::runtime_error("Bad checkpoint " + path);
        for (Mlp *n : nets) {
            uint32_t layers, act;
            f.read((char *)&layers, 4);
            std::vector<int> sizes(layers);
            f.read((char *)sizes.data(), layers * 4);
            f.read((char *)&act, 4);
            *n = Mlp(sizes, Activation(act), 0, 1);
            f.read((char *)n->params.data(), n->params.size() * 4);
        }
        uint32_t dims;
        f.read((char *)&dims, 4);
        *norm = Normalizer(int(dims));
        f.read((char *)norm->mean.data(), dims * 8);
        f.read((char *)norm->var.data(), dims * 8);
        f.read((char *)&norm->count, 8);
        uint32_t ls;
        f.read((char *)&ls, 4);
        log_std->resize(ls);
        f.read((char *)log_std->data(), ls * 4);
        if (!f)
            throw std::runtime_error("Truncated checkpoint " + path);
    }
};
