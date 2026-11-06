#pragma once

// PPO for the privileged teacher and DAgger for the deployable student. Works with any
// environment backend that provides:
//
//   uint32_t count;
//   void reset(uint32_t seed);
//   void observe(float *teacher, float *student);       // once per tick, after reset/step
//   void step(const float *actions, float *rewards, uint32_t *dones);
//   void set_level(float sensor_level);
//   EnvTotals totals();                                  // counters summed over all envs
//
// Include nav_task.h first.

#include "mlp.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <numeric>

struct EnvTotals {
    uint64_t episodes = 0, contacts = 0, arrivals = 0, completions = 0;
};

struct PpoConfig {
    int iterations = 1500, horizon = 32, epochs = 4, minibatches = 4, hidden = 256;
    float gamma = .99f, lambda = .95f, lr = 3e-4f, clip = .2f, entropy = .003f, value = .5f, grad_norm = 1;
    float level_start = .25f, level_end = 1;
    int level_ramp = 300, save_every = 25;
    uint32_t seed = 1;
    bool prior = true;
    std::string checkpoint = "teacher.bin", log = "teacher.csv";
};

inline double trainer_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Teacher {
    Mlp actor, critic;
    Normalizer norm{NT_TEACHER_OBS};
    std::vector<float> log_std = std::vector<float>(NT_ACTIONS, -2.3f);
    Checkpoint checkpoint() { return Checkpoint{{&actor, &critic}, &norm, &log_std}; }
};

template <typename Env> void train_teacher(Env &env, const PpoConfig &c, const std::string &resume = "") {
    const int N = int(env.count), T = c.horizon, O = NT_TEACHER_OBS, A = NT_ACTIONS;
    Teacher net;
    net.actor = Mlp({O, c.hidden, c.hidden, A}, Activation::tanh, c.seed, .01f);
    net.critic = Mlp({O, c.hidden, c.hidden, 1}, Activation::tanh, c.seed + 1, 1);
    if (!resume.empty())
        net.checkpoint().load(resume);
    Adam actor_opt, critic_opt, std_opt;
    std::vector<float> std_grad(A, 0);
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> normal(0, 1);

    std::vector<float> obs(size_t(T + 1) * N * O), student(size_t(N) * NT_STUDENT_OBS), act(size_t(T) * N * A),
        logp(size_t(T) * N), val(size_t(T + 1) * N), rew(size_t(T) * N), adv(size_t(T) * N), ret(size_t(T) * N);
    std::vector<uint32_t> done(size_t(T) * N);
    std::vector<float> raw(size_t(N) * O), executed(size_t(N) * A);

    env.reset(c.seed);
    env.observe(raw.data(), student.data());
    FILE *log = std::fopen(c.log.c_str(), resume.empty() ? "w" : "a");
    if (resume.empty())
        std::fprintf(log, "iteration,steps,seconds,level,reward,episodes,completion,contact,timeout,arrivals,"
                          "value_loss,clip_fraction,std\n");
    EnvTotals last = env.totals();
    uint64_t steps = 0;
    const double start = trainer_now();
    for (int it = 0; it < c.iterations; it++) {
        const float level =
            c.level_start + (c.level_end - c.level_start) * std::min(1.0f, float(it) / std::max(1, c.level_ramp));
        env.set_level(level);
        double reward_sum = 0;
        const double rollout_start = trainer_now();
        uint64_t timeouts = 0;
        for (int t = 0; t <= T; t++) {
            float *o = &obs[size_t(t) * N * O];
            net.norm.update(raw.data(), N);
            std::copy(raw.begin(), raw.end(), o);
            net.norm.apply(o, N);
            const float *v = net.critic.forward(o, N);
            std::copy(v, v + N, &val[size_t(t) * N]);
            if (t == T)
                break;
            const float *mu = net.actor.forward(o, N);
            float *a = &act[size_t(t) * N * A];
            for (int i = 0; i < N; i++) {
                float lp = 0;
                for (int j = 0; j < A; j++) {
                    const float s = std::exp(net.log_std[j]), e = normal(rng);
                    a[i * A + j] = mu[i * A + j] + s * e;
                    lp += -.5f * e * e - net.log_std[j];
                }
                logp[size_t(t) * N + i] = lp;
            }
            if (c.prior) {
                for (int i = 0; i < N; i++) {
                    float p[NT_ACTIONS];
                    nt_teacher_prior(&raw[size_t(i) * O], p);
                    for (int j = 0; j < A; j++)
                        executed[size_t(i) * A + j] = p[j] + a[i * A + j];
                    nt_teacher_certify(&raw[size_t(i) * O], &executed[size_t(i) * A]);
                }
                env.step(executed.data(), &rew[size_t(t) * N], &done[size_t(t) * N]);
            } else {
                env.step(a, &rew[size_t(t) * N], &done[size_t(t) * N]);
            }
            for (int i = 0; i < N; i++) {
                reward_sum += rew[size_t(t) * N + i];
                if (done[size_t(t) * N + i] == 2) {
                    rew[size_t(t) * N + i] += c.gamma * val[size_t(t) * N + i];
                    ++timeouts;
                }
            }
            env.observe(raw.data(), student.data());
        }
        steps += uint64_t(N) * T;
        const double rollout_end = trainer_now();
        for (int i = 0; i < N; i++) {
            float g = 0;
            for (int t = T - 1; t >= 0; t--) {
                const size_t k = size_t(t) * N + i;
                const float mask = done[k] ? 0 : 1;
                const float delta = rew[k] + c.gamma * mask * val[k + N] - val[k];
                g = delta + c.gamma * c.lambda * mask * g;
                adv[k] = g;
                ret[k] = g + val[k];
            }
        }

        const int total = N * T, batch = total / c.minibatches;
        std::vector<int> order(total);
        std::iota(order.begin(), order.end(), 0);
        std::vector<float> mo(size_t(batch) * O), ma(size_t(batch) * A), mlp_old(batch), madv(batch), mret(batch),
            dmu(size_t(batch) * A), dv(batch);
        double value_loss = 0, clipped = 0;
        int updates = 0;
        for (int epoch = 0; epoch < c.epochs; epoch++) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int mb = 0; mb < c.minibatches; mb++) {
                double am = 0, as = 0;
                for (int b = 0; b < batch; b++) {
                    const int k = order[size_t(mb) * batch + b];
                    std::copy(&obs[size_t(k) * O], &obs[size_t(k) * O] + O, &mo[size_t(b) * O]);
                    std::copy(&act[size_t(k) * A], &act[size_t(k) * A] + A, &ma[size_t(b) * A]);
                    mlp_old[b] = logp[k];
                    madv[b] = adv[k];
                    mret[b] = ret[k];
                    am += adv[k];
                }
                am /= batch;
                for (int b = 0; b < batch; b++)
                    as += (madv[b] - am) * (madv[b] - am);
                as = std::sqrt(as / batch) + 1e-8;
                const float *mu = net.actor.forward(mo.data(), batch);
                std::fill(std_grad.begin(), std_grad.end(), 0.0f);
                for (int b = 0; b < batch; b++) {
                    const float A_ = float((madv[b] - am) / as);
                    float lp = 0;
                    for (int j = 0; j < A; j++) {
                        const float z = (ma[b * A + j] - mu[b * A + j]) / std::exp(net.log_std[j]);
                        lp += -.5f * z * z - net.log_std[j];
                    }
                    const float ratio = std::exp(lp - mlp_old[b]);
                    const bool active = !((A_ > 0 && ratio > 1 + c.clip) || (A_ < 0 && ratio < 1 - c.clip));
                    clipped += active ? 0 : 1;
                    const float dlp = active ? -A_ * ratio / batch : 0;
                    for (int j = 0; j < A; j++) {
                        const float s = std::exp(net.log_std[j]), z = (ma[b * A + j] - mu[b * A + j]) / s;
                        dmu[b * A + j] = dlp * z / s;
                        std_grad[j] += dlp * (z * z - 1);
                    }
                }
                for (int j = 0; j < A; j++)
                    std_grad[j] -= c.entropy;
                net.actor.backward(dmu.data(), batch);
                const float *v = net.critic.forward(mo.data(), batch);
                for (int b = 0; b < batch; b++) {
                    const float e = v[b] - mret[b];
                    value_loss += .5 * e * e / batch;
                    dv[b] = c.value * e / batch;
                }
                net.critic.backward(dv.data(), batch);
                clip_grad_norm(net.actor.grads, c.grad_norm);
                clip_grad_norm(net.critic.grads, c.grad_norm);
                actor_opt.step(net.actor.params, net.actor.grads, c.lr);
                critic_opt.step(net.critic.params, net.critic.grads, c.lr);
                std_opt.step(net.log_std, std_grad, c.lr);
                for (auto &s : net.log_std)
                    s = std::fmin(0.0f, std::fmax(-3.5f, s));
                ++updates;
            }
        }

        const double update_end = trainer_now();
        const EnvTotals now = env.totals();
        const double episodes = double(now.episodes - last.episodes);
        const double e = std::max(1.0, episodes);
        std::fprintf(log, "%d,%llu,%.1f,%.3f,%.5f,%.0f,%.4f,%.4f,%.4f,%llu,%.4f,%.4f,%.4f\n", it,
                     (unsigned long long)steps, trainer_now() - start, level, reward_sum / (double(N) * T), episodes,
                     (now.completions - last.completions) / e, (now.contacts - last.contacts) / e, timeouts / e,
                     (unsigned long long)(now.arrivals - last.arrivals), value_loss / updates, clipped / (double(batch) * updates),
                     std::exp(net.log_std[0]));
        std::fflush(log);
        if (it % 10 == 0)
            std::printf("it %d steps %llu %.0fs level %.2f reward %.4f episodes %.0f completion %.3f contact %.3f "
                        "timeout %.3f vloss %.3f rollout %.2fs update %.2fs\n",
                        it, (unsigned long long)steps, trainer_now() - start, level, reward_sum / (double(N) * T),
                        episodes, (now.completions - last.completions) / e, (now.contacts - last.contacts) / e,
                        timeouts / e, value_loss / updates, rollout_end - rollout_start, update_end - rollout_end);
        std::fflush(stdout);
        last = now;
        if ((it + 1) % c.save_every == 0 || it + 1 == c.iterations)
            net.checkpoint().save(c.checkpoint);
    }
    std::fclose(log);
}

struct DaggerConfig {
    int iterations = 300, horizon = 32, hidden = 160, buffer = 1 << 19, batch = 4096, updates = 64, beta_iterations = 40;
    float lr = 1e-3f, level = 1;
    uint32_t seed = 2;
    std::string checkpoint = "student.bin", log = "student.csv";
};

struct Student {
    Mlp actor;
    Normalizer norm{NT_STUDENT_OBS};
    std::vector<float> log_std;
    Checkpoint checkpoint() { return Checkpoint{{&actor}, &norm, &log_std}; }
};

// The teacher's mean action including the prior, clipped to the command range.
inline void teacher_actions(Teacher &t, const float *raw, int n, std::vector<float> &scratch, float *out) {
    const int O = NT_TEACHER_OBS, A = NT_ACTIONS;
    scratch.assign(raw, raw + size_t(n) * O);
    t.norm.apply(scratch.data(), n);
    const float *mu = t.actor.forward(scratch.data(), n);
    for (int i = 0; i < n; i++) {
        float p[NT_ACTIONS];
        nt_teacher_prior(&raw[size_t(i) * O], p);
        for (int j = 0; j < A; j++)
            out[size_t(i) * A + j] = std::fmax(-1.0f, std::fmin(1.0f, p[j] + mu[size_t(i) * A + j]));
        nt_teacher_certify(&raw[size_t(i) * O], &out[size_t(i) * A]);
    }
}

// DAgger: the student flies (mixed with the teacher early on), the teacher labels every visited
// state, and the student regresses onto the labels from an aggregated ring buffer.
template <typename Env>
void train_student(Env &env, const DaggerConfig &c, const std::string &teacher_path, const std::string &resume = "") {
    const int N = int(env.count), O = NT_TEACHER_OBS, S = NT_STUDENT_OBS, A = NT_ACTIONS;
    Teacher teacher;
    teacher.checkpoint().load(teacher_path);
    Student net;
    net.actor = Mlp({S, c.hidden, c.hidden, A}, Activation::relu, c.seed, .01f);
    if (!resume.empty())
        net.checkpoint().load(resume);
    Adam opt;
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uniform(0, 1);

    std::vector<float> raw(size_t(N) * O), student(size_t(N) * S), scratch, label(size_t(N) * A), sobs(size_t(N) * S),
        executed(size_t(N) * A), rew(N);
    std::vector<uint32_t> done(N);
    std::vector<float> bo(size_t(c.buffer) * S), ba(size_t(c.buffer) * A), mo(size_t(c.batch) * S), dy(size_t(c.batch) * A);
    std::vector<uint8_t> use_teacher(N, 1);
    size_t filled = 0, head = 0;

    env.set_level(c.level);
    env.reset(c.seed);
    env.observe(raw.data(), student.data());
    FILE *log = std::fopen(c.log.c_str(), resume.empty() ? "w" : "a");
    if (resume.empty())
        std::fprintf(log, "iteration,steps,seconds,beta,episodes,completion,contact,timeout,loss,disagreement\n");
    EnvTotals last = env.totals();
    uint64_t steps = 0;
    const double start = trainer_now();
    for (int it = 0; it < c.iterations; it++) {
        const float beta = std::max(0.0f, 1 - float(it) / std::max(1, c.beta_iterations));
        uint64_t timeouts = 0;
        double disagreement = 0;
        for (int t = 0; t < c.horizon; t++) {
            teacher_actions(teacher, raw.data(), N, scratch, label.data());
            net.norm.update(student.data(), N);
            sobs = student;
            net.norm.apply(sobs.data(), N);
            const float *mu = net.actor.forward(sobs.data(), N);
            for (int i = 0; i < N; i++) {
                // The network learns the correction on top of the student's own prior.
                float prior[NT_ACTIONS];
                nt_student_prior(&student[size_t(i) * S], prior);
                std::copy(&student[size_t(i) * S], &student[size_t(i) * S] + S, &bo[head * S]);
                for (int j = 0; j < A; j++)
                    ba[head * A + j] = nt_student_residual(label[size_t(i) * A + j] - prior[j]);
                head = (head + 1) % size_t(c.buffer);
                filled = std::min(filled + 1, size_t(c.buffer));
                float combined[NT_ACTIONS];
                nt_student_combine(prior, &mu[size_t(i) * A], combined);
                for (int j = 0; j < A; j++) {
                    const float own = std::fmax(-1.0f, std::fmin(1.0f, combined[j]));
                    const float d = own - label[size_t(i) * A + j];
                    disagreement += d * d;
                    executed[size_t(i) * A + j] = use_teacher[i] ? label[size_t(i) * A + j] : own;
                }
                if (!use_teacher[i])
                    nt_student_certify(&student[size_t(i) * S], &executed[size_t(i) * A]);
            }
            env.step(executed.data(), rew.data(), done.data());
            for (int i = 0; i < N; i++)
                if (done[i]) {
                    timeouts += done[i] == 2;
                    use_teacher[i] = uniform(rng) < beta;
                }
            env.observe(raw.data(), student.data());
        }
        steps += uint64_t(N) * c.horizon;

        double loss = 0;
        std::uniform_int_distribution<size_t> pick(0, filled - 1);
        for (int u = 0; u < c.updates; u++) {
            std::vector<size_t> idx(c.batch);
            for (int b = 0; b < c.batch; b++) {
                idx[b] = pick(rng);
                std::copy(&bo[idx[b] * S], &bo[idx[b] * S] + S, &mo[size_t(b) * S]);
            }
            net.norm.apply(mo.data(), c.batch);
            const float *y = net.actor.forward(mo.data(), c.batch);
            for (int b = 0; b < c.batch; b++)
                for (int j = 0; j < A; j++) {
                    const float e = y[size_t(b) * A + j] - ba[idx[b] * A + j];
                    loss += double(e) * e / (double(c.batch) * c.updates);
                    dy[size_t(b) * A + j] = 2 * e / c.batch;
                }
            net.actor.backward(dy.data(), c.batch);
            clip_grad_norm(net.actor.grads, 1);
            opt.step(net.actor.params, net.actor.grads, c.lr);
        }

        const EnvTotals now = env.totals();
        const double episodes = double(now.episodes - last.episodes), e = std::max(1.0, episodes);
        std::fprintf(log, "%d,%llu,%.1f,%.3f,%.0f,%.4f,%.4f,%.4f,%.5f,%.5f\n", it, (unsigned long long)steps,
                     trainer_now() - start, beta, episodes, (now.completions - last.completions) / e,
                     (now.contacts - last.contacts) / e, timeouts / e, loss,
                     disagreement / (double(N) * c.horizon * A));
        std::fflush(log);
        last = now;
        if ((it + 1) % 10 == 0 || it + 1 == c.iterations)
            net.checkpoint().save(c.checkpoint);
    }
    std::fclose(log);
}

// PPO on the student itself, after DAgger: the actor sees only the student observation and
// keeps its bounded residual on the route prior; the critic sees the privileged teacher
// observation. This lets the student learn what is safe with its own view, which copying the
// teacher cannot teach.
template <typename Env> void train_student_ppo(Env &env, const PpoConfig &c, const std::string &student_path) {
    const int N = int(env.count), T = c.horizon, O = NT_TEACHER_OBS, S = NT_STUDENT_OBS, A = NT_ACTIONS;
    // "zero" starts from a network that outputs nothing, so training starts at the prior.
    Student net;
    const bool fresh = student_path == "zero";
    net.actor = Mlp({S, 160, 160, A}, Activation::relu, c.seed, fresh ? 0.0f : .01f);
    if (!fresh)
        net.checkpoint().load(student_path);
    net.log_std.assign(A, std::log(.03f));
    Mlp critic({O, c.hidden, c.hidden, 1}, Activation::tanh, c.seed + 1, 1);
    Normalizer critic_norm{O};
    Adam actor_opt, critic_opt, std_opt;
    std::vector<float> std_grad(A, 0);
    std::mt19937 rng(c.seed);
    std::normal_distribution<float> normal(0, 1);

    std::vector<float> sobs(size_t(T + 1) * N * S), cobs(size_t(T + 1) * N * O), act(size_t(T) * N * A),
        logp(size_t(T) * N), val(size_t(T + 1) * N), rew(size_t(T) * N), adv(size_t(T) * N), ret(size_t(T) * N);
    std::vector<uint32_t> done(size_t(T) * N);
    std::vector<float> raw(size_t(N) * O), student(size_t(N) * S), executed(size_t(N) * A);

    env.reset(c.seed);
    env.set_level(1);
    env.observe(raw.data(), student.data());
    FILE *log = std::fopen(c.log.c_str(), "w");
    std::fprintf(log, "iteration,steps,seconds,reward,episodes,completion,contact,timeout,value_loss,clip_fraction,std\n");
    EnvTotals last = env.totals();
    uint64_t steps = 0;
    const double start = trainer_now();
    for (int it = 0; it < c.iterations; it++) {
        double reward_sum = 0;
        uint64_t timeouts = 0;
        for (int t = 0; t <= T; t++) {
            float *so = &sobs[size_t(t) * N * S], *co = &cobs[size_t(t) * N * O];
            critic_norm.update(raw.data(), N);
            std::copy(raw.begin(), raw.end(), co);
            critic_norm.apply(co, N);
            const float *v = critic.forward(co, N);
            std::copy(v, v + N, &val[size_t(t) * N]);
            if (t == T)
                break;
            for (int i = 0; i < N; i++) {
                float prior[NT_ACTIONS];
                nt_student_prior(&student[size_t(i) * S], prior);
                std::copy(prior, prior + A, &executed[size_t(i) * A]);
            }
            if (fresh && it < c.level_ramp)
                net.norm.update(student.data(), N);
            std::copy(student.begin(), student.end(), so);
            net.norm.apply(so, N);
            const float *mu = net.actor.forward(so, N);
            float *a = &act[size_t(t) * N * A];
            for (int i = 0; i < N; i++) {
                float lp = 0;
                for (int j = 0; j < A; j++) {
                    const float sd = std::exp(net.log_std[j]), e = normal(rng);
                    a[i * A + j] = mu[i * A + j] + sd * e;
                    lp += -.5f * e * e - net.log_std[j];
                }
                logp[size_t(t) * N + i] = lp;
                float prior[NT_ACTIONS];
                std::copy(&executed[size_t(i) * A], &executed[size_t(i) * A] + A, prior);
                nt_student_combine(prior, &a[i * A], &executed[size_t(i) * A]);
                nt_student_certify(&student[size_t(i) * S], &executed[size_t(i) * A]);
            }
            env.step(executed.data(), &rew[size_t(t) * N], &done[size_t(t) * N]);
            for (int i = 0; i < N; i++) {
                reward_sum += rew[size_t(t) * N + i];
                if (done[size_t(t) * N + i] == 2) {
                    rew[size_t(t) * N + i] += c.gamma * val[size_t(t) * N + i];
                    ++timeouts;
                }
            }
            env.observe(raw.data(), student.data());
        }
        steps += uint64_t(N) * T;
        for (int i = 0; i < N; i++) {
            float g = 0;
            for (int t = T - 1; t >= 0; t--) {
                const size_t k = size_t(t) * N + i;
                const float mask = done[k] ? 0 : 1;
                const float delta = rew[k] + c.gamma * mask * val[k + N] - val[k];
                g = delta + c.gamma * c.lambda * mask * g;
                adv[k] = g;
                ret[k] = g + val[k];
            }
        }
        const int total = N * T, batch = total / c.minibatches;
        std::vector<int> order(total);
        std::iota(order.begin(), order.end(), 0);
        std::vector<float> ms(size_t(batch) * S), mc(size_t(batch) * O), ma(size_t(batch) * A), mlp_old(batch),
            madv(batch), mret(batch), dmu(size_t(batch) * A), dv(batch);
        double value_loss = 0, clipped = 0;
        int updates = 0;
        for (int epoch = 0; epoch < c.epochs; epoch++) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int mb = 0; mb < c.minibatches; mb++) {
                double am = 0, as = 0;
                for (int b = 0; b < batch; b++) {
                    const int k = order[size_t(mb) * batch + b];
                    std::copy(&sobs[size_t(k) * S], &sobs[size_t(k) * S] + S, &ms[size_t(b) * S]);
                    std::copy(&cobs[size_t(k) * O], &cobs[size_t(k) * O] + O, &mc[size_t(b) * O]);
                    std::copy(&act[size_t(k) * A], &act[size_t(k) * A] + A, &ma[size_t(b) * A]);
                    mlp_old[b] = logp[k];
                    madv[b] = adv[k];
                    mret[b] = ret[k];
                    am += adv[k];
                }
                am /= batch;
                for (int b = 0; b < batch; b++)
                    as += (madv[b] - am) * (madv[b] - am);
                as = std::sqrt(as / batch) + 1e-8;
                const float *mu = net.actor.forward(ms.data(), batch);
                std::fill(std_grad.begin(), std_grad.end(), 0.0f);
                for (int b = 0; b < batch; b++) {
                    const float A_ = float((madv[b] - am) / as);
                    float lp = 0;
                    for (int j = 0; j < A; j++) {
                        const float z = (ma[b * A + j] - mu[b * A + j]) / std::exp(net.log_std[j]);
                        lp += -.5f * z * z - net.log_std[j];
                    }
                    const float ratio = std::exp(lp - mlp_old[b]);
                    const bool active = !((A_ > 0 && ratio > 1 + c.clip) || (A_ < 0 && ratio < 1 - c.clip));
                    clipped += active ? 0 : 1;
                    const float dlp = active ? -A_ * ratio / batch : 0;
                    for (int j = 0; j < A; j++) {
                        const float sd = std::exp(net.log_std[j]), z = (ma[b * A + j] - mu[b * A + j]) / sd;
                        dmu[b * A + j] = dlp * z / sd;
                        std_grad[j] += dlp * (z * z - 1);
                    }
                }
                for (int j = 0; j < A; j++)
                    std_grad[j] -= c.entropy;
                net.actor.backward(dmu.data(), batch);
                const float *v = critic.forward(mc.data(), batch);
                for (int b = 0; b < batch; b++) {
                    const float e = v[b] - mret[b];
                    value_loss += .5 * e * e / batch;
                    dv[b] = c.value * e / batch;
                }
                critic.backward(dv.data(), batch);
                clip_grad_norm(net.actor.grads, c.grad_norm);
                clip_grad_norm(critic.grads, c.grad_norm);
                // The critic starts from nothing: let it settle before the actor moves.
                if (it >= c.level_ramp) {
                    actor_opt.step(net.actor.params, net.actor.grads, c.lr);
                    std_opt.step(net.log_std, std_grad, c.lr);
                }
                critic_opt.step(critic.params, critic.grads, c.lr);
                for (auto &x : net.log_std)
                    x = std::fmin(std::log(.1f), std::fmax(std::log(.005f), x));
                ++updates;
            }
        }
        const EnvTotals now = env.totals();
        const double episodes = double(now.episodes - last.episodes), e = std::max(1.0, episodes);
        std::fprintf(log, "%d,%llu,%.1f,%.5f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", it, (unsigned long long)steps,
                     trainer_now() - start, reward_sum / (double(N) * T), episodes, (now.completions - last.completions) / e,
                     (now.contacts - last.contacts) / e, timeouts / e, value_loss / updates,
                     clipped / (double(batch) * updates), std::exp(net.log_std[0]));
        std::fflush(log);
        last = now;
        if ((it + 1) % c.save_every == 0 || it + 1 == c.iterations)
            net.checkpoint().save(c.checkpoint);
    }
    std::fclose(log);
}
