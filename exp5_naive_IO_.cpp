// #include <cblas.h>
#include <cmath>
#include <chrono>
#include <thread>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iomanip>

// ---------------- utilities ----------------
template <class T>
inline void DoNotOptimize(T &&value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

inline void FakeUse(float v)
{
    asm volatile("" ::"r"(v));
}

// A C++ implementation to numpy.allclose()
bool allclose(const std::vector<float> &a,
              const std::vector<float> &b,
              float rtol = 1e-5f,
              float atol = 1e-6f)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
    {
        float diff = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::fabs(b[i]);
        if (diff > tol)
            return false;
    }
    return true;
}

static inline float row_max(const float *row, int len)
{
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; ++i)
        m = std::max(m, row[i]);
    return m;
}
static inline float row_sum(const float *row, int len)
{
    float s = 0.f;
    for (int i = 0; i < len; ++i)
        s += row[i];
    return s;
}
static inline void random_matrix(std::vector<float> &A)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto &x : A)
        x = dist(rng);
}

// ---------------- globle counters ----------------
std::vector<double> g_dt_naive_seg1;
std::vector<double> g_dt_naive_seg2;
std::vector<double> g_dt_naive_seg3;

// ---------------- naive attention ----------------
std::vector<float> naive_attention(const float *__restrict__ Q, const float *__restrict__ Kt, const float *__restrict__ V,
                                   int N, int d, float scale)
{
    // // for test
    // std::vector<float> _O(N * d);
    // return _O;
    // calculate S = QK transpose
    float tmp = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> S(N * N);

    // float sum;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp = 0.0f;
            // sum = 0;
            for (int k = 0; k < d; k++)
            {
                float a = Q[i * d + k];
                float b = Kt[k * N + j];
                FakeUse(a);
                FakeUse(b);

                // tmp += a * 0.0f + b * 0.0f;
                // sum += Q[i * d + k] * Kt[k * N + j];
            }
            S[i * N + j] = tmp;
            FakeUse(S[i * N + j]);

            // S[i * N + j] = sum * scale;
        }
    }
    // calculate P = softmax(S)
    auto t1 = std::chrono::steady_clock::now();
    g_dt_naive_seg1.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    std::vector<float> P(N * N);
    for (int i = 0; i < N; i++)
    {
        tmp = 0.0f;
        for (int j = 0; j < N; j++)
        {
            float v = S[i * N + j];

            FakeUse(v);
            // tmp += v * 0.0f;
        }
        // float m = row_max(S.data() + (i * N), N);
        for (int j = 0; j < N; ++j)
        {
            float v = S[i * N + j];
            P[i * N + j] = v;
            FakeUse(v);
            FakeUse(P[i * N + j]);
            // P[i * N + j] = std::exp(S[i * N + j] - m);
        }
        for (int j = 0; j < N; j++)
        {
            float v = P[i * N + j];
            FakeUse(v);
            // tmp += v;
        }
        // float sum = row_sum(P.data() + (i * N), N);
        // sum = (sum > 0.f) ? 1.f / sum : 0.f;
        for (int j = 0; j < N; ++j)
        {
            float v = P[i * N + j];
            P[i * N + j] = v;

            FakeUse(v);
            FakeUse(P[i * N + j]);
            // P[i * N + j] *= sum;
        }
    }
    // calculate O = PV
    auto t2 = std::chrono::steady_clock::now();
    g_dt_naive_seg2.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    std::vector<float> O(N * d);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            tmp = 0.0f;
            // sum = 0;
            for (int k = 0; k < N; k++)
            {
                float a = P[i * N + k];
                float b = V[k * d + j];
                FakeUse(a);
                FakeUse(b);
                // tmp += a * 0.0f + b * 0.0f;
                // sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = tmp;
            FakeUse(O[i * d + j]);
            // O[i * d + j] = sum;
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    g_dt_naive_seg3.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    return O;
}

std::string format_sig5(double x)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.5g", x);
    std::string s(buf);

    if (s.find('e') != std::string::npos || s.find('E') != std::string::npos)
    {
        double val = std::stod(s);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(10) << val;
        s = oss.str();
    }

    int int_digits = (x == 0 ? 1 : (int)std::floor(std::log10(std::fabs(x))) + 1);
    if (int_digits < 1)
        int_digits = 1;

    int decimals = std::max(0, 5 - int_digits);

    std::ostringstream oss2;
    oss2 << std::fixed << std::setprecision(decimals) << x;
    return oss2.str();
}

void print_row(const std::vector<double> &v)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        std::cout << format_sig5(v[i]);
        if (i + 1 < v.size())
            std::cout << '\t'; // Excel 分欄
    }
    std::cout << "\n";
}

// --------- Main: Compare correctness + timing ----------
int main()
{

    std::vector<int> D_sizes;
    std::vector<int> N_sizes;
    int min_trial = 5;
    int max_trial = 20;
    int time_limit = 10000;
    for (int i = 4; i < 13; i++)
    {
        N_sizes.push_back(1 << i);
    }
    for (int i = 4; i < 13; i++)
    {
        D_sizes.push_back(1 << i);
    }
    for (int d : D_sizes)
    {
        auto N_start = std::chrono::steady_clock::now();
        std::cout << "////////////////////////////////////////////////////////////////////////\n"
                  << "D=" << d << "\n////////////////////////////////////////////////////////////////////////\n";
        std::vector<double> dt_naive_mean;
        std::vector<double> dt_naive_mean_seg1;
        std::vector<double> dt_naive_mean_seg2;
        std::vector<double> dt_naive_mean_seg3;
        for (int N : N_sizes)
        {
            float scale = 1.0f / std::sqrt(float(d));
            std::vector<float> Q(N * d), Kt(d * N), V(N * d);
            random_matrix(Q);
            random_matrix(Kt);
            random_matrix(V);
            // --- warm-up (important) ---
            (void)naive_attention(Q.data(), Kt.data(), V.data(), N, d, scale);
            std::vector<double> dt_naive;
            int trial = max_trial;
            for (int i = 0; i < trial; ++i)
            {
                // --- time naive ---
                auto t1 = std::chrono::steady_clock::now();
                auto tmp = naive_attention(Q.data(), Kt.data(), V.data(), N, d, scale);
                DoNotOptimize(tmp.data());
                auto t2 = std::chrono::steady_clock::now();
                double t = std::chrono::duration<double, std::milli>(t2 - t1).count();
                dt_naive.push_back(t);
                trial = std::min(std::max((int)(time_limit / t), min_trial), max_trial);
            }
            double mean = std::accumulate(dt_naive.begin(), dt_naive.end(), 0.0) / dt_naive.size();
            double var = 0.0;
            for (double v : dt_naive)
                var += (v - mean) * (v - mean);
            var /= dt_naive.size();
            double stddev = std::sqrt(var);

            double mean_seg1 = std::accumulate(g_dt_naive_seg1.begin(), g_dt_naive_seg1.end(), 0.0) / g_dt_naive_seg1.size();
            double mean_seg2 = std::accumulate(g_dt_naive_seg2.begin(), g_dt_naive_seg2.end(), 0.0) / g_dt_naive_seg2.size();
            double mean_seg3 = std::accumulate(g_dt_naive_seg3.begin(), g_dt_naive_seg3.end(), 0.0) / g_dt_naive_seg3.size();
            dt_naive_mean.push_back(mean);
            dt_naive_mean_seg1.push_back(mean_seg1);
            dt_naive_mean_seg2.push_back(mean_seg2);
            dt_naive_mean_seg3.push_back(mean_seg3);
            g_dt_naive_seg1 = std::vector<double>();
            g_dt_naive_seg2 = std::vector<double>();
            g_dt_naive_seg3 = std::vector<double>();

            std::cout << "N=" << N << "\n========================================================================\n";
            std::cout << "\tNaive time: \n";
            std::cout << "\t  trial = " << trial << " \n";
            std::cout << "\t  mean  = " << mean << " ms\n";
            std::cout << "\t  seg1  = " << mean_seg1 << " ms\n";
            std::cout << "\t  seg2  = " << mean_seg2 << " ms\n";
            std::cout << "\t  seg3  = " << mean_seg3 << " ms\n";
            std::cout << "\t  stdev = " << stddev << " ms\n";
            std::cout << "========================================================================\n";
            std::cout.flush();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        auto N_end = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "========================================================================\n";
        std::cout << "d=" << d << ":\n";
        print_row(dt_naive_mean);
        std::cout << "\n";
        std::cout << "Seg1:\n";
        print_row(dt_naive_mean_seg1);
        std::cout << "\n";
        std::cout << "Seg2:\n";
        print_row(dt_naive_mean_seg2);
        std::cout << "\n";
        std::cout << "Seg3:\n";
        print_row(dt_naive_mean_seg3);
        std::cout << "\n";
        std::cout << "========================================================================\n";
        std::cout.flush();
        std::cout << "[INFO] Completed D = " << d
                  << ", sleeping 1 seconds before next N...\n";
        std::cout << "[INFO] Time used:  " << std::chrono::duration<double, std::milli>(N_end - N_start).count() << "\n";
        std::cout.flush();
    }
}