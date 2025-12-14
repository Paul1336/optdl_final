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

// ---------------- naive attention ----------------
std::vector<float> flash_attention(const float *__restrict__ Q, const float *__restrict__ Kt, const float *__restrict__ V,
                                   int N, int d, int M_bytes, float scale, bool if_print)
{
    // // for test
    // std::vector<float> _O(N * d);
    // return _O;
    std::vector<float> O(N * d, 0.f);
    // Set block sizes
    int bc = M_bytes / (4 * d * sizeof(float));
    int br = std::min(int(M_bytes / (4 * d * sizeof(float))), d);
    bc = std::min(std::max(bc, 1), N);
    br = std::min(std::max(br, 1), N);

    const int tc = (N + bc - 1) / bc;
    const int tr = (N + br - 1) / br;
    if (if_print)
    {

        std::cout << "M=" << M_bytes / 1024 << " KiB\n";
        std::cout << "\tFlashAttention block size: (" << br << ", " << bc << ")\n";
        std::cout << "\tFlashAttention block number: (" << tr << ", " << tc << ")\n";
    }
    // Initialize O, l, m
    std::vector<float> L(N, 0.f);
    std::vector<float> M(N, -std::numeric_limits<float>::infinity());
    std::vector<float> S(br * bc);
    std::vector<float> Mi_new(br);
    std::vector<float> Li_new(br);

    std::vector<float> Q_tile(br * d);
    std::vector<float> Kt_tile(d * bc);
    std::vector<float> V_tile(bc * d);
    float sum;
    for (int i = 0; i < tr; ++i)
    {
        // Load Qi, and the partial outputs/statistics (Oi, mi, ℓi) to on-chip memory
        int current_br = std::min(br, N - i * br);
        // const float *Q_i = Q + i * br * d;
        std::memcpy(Q_tile.data(), Q + i * br * d, current_br * d * sizeof(float));

        float *O_i = O.data() + i * br * d;
        float *L_i = L.data() + i * br;
        float *M_i = M.data() + i * br;

        for (int j = 0; j < tc; j++)
        {
            // Load Kj to on-chip memory.
            int current_bc = std::min(bc, N - j * bc);
            for (int m = 0; m < d; ++m)
            {
                std::memcpy(&Kt_tile[m * current_bc],
                            &Kt[m * N + j * bc],
                            current_bc * sizeof(float));
            }
            // compute Sij = QiKTj
            for (int k = 0; k < current_br; k++)
            {
                for (int l = 0; l < current_bc; l++)
                {
                    sum = 0;
                    for (int m = 0; m < d; m++)
                    {
                        sum += Q_tile[k * d + m] * Kt_tile[m * current_bc + l];
                    }
                    S[k * current_bc + l] = sum * scale;
                }
            }
            // mnewi = max(mi, rowmax(Sij ))
            for (int k = 0; k < current_br; k++)
            {
                Mi_new[k] = std::max(M_i[k], row_max(S.data() + (k * current_bc), current_bc));
                // Pij = exp(Sij − mnewi), ℓnewi = ℓie (mi−mnewi) + rowsum(Pij )
                for (int l = 0; l < current_bc; l++)
                {
                    S[k * current_bc + l] = std::exp(S[k * current_bc + l] - Mi_new[k]);
                }
                Li_new[k] = L_i[k] * std::exp(M_i[k] - Mi_new[k]) + row_sum(S.data() + k * current_bc, current_bc);
            }
            // Load Vj , Accumulate output
            std::memcpy(V_tile.data(),
                        V + j * bc * d,
                        current_bc * d * sizeof(float));
            for (int k = 0; k < current_br; k++)
            {
                float scale_old = L_i[k] * std::exp(M_i[k] - Mi_new[k]) / Li_new[k];
                float scale_new = 1 / Li_new[k];
                for (int l = 0; l < d; l++)
                {
                    sum = 0;
                    for (int m = 0; m < current_bc; m++)
                    {
                        sum += S[k * current_bc + m] * V_tile[m * d + l];
                    }
                    O_i[k * d + l] = O_i[k * d + l] * scale_old + sum * scale_new;
                }
                // Update mi, ℓi
                L_i[k] = Li_new[k];
                M_i[k] = Mi_new[k];
            }
        }
    }
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
    std::vector<int> M_sizes;
    int min_trial = 5;
    int max_trial = 20;
    int time_limit = 10000;
    for (int i = 2; i < 11; i++)
    {
        M_sizes.push_back((1 << i) * 1024);
    }
    // M_sizes.push_back(16 * 1024);
    for (int M : M_sizes)
    {
        for (int i = 4; i < 13; i++)
        {
            N_sizes.push_back(1 << i);
        }
        for (int i = 4; i < 13; i++)
        {
            D_sizes.push_back(1 << i);
        }
        std::cout << "************************************************************************\n"
                  << "M=" << M << "\n************************************************************************\n";
        for (int d : D_sizes)
        {
            auto N_start = std::chrono::steady_clock::now();
            std::cout << "////////////////////////////////////////////////////////////////////////\n"
                      << "D=" << d << "\n////////////////////////////////////////////////////////////////////////\n";
            std::vector<double> dt_naive_mean;
            for (int N : N_sizes)
            {
                float scale = 1.0f / std::sqrt(float(d));
                std::vector<float> Q(N * d), Kt(d * N), V(N * d);
                random_matrix(Q);
                random_matrix(Kt);
                random_matrix(V);
                // --- warm-up (important) ---
                //(void)naive_attention(Q.data(), Kt.data(), V.data(), N, d, scale);
                (void)flash_attention(Q.data(), Kt.data(), V.data(), N, d, M, scale, true);
                std::vector<double> dt_naive;
                int trial = max_trial;
                for (int i = 0; i < trial; ++i)
                {
                    // --- time naive ---
                    auto t1 = std::chrono::steady_clock::now();
                    auto tmp = flash_attention(Q.data(), Kt.data(), V.data(), N, d, M, scale, false);
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
                dt_naive_mean.push_back(mean);
                std::cout << "N=" << N << "\n========================================================================\n";
                std::cout << "\tFlashattention time: \n";
                std::cout << "\t  trial = " << trial << " \n";
                std::cout << "\t  mean  = " << mean << " ms\n";
                std::cout << "\t  stdev = " << stddev << " ms\n";
                std::cout << "========================================================================\n";
                std::cout.flush();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            auto N_end = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::seconds(5));
            std::cout << "========================================================================\n";
            std::cout << "D=" << d << ":\t";
            print_row(dt_naive_mean);

            std::cout << "\n";
            std::cout << "========================================================================\n";
            std::cout.flush();
            std::cout << "[INFO] Completed D = " << d
                      << ", sleeping 5 seconds before next N...\n";
            std::cout << "[INFO] Time used:  " << std::chrono::duration<double, std::milli>(N_end - N_start).count() << "\n";
            std::cout.flush();
        }
        N_sizes = std::vector<int>();
        D_sizes = std::vector<int>();
    }
}