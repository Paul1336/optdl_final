// #include <cblas.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

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

// ---------------- naive attention ----------------
std::vector<float> naive_attention(const float *__restrict__ Q, const float *__restrict__ Kt, const float *__restrict__ V,
                                   int N, int d, float scale)
{
    // calculate S = QK transpose
    std::vector<float> S(N * N);
    float sum;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sum = 0;
            for (int k = 0; k < d; k++)
            {
                sum += Q[i * d + k] * Kt[k * N + j];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // calculate P = softmax(S)
    std::vector<float> P(N * N);
    for (int i = 0; i < N; i++)
    {
        float m = row_max(S.data() + (i * N), N);
        for (int j = 0; j < N; ++j)
        {
            P[i * N + j] = std::exp(S[i * N + j] - m);
        }
        float sum = row_sum(P.data() + (i * N), N);
        sum = (sum > 0.f) ? 1.f / sum : 0.f;
        for (int j = 0; j < N; ++j)
        {
            P[i * N + j] *= sum;
        }
    }

    // calculate O = PV
    std::vector<float> O(N * d);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
    return O;
}

// --------------- FlashAttention ---------------
std::vector<float> flash_attention(const float *__restrict__ Q, const float *__restrict__ Kt, const float *__restrict__ V,
                                   int N, int d, int M_bytes, float scale, bool if_print)
{
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
    float sum;
    for (int i = 0; i < tr; ++i)
    {
        // Load Qi, and the partial outputs/statistics (Oi, mi, ℓi) to on-chip memory
        int current_br = std::min(br, N - i * br);
        const float *Q_i = Q + i * br * d;
        float *O_i = O.data() + i * br * d;
        float *L_i = L.data() + i * br;
        float *M_i = M.data() + i * br;

        for (int j = 0; j < tc; j++)
        {
            // Load Kj to on-chip memory.
            int current_bc = std::min(bc, N - j * bc);
            // compute Sij = QiKTj
            for (int k = 0; k < current_br; k++)
            {
                for (int l = 0; l < current_bc; l++)
                {
                    sum = 0;
                    for (int m = 0; m < d; m++)
                    {
                        sum += Q_i[k * d + m] * Kt[m * N + j * bc + l];
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
            for (int k = 0; k < current_br; k++)
            {
                float scale_old = L_i[k] * std::exp(M_i[k] - Mi_new[k]) / Li_new[k];
                float scale_new = 1 / Li_new[k];
                for (int l = 0; l < d; l++)
                {
                    sum = 0;
                    for (int m = 0; m < current_bc; m++)
                    {
                        sum += S[k * current_bc + m] * V[(j * bc + m) * d + l];
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

// --------- Main: Compare correctness + timing ----------
int main()
{

    // int N = 1024; // sequence length
    int d = 64; // head dimension
    float scale = 1.0f / std::sqrt(float(d));
    std::vector<int> M_sizes;
    std::vector<int> N_sizes;

    // for (int i = 2; (1 << i) <= 160 * 1024; i++)
    // {
    //     int base = (1 << i) * 1024;
    //     M_sizes.push_back(base);
    //     int sqrt2 = int(base * std::sqrt(2.0));
    //     M_sizes.push_back(sqrt2);
    // }
    M_sizes.push_back(16 * 1024);
    for (int i = 4; i < 14; i++)
    {
        N_sizes.push_back(1 << i);
    }
    for (int N : N_sizes)
    {
        std::cout << "------------------------------------------------------------------------\n"
                  << "N=" << N << "\n------------------------------------------------------------------------\n";
        int trial = 10;

        std::vector<float> Q(N * d), Kt(d * N), V(N * d);
        random_matrix(Q);
        random_matrix(Kt);
        random_matrix(V);

        // --- warm-up (important) ---
        (void)naive_attention(Q.data(), Kt.data(), V.data(), N, d, scale);
        std::vector<double> dt_naive;
        std::vector<float> O_naive;
        for (int i = 0; i < trial; ++i)
        {
            // --- time naive ---
            auto t1 = std::chrono::steady_clock::now();
            auto tmp = naive_attention(Q.data(), Kt.data(), V.data(), N, d, scale);
            auto t2 = std::chrono::steady_clock::now();
            DoNotOptimize(tmp.data());
            O_naive = tmp;
            dt_naive.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        double mean = std::accumulate(dt_naive.begin(), dt_naive.end(), 0.0) / dt_naive.size();
        double var = 0.0;
        for (double v : dt_naive)
            var += (v - mean) * (v - mean);
        var /= dt_naive.size();
        double stddev = std::sqrt(var);
        std::cout << "\tNaive time: \n";
        std::cout << "\t  mean  = " << mean << " ms\n";
        std::cout << "\t  stdev = " << stddev << " ms\n";
        // std::cout << "\tNaive time: " << dt_naive << " ms\n";

        for (int M_bytes : M_sizes)
        {
            // --- warm-up (important) ---
            (void)flash_attention(Q.data(), Kt.data(), V.data(), N, d, M_bytes, scale, true);
            std::vector<double> dt_flash;
            std::vector<float> O_flash;
            for (int i = 0; i < trial; ++i)
            {
                // --- time flash ---
                auto t3 = std::chrono::steady_clock::now();
                auto tmp = flash_attention(Q.data(), Kt.data(), V.data(), N, d, M_bytes, scale, false);
                auto t4 = std::chrono::steady_clock::now();
                DoNotOptimize(tmp.data());
                O_flash = tmp;
                dt_flash.push_back(std::chrono::duration<double, std::milli>(t4 - t3).count());
            }
            double mean = std::accumulate(dt_flash.begin(), dt_flash.end(), 0.0) / dt_flash.size();
            double var = 0.0;
            for (double v : dt_flash)
                var += (v - mean) * (v - mean);
            var /= dt_flash.size();
            double stddev = std::sqrt(var);

            std::cout << "\tFlashAttention time: \n";
            std::cout << "\t  mean  = " << mean << " ms\n";
            std::cout << "\t  stdev = " << stddev << " ms\n";
            if (allclose(O_naive, O_flash))
            {
                std::cout << "\tOutputs match within tolerance\n";
            }
            else
            {
                std::cout << "\tOutputs differ!\n";
            }
        }
    }
}