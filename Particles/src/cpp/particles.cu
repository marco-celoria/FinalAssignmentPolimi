/*
================================================================================
Particle System Solver - CUDA C++17 Reference Solution
================================================================================

Official reference CUDA parallelization of the serial C++17 baseline for the
final HPC assignment. The numerical model is intentionally aligned with the
serial baseline; only the implementation strategy is changed.

Primary CUDA targets
--------------------
  1. computeForcesKernelTiled(...): O(N^2) all-pairs force kernel using a
     conservative one-thread-per-target-particle strategy and shared-memory
     tiling over source particles.
  2. halfKickDriftKernel(...) and halfKickKernel(...): Velocity-Verlet update.
  3. buildScreenKernel(...): optional visualization/debug screen for HDF5 mode.

Host-side initialization
------------------------
  The Mandelbrot/generating field is intentionally computed on the host using
  the same scalar algorithm and operation order as the serial C++17 baseline.
  This avoids CPU/GPU boundary-classification differences that can otherwise
  change the particle count and initial particle set by one or more particles.

Reference benchmark/no-output mode
----------------------------------
  ./particles_cuda input/Particles.in none 0

Reference HDF5 correctness mode
-------------------------------
  ./particles_cuda input/Particles.in output/Particles_cuda.h5 1000

Build without HDF5:
  nvcc -O3 -std=c++17 -Xcompiler "-Wall -Wextra -pedantic" \
       particles.cu -o particles_cuda

Build with HDF5:
  nvcc -O3 -std=c++17 -DUSE_HDF5 -Xcompiler "-Wall -Wextra -pedantic" \
       particles.cu -o particles_cuda -lhdf5_cpp -lhdf5

Depending on the site installation, HDF5 include/library paths may need to be
provided explicitly, for example with -I and -L options.

Command line:
  ./particles_cuda [inputFile] [h5File|none|--no-hdf5] [outputEvery]

Input format:
  Same as the serial baseline.

================================================================================
*/

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

// ============================================================
// CUDA ERROR HANDLING
// ============================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::ostringstream oss__;                                           \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                  << " code=" << static_cast<int>(err__)                        \
                  << " (" << cudaGetErrorString(err__) << ") while calling "    \
                  << #call;                                                     \
            throw std::runtime_error(oss__.str());                              \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST()                                                       \
    do {                                                                        \
        cudaError_t err__ = cudaGetLastError();                                 \
        if (err__ != cudaSuccess) {                                             \
            std::ostringstream oss__;                                           \
            oss__ << "CUDA kernel launch error at "                             \
                  << __FILE__ << ":" << __LINE__                                \
                  << " code=" << static_cast<int>(err__)                        \
                  << " (" << cudaGetErrorString(err__) << ")";                  \
            throw std::runtime_error(oss__.str());                              \
        }                                                                       \
    } while (0)

// ============================================================
// CONSTANTS: part of the numerical model. Do not change.
// ============================================================

constexpr double kForce = 1.0e-3;
constexpr double eps    = 1.0e-2;
constexpr double eps2   = eps * eps;
constexpr int CUDA_BLOCK_SIZE = 256;

// ============================================================
// RAII CUDA HELPERS
// ============================================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n) : ptr_(nullptr), size_(n) {
        if (size_ > 0) {
            if (size_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::overflow_error("DeviceBuffer allocation size overflow");
            }
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), size_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    friend void swap(DeviceBuffer& a, DeviceBuffer& b) noexcept {
        std::swap(a.ptr_, b.ptr_);
        std::swap(a.size_, b.size_);
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }

private:
    T* ptr_{nullptr};
    std::size_t size_{0};
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }

    ~CudaEvent() noexcept {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

float elapsedMilliseconds(const CudaEvent& start, const CudaEvent& stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
    return ms;
}

// ============================================================
// UTILITIES
// ============================================================

__host__ __device__
inline std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) noexcept {
    return i + j * nx;
}

std::size_t safeMul(std::size_t a, std::size_t b, const std::string& what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error("Overflow while computing " + what);
    }
    return a * b;
}

std::size_t safeGridSize(std::size_t nx, std::size_t ny) {
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument("Grid dimensions must be > 0");
    }
    return safeMul(nx, ny, "grid size");
}

long long parseStrictLongLong(const std::string& s, const std::string& what) {
    long long value = 0;
    const char* begin = s.data();
    const char* end   = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid " + what + ": '" + s + "'");
    }
    return value;
}

bool isNoHdf5Token(const std::string& text) {
    return text == "none" || text == "NONE" || text == "-" || text == "--no-hdf5";
}

void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [inputFile] [h5File|none|--no-hdf5] [outputEvery]\n"
        << "Examples:\n"
        << "  " << prog << " input/Particles.in none 0\n"
        << "  " << prog << " input/Particles.in output/Particles_cuda.h5 1000\n";
}

bool shouldWriteStep(std::size_t step, std::size_t finalStep, std::size_t outputEvery) {
    if (step == finalStep) {
        return true;
    }
    if (outputEvery == 0) {
        return false;
    }
    if (step == 0) {
        return true;
    }
    return (step % outputEvery) == 0;
}

void printCudaDeviceSummary() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "CUDA device:                " << device << "\n";
    std::cout << "CUDA device name:           " << prop.name << "\n";
    std::cout << "CUDA compute capability:    " << prop.major << "." << prop.minor << "\n";
    std::cout << "CUDA multiprocessors:       " << prop.multiProcessorCount << "\n";
    std::cout << "CUDA global memory:         "
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
}

// ============================================================
// DATA STRUCTURES
// ============================================================

struct Grid {
    std::size_t nx{};
    std::size_t ny{};
    double xs{};
    double xe{};
    double ys{};
    double ye{};
    std::vector<unsigned long long> values;

    void allocate() { values.assign(safeGridSize(nx, ny), 0ULL); }
};

struct Particles {
    std::size_t n{};
    std::vector<double> w;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> vx;
    std::vector<double> vy;

    void resize(std::size_t N) {
        if (N == 0) {
            throw std::runtime_error("Particles.resize: N must be > 0");
        }
        n = N;
        w.resize(N);
        x.resize(N);
        y.resize(N);
        vx.assign(N, 0.0);
        vy.assign(N, 0.0);
    }
};

struct Config {
    std::size_t maxIters{};
    std::size_t maxSteps{};
    std::size_t outputEvery{0};
    double dt{};
};

struct ValidationQuantities {
    double sum_x{};
    double sum_y{};
    double sum_vx{};
    double sum_vy{};
    double weighted_sum_x{};
    double weighted_sum_y{};
    double momentum_x{};
    double momentum_y{};
    double kinetic_energy{};
    double potential_like{};
    double energy_like{};
};

// ============================================================
// INPUT PARSER
// ============================================================

template <typename T>
bool parseLine(std::istream& in, T& value) {
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }

        std::istringstream iss(line);
        if (!(iss >> value)) {
            throw std::runtime_error("Parse error: " + line);
        }

        iss >> std::ws;
        if (!iss.eof()) {
            if (iss.peek() == '#') {
                return true;
            }
            throw std::runtime_error("Trailing junk: " + line);
        }
        return true;
    }
    return false;
}

template <typename T>
void requireLine(std::istream& in, T& value, const std::string& name) {
    if (!parseLine(in, value)) {
        throw std::runtime_error("Unexpected EOF while reading " + name);
    }
}

Config readInput(const std::string& file, Grid& g, Grid& pg) {
    std::ifstream in(file);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + file);
    }

    long long raw_g_nx = 0, raw_g_ny = 0, raw_pg_nx = 0, raw_pg_ny = 0;
    requireLine(in, raw_g_nx, "g.nx");
    requireLine(in, raw_g_ny, "g.ny");
    requireLine(in, g.xs, "g.xs");
    requireLine(in, g.xe, "g.xe");
    requireLine(in, g.ys, "g.ys");
    requireLine(in, g.ye, "g.ye");
    requireLine(in, raw_pg_nx, "pg.nx");
    requireLine(in, raw_pg_ny, "pg.ny");
    requireLine(in, pg.xs, "pg.xs");
    requireLine(in, pg.xe, "pg.xe");
    requireLine(in, pg.ys, "pg.ys");
    requireLine(in, pg.ye, "pg.ye");

    long long raw_max_iters = 0, raw_max_steps = 0, raw_output_every = 0;
    Config cfg{};
    requireLine(in, raw_max_iters, "maxIters");
    requireLine(in, raw_max_steps, "maxSteps");
    requireLine(in, cfg.dt, "dt");
    requireLine(in, raw_output_every, "outputEvery");

    if (raw_g_nx < 2 || raw_g_ny < 2 || raw_pg_nx < 2 || raw_pg_ny < 2) {
        throw std::runtime_error("Grids must have at least 2 points in each direction");
    }
    if (g.xe <= g.xs || g.ye <= g.ys) {
        throw std::runtime_error("Invalid generating domain");
    }
    if (pg.xe <= pg.xs || pg.ye <= pg.ys) {
        throw std::runtime_error("Invalid particle/screen domain");
    }
    if (cfg.dt <= 0.0) {
        throw std::runtime_error("dt must be > 0");
    }
    if (raw_max_iters <= 0 || raw_max_steps <= 0) {
        throw std::runtime_error("maxIters and maxSteps must be > 0");
    }
    if (raw_output_every < 0) {
        throw std::runtime_error("outputEvery must be >= 0");
    }

    g.nx  = static_cast<std::size_t>(raw_g_nx);
    g.ny  = static_cast<std::size_t>(raw_g_ny);
    pg.nx = static_cast<std::size_t>(raw_pg_nx);
    pg.ny = static_cast<std::size_t>(raw_pg_ny);
    cfg.maxIters    = static_cast<std::size_t>(raw_max_iters);
    cfg.maxSteps    = static_cast<std::size_t>(raw_max_steps);
    cfg.outputEvery = static_cast<std::size_t>(raw_output_every);

    g.allocate();
    pg.allocate();
    return cfg;
}

// ============================================================
// HOST MANDELBROT FIELD GENERATION
// ============================================================

void computeGeneratingField(Grid& g, std::size_t maxIter) {
    if (g.values.empty()) {
        throw std::runtime_error("computeGeneratingField: empty grid");
    }

    const double dx = (g.xe - g.xs) / static_cast<double>(g.nx - 1);
    const double dy = (g.ye - g.ys) / static_cast<double>(g.ny - 1);

    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const double ca = g.xs + static_cast<double>(i) * dx;
            const double cb = g.ys + static_cast<double>(j) * dy;

            double za = 0.0;
            double zb = 0.0;
            std::size_t iter = 0;

            while (iter < maxIter) {
                const double a = za * za - zb * zb + ca;
                const double b = 2.0 * za * zb + cb;

                za = a;
                zb = b;

                if (za * za + zb * zb > 4.0) {
                    break;
                }

                ++iter;
            }

            g.values[idx2D(i, j, g.nx)] = static_cast<unsigned long long>(iter);
        }
    }
}

// ============================================================
// CUDA KERNELS
// ============================================================

template <int TILE_SIZE>
__global__
void computeForcesKernelTiled(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ w,
    double* __restrict__ fx,
    double* __restrict__ fy,
    std::size_t N
) {
    __shared__ double sh_x[TILE_SIZE];
    __shared__ double sh_y[TILE_SIZE];
    __shared__ double sh_w[TILE_SIZE];

    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const bool active = (i < N);

    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double wi = active ? w[i] : 0.0;

    double fxi = 0.0;
    double fyi = 0.0;

    const std::size_t tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const std::size_t jLoad = tile * TILE_SIZE + static_cast<std::size_t>(tid);

        if (jLoad < N) {
            sh_x[tid] = x[jLoad];
            sh_y[tid] = y[jLoad];
            sh_w[tid] = w[jLoad];
        } else {
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_w[tid] = 0.0;
        }

        __syncthreads();

        if (active) {
#pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                const std::size_t j = tile * TILE_SIZE + static_cast<std::size_t>(k);
                if (j >= N || j == i) {
                    continue;
                }

                const double dx = sh_x[k] - xi;
                const double dy = sh_y[k] - yi;
                const double r2 = dx * dx + dy * dy + eps2;
                const double invr = 1.0 / sqrt(r2);
                const double invr3 = invr * invr * invr;
                const double coeff = kForce * wi * sh_w[k] * invr3;

                fxi += coeff * dx;
                fyi += coeff * dy;
            }
        }

        __syncthreads();
    }

    if (active) {
        fx[i] = fxi;
        fy[i] = fyi;
    }
}

__global__
void halfKickDriftKernel(
    double* x,
    double* y,
    double* vx,
    double* vy,
    const double* fx,
    const double* fy,
    const double* w,
    double dt,
    std::size_t N
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const double invm = 1.0 / w[i];
    vx[i] += 0.5 * fx[i] * invm * dt;
    vy[i] += 0.5 * fy[i] * invm * dt;
    x[i]  += vx[i] * dt;
    y[i]  += vy[i] * dt;
}

__global__
void halfKickKernel(
    double* vx,
    double* vy,
    const double* fx_new,
    const double* fy_new,
    const double* w,
    double dt,
    std::size_t N
) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const double invm = 1.0 / w[i];
    vx[i] += 0.5 * fx_new[i] * invm * dt;
    vy[i] += 0.5 * fy_new[i] * invm * dt;
}

__global__
void buildScreenKernel(
    unsigned long long* screen,
    const double* x,
    const double* y,
    const double* w,
    std::size_t nx,
    std::size_t ny,
    double xs,
    double xe,
    double ys,
    double ye,
    double wmin,
    double wr,
    std::size_t N
) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }

    const double invdx = static_cast<double>(nx - 1) / (xe - xs);
    const double invdy = static_cast<double>(ny - 1) / (ye - ys);

    int ix = static_cast<int>((x[n] - xs) * invdx);
    int iy = static_cast<int>((y[n] - ys) * invdy);
    if (ix < 0) {
        ix = 0;
    } else if (ix > static_cast<int>(nx - 1)) {
        ix = static_cast<int>(nx - 1);
    }

    if (iy < 0) {
        iy = 0;
    } else if (iy > static_cast<int>(ny - 1)) {
        iy = static_cast<int>(ny - 1);
    }

    int wp_i = static_cast<int>(10.0 * (w[n] - wmin) / wr);
    if (wp_i < 0) {
        wp_i = 0;
    } else if (wp_i > 1000) {
        wp_i = 1000;
    }
    const auto wp = static_cast<unsigned long long>(wp_i);

    for (int dj = -1; dj <= 1; ++dj) {
        const int jy = iy + dj;
        if (jy < 0 || jy >= static_cast<int>(ny)) {
            continue;
        }
        for (int di = -1; di <= 1; ++di) {
            const int jx = ix + di;
            if (jx < 0 || jx >= static_cast<int>(nx)) {
                continue;
            }
            const std::size_t p = static_cast<std::size_t>(jx) + static_cast<std::size_t>(jy) * nx;
            atomicAdd(&screen[p], wp);
        }
    }
}

// ============================================================
// CPU PARTICLE GENERATION
// ============================================================

Particles generateParticles(const Grid& g, const Grid& pg) {
    if (g.values.empty()) {
        throw std::runtime_error("generateParticles: empty generating field");
    }

    Particles P;
    const auto vmax = *std::max_element(g.values.begin(), g.values.end());
    auto vmin       = *std::min_element(g.values.begin(), g.values.end());

    const unsigned long long qmax = vmax / 30ULL;
    const unsigned long long rmax = vmax % 30ULL;
    const unsigned long long qmin = vmin / 30ULL;
    const unsigned long long rmin = vmin % 30ULL;
    vmin = 29ULL * qmax + qmin + (29ULL * rmax + rmin) / 30ULL;

    const std::size_t count = static_cast<std::size_t>(
        std::count_if(g.values.begin(), g.values.end(), [&](unsigned long long v) {
            return v >= vmin;
        })
    );

    if (count == 0) {
        throw std::runtime_error("No particles generated");
    }

    P.resize(count);
    std::size_t n = 0;

    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto v = g.values[idx2D(i, j, g.nx)];
            if (v < vmin) {
                continue;
            }

            P.w[n] = std::max(1.0, 10.0 * static_cast<double>(v));
            P.x[n] = pg.xs + (pg.xe - pg.xs) * static_cast<double>(i) / static_cast<double>(g.nx - 1);
            P.y[n] = pg.ys + (pg.ye - pg.ys) * static_cast<double>(j) / static_cast<double>(g.ny - 1);
            P.vx[n] = 0.0;
            P.vy[n] = 0.0;
            ++n;
        }
    }

    if (n != count) {
        throw std::runtime_error("generateParticles: internal particle count mismatch");
    }
    return P;
}

// ============================================================
// VALIDATION QUANTITIES
// ============================================================

ValidationQuantities computeValidationQuantities(const Particles& P) {
    ValidationQuantities q{};
    const std::size_t N = P.n;

    for (std::size_t i = 0; i < N; ++i) {
        const double wi  = P.w[i];
        const double xi  = P.x[i];
        const double yi  = P.y[i];
        const double vxi = P.vx[i];
        const double vyi = P.vy[i];

        q.sum_x += xi;
        q.sum_y += yi;
        q.sum_vx += vxi;
        q.sum_vy += vyi;
        q.weighted_sum_x += wi * xi;
        q.weighted_sum_y += wi * yi;
        q.momentum_x += wi * vxi;
        q.momentum_y += wi * vyi;
        q.kinetic_energy += 0.5 * wi * (vxi * vxi + vyi * vyi);
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            const double dx = P.x[j] - P.x[i];
            const double dy = P.y[j] - P.y[i];
            const double r2 = dx * dx + dy * dy + eps2;
            q.potential_like += kForce * P.w[i] * P.w[j] / std::sqrt(r2);
        }
    }

    q.energy_like = q.kinetic_energy + q.potential_like;
    return q;
}

void printValidationQuantities(const ValidationQuantities& q) {
    std::cout << std::setprecision(17);
    std::cout << "Final validation quantities:\n";
    std::cout << "  sum_x:            " << q.sum_x << "\n";
    std::cout << "  sum_y:            " << q.sum_y << "\n";
    std::cout << "  sum_vx:           " << q.sum_vx << "\n";
    std::cout << "  sum_vy:           " << q.sum_vy << "\n";
    std::cout << "  weighted_sum_x:   " << q.weighted_sum_x << "\n";
    std::cout << "  weighted_sum_y:   " << q.weighted_sum_y << "\n";
    std::cout << "  momentum_x:       " << q.momentum_x << "\n";
    std::cout << "  momentum_y:       " << q.momentum_y << "\n";
    std::cout << "  kinetic_energy:   " << q.kinetic_energy << "\n";
    std::cout << "  potential_like:   " << q.potential_like << "\n";
    std::cout << "  energy_like:      " << q.energy_like << "\n";
}

// ============================================================
// HDF5 STREAM WRITER
// ============================================================

#ifdef USE_HDF5

class H5StreamWriter {
public:
    H5StreamWriter(
        const std::string& name,
        std::size_t np,
        std::size_t nx,
        std::size_t ny,
        std::size_t chunkFrames = 64,
        std::size_t screenTileY = 256,
        std::size_t screenTileX = 256
    )
        : file_(name, H5F_ACC_TRUNC),
          np_(np),
          nx_(nx),
          ny_(ny),
          chunkFrames_(chunkFrames),
          capacity_(chunkFrames),
          currentFrame_(0),
          Pbuf_(safeMul(np, 2, "position buffer size")),
          Vbuf_(safeMul(np, 2, "velocity buffer size")),
          closed_(false)
    {
        if (np_ == 0) {
            throw std::invalid_argument("H5StreamWriter: np must be > 0");
        }
        if (nx_ == 0 || ny_ == 0) {
            throw std::invalid_argument("H5StreamWriter: nx and ny must be > 0");
        }
        if (chunkFrames_ == 0) {
            throw std::invalid_argument("H5StreamWriter: chunkFrames must be > 0");
        }
        if (screenTileY == 0 || screenTileX == 0) {
            throw std::invalid_argument("H5StreamWriter: screen tile sizes must be > 0");
        }

        createParticleDatasets();
        createScreenDataset(screenTileY, screenTileX);
        createStepDataset();
        createWeightDataset();
        extendDatasets(capacity_);
    }

    ~H5StreamWriter() noexcept {
        try { close(); } catch (...) {}
    }

    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;

    void writeMetadata(const std::string& inputFile, const Config& cfg, const Grid& gen, const Grid& screenGrid) {
        H5::Group root = file_.openGroup("/");
        writeStringAttribute(root, "application", "Particle System Solver - CUDA Reference");
        writeStringAttribute(root, "format_version", "2.0");
        writeStringAttribute(root, "input_file", inputFile);
        writeStringAttribute(root, "screen_dataset_note", "For visualization/debugging; not recommended for strict grading.");

        writeULLAttribute(root, "particles", static_cast<unsigned long long>(np_));
        writeULLAttribute(root, "generating_grid_nx", static_cast<unsigned long long>(gen.nx));
        writeULLAttribute(root, "generating_grid_ny", static_cast<unsigned long long>(gen.ny));
        writeDoubleAttribute(root, "generating_grid_xs", gen.xs);
        writeDoubleAttribute(root, "generating_grid_xe", gen.xe);
        writeDoubleAttribute(root, "generating_grid_ys", gen.ys);
        writeDoubleAttribute(root, "generating_grid_ye", gen.ye);
        writeULLAttribute(root, "screen_grid_nx", static_cast<unsigned long long>(screenGrid.nx));
        writeULLAttribute(root, "screen_grid_ny", static_cast<unsigned long long>(screenGrid.ny));
        writeDoubleAttribute(root, "screen_grid_xs", screenGrid.xs);
        writeDoubleAttribute(root, "screen_grid_xe", screenGrid.xe);
        writeDoubleAttribute(root, "screen_grid_ys", screenGrid.ys);
        writeDoubleAttribute(root, "screen_grid_ye", screenGrid.ye);
        writeULLAttribute(root, "max_iters", static_cast<unsigned long long>(cfg.maxIters));
        writeULLAttribute(root, "max_steps", static_cast<unsigned long long>(cfg.maxSteps));
        writeULLAttribute(root, "output_every", static_cast<unsigned long long>(cfg.outputEvery));
        writeDoubleAttribute(root, "dt", cfg.dt);
        writeDoubleAttribute(root, "kForce", kForce);
        writeDoubleAttribute(root, "eps", eps);
        writeDoubleAttribute(root, "eps2", eps2);
        root.close();
    }

    void writeWeights(const Particles& P) {
        if (closed_) {
            throw std::runtime_error("H5StreamWriter: writeWeights after close");
        }
        if (P.n != np_) {
            throw std::runtime_error("H5StreamWriter: particle size mismatch in writeWeights");
        }
        weight_.write(P.w.data(), H5::PredType::NATIVE_DOUBLE);
    }

    void writeFrame(
        std::size_t stepNumber,
        const double* x,
        const double* y,
        const double* vx,
        const double* vy,
        const unsigned long long* screen
    ) {
        if (closed_) {
            throw std::runtime_error("H5StreamWriter: write after close");
        }
        if (!x || !y || !vx || !vy || !screen) {
            throw std::runtime_error("H5StreamWriter: null input pointer");
        }
        if (currentFrame_ >= capacity_) {
            capacity_ += chunkFrames_;
            extendDatasets(capacity_);
        }

        for (std::size_t i = 0; i < np_; ++i) {
            Pbuf_[2 * i]     = x[i];
            Pbuf_[2 * i + 1] = y[i];
            Vbuf_[2 * i]     = vx[i];
            Vbuf_[2 * i + 1] = vy[i];
        }

        writeDoubleFrame(pos_, Pbuf_.data(), currentFrame_, np_, 2);
        writeDoubleFrame(vel_, Vbuf_.data(), currentFrame_, np_, 2);
        writeScreenFrame(screen_, screen, currentFrame_, ny_, nx_);
        writeStep(step_, static_cast<long long>(stepNumber), currentFrame_);
        ++currentFrame_;
    }

    void close() {
        if (closed_) {
            return;
        }
        shrinkToFit();
        file_.flush(H5F_SCOPE_GLOBAL);
        pos_.close();
        vel_.close();
        screen_.close();
        step_.close();
        weight_.close();
        file_.close();
        closed_ = true;
    }

private:
    void createParticleDatasets() {
        hsize_t dims[3] = {0, static_cast<hsize_t>(np_), 2};
        hsize_t maxdims[3] = {H5S_UNLIMITED, static_cast<hsize_t>(np_), 2};
        H5::DataSpace space(3, dims, maxdims);
        H5::DSetCreatPropList prop;
        hsize_t chunk[3] = {1, static_cast<hsize_t>(np_), 2};
        prop.setChunk(3, chunk);
        pos_ = file_.createDataSet("/pos", H5::PredType::NATIVE_DOUBLE, space, prop);
        vel_ = file_.createDataSet("/vel", H5::PredType::NATIVE_DOUBLE, space, prop);
    }

    void createScreenDataset(std::size_t screenTileY, std::size_t screenTileX) {
        const hsize_t screenChunkY = static_cast<hsize_t>(std::min<std::size_t>(ny_, screenTileY));
        const hsize_t screenChunkX = static_cast<hsize_t>(std::min<std::size_t>(nx_, screenTileX));
        hsize_t dims[3] = {0, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        hsize_t maxdims[3] = {H5S_UNLIMITED, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        H5::DataSpace space(3, dims, maxdims);
        H5::DSetCreatPropList prop;
        hsize_t chunk[3] = {1, screenChunkY, screenChunkX};
        prop.setChunk(3, chunk);
        screen_ = file_.createDataSet("/screen", H5::PredType::NATIVE_ULLONG, space, prop);
    }

    void createStepDataset() {
        hsize_t dims[1] = {0};
        hsize_t maxdims[1] = {H5S_UNLIMITED};
        H5::DataSpace space(1, dims, maxdims);
        H5::DSetCreatPropList prop;
        hsize_t chunk[1] = {static_cast<hsize_t>(chunkFrames_)};
        prop.setChunk(1, chunk);
        step_ = file_.createDataSet("/step", H5::PredType::NATIVE_LLONG, space, prop);
    }

    void createWeightDataset() {
        hsize_t dims[1] = {static_cast<hsize_t>(np_)};
        H5::DataSpace space(1, dims);
        weight_ = file_.createDataSet("/weight", H5::PredType::NATIVE_DOUBLE, space);
    }

    void extendDatasets(hsize_t newSize) {
        std::array<hsize_t, 3> particleSize = {newSize, static_cast<hsize_t>(np_), 2};
        pos_.extend(particleSize.data());
        vel_.extend(particleSize.data());
        std::array<hsize_t, 3> screenSize = {newSize, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
        screen_.extend(screenSize.data());
        std::array<hsize_t, 1> stepSize = {newSize};
        step_.extend(stepSize.data());
    }

    void shrinkToFit() { extendDatasets(currentFrame_); }

    static void writeStringAttribute(H5::H5Object& object, const std::string& name, const std::string& value) {
        H5::DataSpace space(H5S_SCALAR);
        H5::StrType type(H5::PredType::C_S1, value.size() + 1);
        H5::Attribute attr = object.createAttribute(name, type, space);
        attr.write(type, value.c_str());
        attr.close();
    }

    static void writeULLAttribute(H5::H5Object& object, const std::string& name, unsigned long long value) {
        H5::DataSpace space(H5S_SCALAR);
        H5::Attribute attr = object.createAttribute(name, H5::PredType::NATIVE_ULLONG, space);
        attr.write(H5::PredType::NATIVE_ULLONG, &value);
        attr.close();
    }

    static void writeDoubleAttribute(H5::H5Object& object, const std::string& name, double value) {
        H5::DataSpace space(H5S_SCALAR);
        H5::Attribute attr = object.createAttribute(name, H5::PredType::NATIVE_DOUBLE, space);
        attr.write(H5::PredType::NATIVE_DOUBLE, &value);
        attr.close();
    }

    static void writeDoubleFrame(H5::DataSet& dataset, const double* data, hsize_t frame, hsize_t dim1, hsize_t dim2) {
        H5::DataSpace filespace = dataset.getSpace();
        hsize_t start[3] = {frame, 0, 0};
        hsize_t count[3] = {1, dim1, dim2};
        filespace.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace memspace(3, count);
        dataset.write(data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
    }

    static void writeScreenFrame(H5::DataSet& dataset, const unsigned long long* data, hsize_t frame, hsize_t ny, hsize_t nx) {
        H5::DataSpace filespace = dataset.getSpace();
        hsize_t start[3] = {frame, 0, 0};
        hsize_t count[3] = {1, ny, nx};
        filespace.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace memspace(3, count);
        dataset.write(data, H5::PredType::NATIVE_ULLONG, memspace, filespace);
    }

    static void writeStep(H5::DataSet& dataset, long long stepValue, hsize_t frame) {
        H5::DataSpace filespace = dataset.getSpace();
        hsize_t start[1] = {frame};
        hsize_t count[1] = {1};
        filespace.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace memspace(1, count);
        dataset.write(&stepValue, H5::PredType::NATIVE_LLONG, memspace, filespace);
    }

private:
    H5::H5File file_;
    H5::DataSet pos_;
    H5::DataSet vel_;
    H5::DataSet screen_;
    H5::DataSet step_;
    H5::DataSet weight_;
    std::size_t np_{};
    std::size_t nx_{};
    std::size_t ny_{};
    std::size_t chunkFrames_{};
    hsize_t capacity_{};
    hsize_t currentFrame_{};
    std::vector<double> Pbuf_;
    std::vector<double> Vbuf_;
    bool closed_{false};
};

#else

class H5StreamWriter {
public:
    H5StreamWriter(const std::string&, std::size_t, std::size_t, std::size_t, std::size_t = 64, std::size_t = 256, std::size_t = 256) {
        throw std::runtime_error(
            "This executable was built without HDF5 support. Use 'none' as the HDF5 output argument, or rebuild with -DUSE_HDF5."
        );
    }
    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;
    void writeMetadata(const std::string&, const Config&, const Grid&, const Grid&) {}
    void writeWeights(const Particles&) {}
    void writeFrame(std::size_t, const double*, const double*, const double*, const double*, const unsigned long long*) {}
    void close() {}
};

#endif

// ============================================================
// CUDA LAUNCH HELPERS
// ============================================================

unsigned blocks1D(std::size_t n, int threads) {
    if (threads <= 0) {
        throw std::runtime_error("blocks1D: invalid thread count");
    }

    const std::size_t blocks = (n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);

    if (blocks > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
        throw std::runtime_error("blocks1D: grid dimension overflow");
    }

    return static_cast<unsigned>(blocks);
}

void launchComputeForces(const double* d_x, const double* d_y, const double* d_w, double* d_fx, double* d_fy, std::size_t N) {
    computeForcesKernelTiled<CUDA_BLOCK_SIZE><<<blocks1D(N, CUDA_BLOCK_SIZE), CUDA_BLOCK_SIZE>>>(
        d_x, d_y, d_w, d_fx, d_fy, N
    );
    CUDA_CHECK_LAST();
}

void launchHalfKickDrift(double* d_x, double* d_y, double* d_vx, double* d_vy, const double* d_fx, const double* d_fy, const double* d_w, double dt, std::size_t N) {
    halfKickDriftKernel<<<blocks1D(N, CUDA_BLOCK_SIZE), CUDA_BLOCK_SIZE>>>(
        d_x, d_y, d_vx, d_vy, d_fx, d_fy, d_w, dt, N
    );
    CUDA_CHECK_LAST();
}

void launchHalfKick(double* d_vx, double* d_vy, const double* d_fx_new, const double* d_fy_new, const double* d_w, double dt, std::size_t N) {
    halfKickKernel<<<blocks1D(N, CUDA_BLOCK_SIZE), CUDA_BLOCK_SIZE>>>(
        d_vx, d_vy, d_fx_new, d_fy_new, d_w, dt, N
    );
    CUDA_CHECK_LAST();
}

void launchBuildScreen(unsigned long long* d_screen, const double* d_x, const double* d_y, const double* d_w, const Grid& screen, double wmin, double wr, std::size_t N) {
    buildScreenKernel<<<blocks1D(N, CUDA_BLOCK_SIZE), CUDA_BLOCK_SIZE>>>(
        d_screen,
        d_x,
        d_y,
        d_w,
        screen.nx,
        screen.ny,
        screen.xs,
        screen.xe,
        screen.ys,
        screen.ye,
        wmin,
        wr,
        N
    );
    CUDA_CHECK_LAST();
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
#ifdef USE_HDF5
    H5::Exception::dontPrint();
#endif

    try {
        if (argc > 4) {
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }

        const std::string inputFile  = (argc > 1) ? argv[1] : "input/Particles.in";
        const std::string outputFile = (argc > 2) ? argv[2] : "none";
        const bool writeHdf5 = !isNoHdf5Token(outputFile);

#ifndef USE_HDF5
        if (writeHdf5) {
            throw std::runtime_error(
                "HDF5 output requested, but this executable was built without HDF5 support. Use 'none' or rebuild with -DUSE_HDF5."
            );
        }
#endif

        CUDA_CHECK(cudaSetDevice(0));

        Grid gen;
        Grid screen;
        Config cfg = readInput(inputFile, gen, screen);

        if (argc > 3) {
            const long long outEvery = parseStrictLongLong(argv[3], "outputEvery");
            if (outEvery < 0) {
                throw std::runtime_error("outputEvery must be >= 0");
            }
            cfg.outputEvery = static_cast<std::size_t>(outEvery);
        }

        const std::size_t screenSize = safeGridSize(screen.nx, screen.ny);
        const std::size_t screenBytes = safeMul(screenSize, sizeof(unsigned long long), "screen bytes");

        std::cout << std::setprecision(17);
        std::cout << "Input file:                 " << inputFile << "\n";
#ifdef USE_HDF5
        std::cout << "HDF5 compiled:              yes\n";
#else
        std::cout << "HDF5 compiled:              no\n";
#endif
        std::cout << "HDF5 output:                " << (writeHdf5 ? outputFile : "disabled") << "\n";
        std::cout << "Benchmark/no-output mode:   " << (!writeHdf5 ? "yes" : "no") << "\n";
        std::cout << "Generating grid:            " << gen.nx << " x " << gen.ny << "\n";
        std::cout << "Screen grid:                " << screen.nx << " x " << screen.ny << "\n";
        std::cout << "Max iterations:             " << cfg.maxIters << "\n";
        std::cout << "Steps:                      " << cfg.maxSteps << "\n";
        std::cout << "dt:                         " << cfg.dt << "\n";
        if (cfg.outputEvery == 0) {
            std::cout << "Output policy:              final frame only, if HDF5 is enabled\n";
        } else {
            std::cout << "Output policy:              step 0, every "
                      << cfg.outputEvery
                      << " step(s), and final step, if HDF5 is enabled\n";
        }
        printCudaDeviceSummary();

        // ----------------------------------------------------
        // 1. Generate Mandelbrot field on host to match baseline
        // ----------------------------------------------------
        const auto mandelStart = std::chrono::steady_clock::now();
        computeGeneratingField(gen, cfg.maxIters);
        const auto mandelStop = std::chrono::steady_clock::now();
        const double mandelCpuSeconds = std::chrono::duration<double>(mandelStop - mandelStart).count();

        // ----------------------------------------------------
        // 2. Generate particles on host
        // ----------------------------------------------------
        const auto particleStart = std::chrono::steady_clock::now();
        Particles P = generateParticles(gen, screen);
        const auto particleStop = std::chrono::steady_clock::now();
        const double particleGenerationSeconds = std::chrono::duration<double>(particleStop - particleStart).count();

        const std::size_t N = P.n;
        const std::size_t particleBytes = safeMul(N, sizeof(double), "particle bytes");
        std::cout << "Particles:                  " << N << "\n";

        // ----------------------------------------------------
        // 3. Allocate device buffers and copy initial state
        // ----------------------------------------------------
        DeviceBuffer<double> d_x(N), d_y(N), d_vx(N), d_vy(N), d_w(N);
        DeviceBuffer<double> d_fx(N), d_fy(N), d_fx_new(N), d_fy_new(N);
        DeviceBuffer<unsigned long long> d_screen(screenSize);

        CUDA_CHECK(cudaMemcpy(d_x.get(),  P.x.data(),  particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y.get(),  P.y.data(),  particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx.get(), P.vx.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy.get(), P.vy.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w.get(),  P.w.data(),  particleBytes, cudaMemcpyHostToDevice));

        const auto [wminIt, wmaxIt] = std::minmax_element(P.w.begin(), P.w.end());
        const double wmin = *wminIt;
        const double wmax = *wmaxIt;
        const double wr   = std::max(wmax - wmin, 1.0);

        // ----------------------------------------------------
        // 4. Initial force computation
        // ----------------------------------------------------
        CudaEvent initForceStart;
        CudaEvent initForceStop;
        CUDA_CHECK(cudaEventRecord(initForceStart.get()));
        launchComputeForces(d_x.get(), d_y.get(), d_w.get(), d_fx.get(), d_fy.get(), N);
        CUDA_CHECK(cudaEventRecord(initForceStop.get()));
        CUDA_CHECK(cudaEventSynchronize(initForceStop.get()));
        const double initForceGpuSeconds = static_cast<double>(elapsedMilliseconds(initForceStart, initForceStop)) / 1000.0;

        // ----------------------------------------------------
        // 5. Optional HDF5 writer and host staging buffers
        // ----------------------------------------------------
        std::unique_ptr<H5StreamWriter> h5;
        if (writeHdf5) {
            h5 = std::make_unique<H5StreamWriter>(outputFile, N, screen.nx, screen.ny);
            h5->writeMetadata(inputFile, cfg, gen, screen);
            h5->writeWeights(P);
        }

        std::vector<double> h_x(N), h_y(N), h_vx(N), h_vy(N);
        std::vector<unsigned long long> h_screen(screenSize);

        std::size_t outputFrames = 0;
        bool hasLastWrittenStep = false;
        std::size_t lastWrittenStep = 0;
        double screenBuildSeconds = 0.0;
        double deviceToHostSeconds = 0.0;
        double hdf5WriteSeconds = 0.0;

        auto copyParticlesDeviceToHost = [&]() {
            CUDA_CHECK(cudaMemcpy(h_x.data(),  d_x.get(),  particleBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_y.data(),  d_y.get(),  particleBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx.get(), particleBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy.get(), particleBytes, cudaMemcpyDeviceToHost));
        };

        auto writeOutputFrame = [&](std::size_t step) {
            if (!h5) {
                return;
            }
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            const auto screenStart = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemset(d_screen.get(), 0, screenBytes));
            launchBuildScreen(d_screen.get(), d_x.get(), d_y.get(), d_w.get(), screen, wmin, wr, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto screenStop = std::chrono::steady_clock::now();
            screenBuildSeconds += std::chrono::duration<double>(screenStop - screenStart).count();

            const auto copyStart = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaMemcpy(h_screen.data(), d_screen.get(), screenBytes, cudaMemcpyDeviceToHost));
            copyParticlesDeviceToHost();
            const auto copyStop = std::chrono::steady_clock::now();
            deviceToHostSeconds += std::chrono::duration<double>(copyStop - copyStart).count();

            const auto h5Start = std::chrono::steady_clock::now();
            h5->writeFrame(step, h_x.data(), h_y.data(), h_vx.data(), h_vy.data(), h_screen.data());
            const auto h5Stop = std::chrono::steady_clock::now();
            hdf5WriteSeconds += std::chrono::duration<double>(h5Stop - h5Start).count();

            hasLastWrittenStep = true;
            lastWrittenStep = step;
            ++outputFrames;
        };

        // ----------------------------------------------------
        // 6. Simulation loop
        // ----------------------------------------------------
        const auto loopWallStart = std::chrono::steady_clock::now();
        CudaEvent dynamicsStart;
        CudaEvent dynamicsStop;
        double pureDynamicsGpuSeconds = 0.0;

        CUDA_CHECK(cudaEventRecord(dynamicsStart.get()));
        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            if (h5 && shouldWriteStep(step, cfg.maxSteps, cfg.outputEvery)) {
                CUDA_CHECK(cudaEventRecord(dynamicsStop.get()));
                CUDA_CHECK(cudaEventSynchronize(dynamicsStop.get()));
                pureDynamicsGpuSeconds += static_cast<double>(elapsedMilliseconds(dynamicsStart, dynamicsStop)) / 1000.0;

                writeOutputFrame(step);

                CUDA_CHECK(cudaEventRecord(dynamicsStart.get()));
            }

            launchHalfKickDrift(d_x.get(), d_y.get(), d_vx.get(), d_vy.get(), d_fx.get(), d_fy.get(), d_w.get(), cfg.dt, N);
            launchComputeForces(d_x.get(), d_y.get(), d_w.get(), d_fx_new.get(), d_fy_new.get(), N);
            launchHalfKick(d_vx.get(), d_vy.get(), d_fx_new.get(), d_fy_new.get(), d_w.get(), cfg.dt, N);
            swap(d_fx, d_fx_new);
            swap(d_fy, d_fy_new);
        }
        CUDA_CHECK(cudaEventRecord(dynamicsStop.get()));
        CUDA_CHECK(cudaEventSynchronize(dynamicsStop.get()));
        pureDynamicsGpuSeconds += static_cast<double>(elapsedMilliseconds(dynamicsStart, dynamicsStop)) / 1000.0;

        if (h5) {
            writeOutputFrame(cfg.maxSteps);
            h5->close();
        }

        const auto loopWallStop = std::chrono::steady_clock::now();
        const double loopWallSeconds = std::chrono::duration<double>(loopWallStop - loopWallStart).count();

        // ----------------------------------------------------
        // 7. Copy final state and validate
        // ----------------------------------------------------
        const auto finalCopyStart = std::chrono::steady_clock::now();
        copyParticlesDeviceToHost();
        const auto finalCopyStop = std::chrono::steady_clock::now();
        const double finalCopySeconds = std::chrono::duration<double>(finalCopyStop - finalCopyStart).count();

        P.x.swap(h_x);
        P.y.swap(h_y);
        P.vx.swap(h_vx);
        P.vy.swap(h_vy);

        const auto validationStart = std::chrono::steady_clock::now();
        const ValidationQuantities validation = computeValidationQuantities(P);
        const auto validationStop = std::chrono::steady_clock::now();
        const double validationSeconds = std::chrono::duration<double>(validationStop - validationStart).count();

        // ----------------------------------------------------
        // 8. Reporting
        // ----------------------------------------------------
        const double interactions = static_cast<double>(N) * static_cast<double>(N - 1) * static_cast<double>(cfg.maxSteps);
        const double gigaInteractions = interactions / 1.0e9;

        std::cout << "Simulation completed successfully.\n";
        std::cout << "Output frames:              " << outputFrames << "\n";
        std::cout << "Mandelbrot CPU wall time:   " << mandelCpuSeconds << " s\n";
        std::cout << "Particle generation wall:   " << particleGenerationSeconds << " s\n";
        std::cout << "Initial force GPU time:     " << initForceGpuSeconds << " s\n";
        std::cout << "Pure dynamics GPU time:     " << pureDynamicsGpuSeconds << " s\n";
        std::cout << "Screen build time:          " << screenBuildSeconds << " s\n";
        std::cout << "Device-to-host copy time:   " << deviceToHostSeconds << " s\n";
        std::cout << "Final state copy time:      " << finalCopySeconds << " s\n";
        std::cout << "HDF5 write time:            " << hdf5WriteSeconds << " s\n";
        std::cout << "Validation time:            " << validationSeconds << " s\n";
        std::cout << "Loop wall time:             " << loopWallSeconds << " s\n";

        if (cfg.maxSteps > 0 && pureDynamicsGpuSeconds > 0.0) {
            std::cout << "Pure dynamics performance:  "
                      << gigaInteractions / pureDynamicsGpuSeconds
                      << " GInteractions/s\n";
        }
        if (cfg.maxSteps > 0 && loopWallSeconds > 0.0) {
            std::cout << "Loop end-to-end performance: "
                      << gigaInteractions / loopWallSeconds
                      << " GInteractions/s\n";
        }

        printValidationQuantities(validation);
        CUDA_CHECK(cudaDeviceSynchronize());
        return EXIT_SUCCESS;
    }
#ifdef USE_HDF5
    catch (const H5::Exception& e) {
        std::cerr << "HDF5 ERROR: " << e.getDetailMsg() << "\n";
        return EXIT_FAILURE;
    }
#endif
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "ERROR: unknown exception\n";
        return EXIT_FAILURE;
    }
}

