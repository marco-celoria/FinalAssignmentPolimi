#include <H5Cpp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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
// CONSTANTS
// ============================================================

constexpr double kForce = 1.0e-3;
constexpr double eps = 1.0e-2;
constexpr double eps2 = eps * eps;
constexpr int BLOCK_SIZE = 256;


// ============================================================
// RAII CUDA HELPERS
// ============================================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n)
        : ptr_(nullptr), size_(n)
    {
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&ptr_),
                size_ * sizeof(T)
            ));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
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


template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(std::size_t n)
        : ptr_(nullptr), size_(n)
    {
        if (size_ > 0) {
            CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&ptr_),
                size_ * sizeof(T),
                cudaHostAllocDefault
            ));
        }
    }

    ~PinnedHostBuffer() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }

            ptr_ = other.ptr_;
            size_ = other.size_;

            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    std::size_t size() const noexcept { return size_; }

private:
    T* ptr_{nullptr};
    std::size_t size_{0};
};


class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
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
    const char* end = s.data() + s.size();

    auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid " + what + ": '" + s + "'");
    }

    return value;
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

    void allocate() {
        values.assign(safeGridSize(nx, ny), 0ULL);
    }
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
    std::size_t outputEvery{10};
    double dt{};
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

    long long raw_g_nx = 0;
    long long raw_g_ny = 0;
    long long raw_pg_nx = 0;
    long long raw_pg_ny = 0;

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

    long long raw_max_iters = 0;
    long long raw_max_steps = 0;
    long long raw_output_every = 0;

    Config cfg{};

    requireLine(in, raw_max_iters, "maxIters");
    requireLine(in, raw_max_steps, "maxSteps");
    requireLine(in, cfg.dt, "dt");
    requireLine(in, raw_output_every, "outputEvery");

    if (raw_g_nx < 2 || raw_g_ny < 2 ||
        raw_pg_nx < 2 || raw_pg_ny < 2) {
        throw std::runtime_error("Grids must have at least 2 points");
    }

    if (g.xe <= g.xs || g.ye <= g.ys) {
        throw std::runtime_error("Invalid generating domain");
    }

    if (pg.xe <= pg.xs || pg.ye <= pg.ys) {
        throw std::runtime_error("Invalid particle domain");
    }

    if (cfg.dt <= 0.0) {
        throw std::runtime_error("dt must be > 0");
    }

    if (raw_max_iters <= 0 || raw_max_steps <= 0) {
        throw std::runtime_error("maxIters and maxSteps must be > 0");
    }

    if (raw_output_every <= 0) {
        throw std::runtime_error("outputEvery must be > 0");
    }

    g.nx = static_cast<std::size_t>(raw_g_nx);
    g.ny = static_cast<std::size_t>(raw_g_ny);
    pg.nx = static_cast<std::size_t>(raw_pg_nx);
    pg.ny = static_cast<std::size_t>(raw_pg_ny);

    cfg.maxIters = static_cast<std::size_t>(raw_max_iters);
    cfg.maxSteps = static_cast<std::size_t>(raw_max_steps);
    cfg.outputEvery = static_cast<std::size_t>(raw_output_every);

    g.allocate();
    pg.allocate();

    return cfg;
}


// ============================================================
// CUDA KERNELS
// ============================================================

__global__
void mandelbrotKernel(
    unsigned long long* values,
    std::size_t nx,
    std::size_t ny,
    double xs,
    double xe,
    double ys,
    double ye,
    std::size_t maxIter
) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t j =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    const double dx = (xe - xs) / static_cast<double>(nx - 1);
    const double dy = (ye - ys) / static_cast<double>(ny - 1);

    const double ca = xs + static_cast<double>(i) * dx;
    const double cb = ys + static_cast<double>(j) * dy;

    double za = 0.0;
    double zb = 0.0;

    std::size_t iter = 0;

    // Matches the Python CUDA particle implementation:
    // update z, then check escape, then increment if still bounded.
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

    values[idx2D(i, j, nx)] = static_cast<unsigned long long>(iter);
}


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

    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const int tid = threadIdx.x;

    const bool active = (i < N);

    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double wi = active ? w[i] : 0.0;

    double fxi = 0.0;
    double fyi = 0.0;

    const std::size_t tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (std::size_t tile = 0; tile < tiles; ++tile) {
        const std::size_t j = tile * TILE_SIZE + static_cast<std::size_t>(tid);

        if (j < N) {
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_w[tid] = w[j];
        } else {
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_w[tid] = 0.0;
        }

        __syncthreads();

        if (active) {
#pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                const std::size_t global_j =
                    tile * TILE_SIZE + static_cast<std::size_t>(k);

                if (global_j >= N || global_j == i) {
                    continue;
                }

                const double dx = sh_x[k] - xi;
                const double dy = sh_y[k] - yi;

                const double r2 = dx * dx + dy * dy + eps2;

                const double invr = 1.0 / sqrt(r2);
                const double invr2 = invr * invr;
                const double invr3 = invr2 * invr;

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
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    const double invm = 1.0 / w[i];

    vx[i] += 0.5 * fx[i] * invm * dt;
    vy[i] += 0.5 * fy[i] * invm * dt;

    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
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
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

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
    const std::size_t n =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (n >= N) {
        return;
    }

    const double invdx = (xe != xs)
        ? static_cast<double>(nx - 1) / (xe - xs)
        : 0.0;

    const double invdy = (ye != ys)
        ? static_cast<double>(ny - 1) / (ye - ys)
        : 0.0;

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

    int wp = static_cast<int>(10.0 * (w[n] - wmin) / wr);

    if (wp < 0) {
        wp = 0;
    } else if (wp > 1000) {
        wp = 1000;
    }

    const unsigned long long wp_u =
        static_cast<unsigned long long>(wp);

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

            const std::size_t p =
                static_cast<std::size_t>(jx)
                + static_cast<std::size_t>(jy) * nx;

            atomicAdd(&screen[p], wp_u);
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
    auto vmin = *std::min_element(g.values.begin(), g.values.end());

    vmin = (29ULL * vmax + vmin) / 30ULL;

    const std::size_t count = static_cast<std::size_t>(
        std::count_if(
            g.values.begin(),
            g.values.end(),
            [&](unsigned long long v) {
                return v >= vmin;
            }
        )
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

            P.x[n] = pg.xs
                + (pg.xe - pg.xs)
                * static_cast<double>(i)
                / static_cast<double>(g.nx - 1);

            P.y[n] = pg.ys
                + (pg.ye - pg.ys)
                * static_cast<double>(j)
                / static_cast<double>(g.ny - 1);

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
// HDF5 WRITER
// ============================================================

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

        const hsize_t screenChunkY =
            static_cast<hsize_t>(std::min<std::size_t>(ny_, screenTileY));

        const hsize_t screenChunkX =
            static_cast<hsize_t>(std::min<std::size_t>(nx_, screenTileX));

        // /pos and /vel datasets: [frame, particle, component]
        {
            hsize_t dims[3] = {
                0,
                static_cast<hsize_t>(np_),
                2
            };

            hsize_t maxdims[3] = {
                H5S_UNLIMITED,
                static_cast<hsize_t>(np_),
                2
            };

            H5::DataSpace space(3, dims, maxdims);
            H5::DSetCreatPropList prop;

            hsize_t chunk[3] = {
                1,
                static_cast<hsize_t>(np_),
                2
            };

            prop.setChunk(3, chunk);

            pos_ = file_.createDataSet(
                "/pos",
                H5::PredType::NATIVE_DOUBLE,
                space,
                prop
            );

            vel_ = file_.createDataSet(
                "/vel",
                H5::PredType::NATIVE_DOUBLE,
                space,
                prop
            );
        }

        // /screen dataset: [frame, y, x]
        {
            hsize_t dims[3] = {
                0,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            hsize_t maxdims[3] = {
                H5S_UNLIMITED,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            H5::DataSpace space(3, dims, maxdims);
            H5::DSetCreatPropList prop;

            hsize_t chunk[3] = {
                1,
                screenChunkY,
                screenChunkX
            };

            prop.setChunk(3, chunk);

            screen_ = file_.createDataSet(
                "/screen",
                H5::PredType::NATIVE_ULLONG,
                space,
                prop
            );
        }

        // /step dataset: [frame]
        {
            hsize_t dims[1] = {0};
            hsize_t maxdims[1] = {H5S_UNLIMITED};

            H5::DataSpace space(1, dims, maxdims);
            H5::DSetCreatPropList prop;

            hsize_t chunk[1] = {
                static_cast<hsize_t>(chunkFrames_)
            };

            prop.setChunk(1, chunk);

            step_ = file_.createDataSet(
                "/step",
                H5::PredType::NATIVE_LLONG,
                space,
                prop
            );
        }

        extendDatasets(capacity_);
    }

    ~H5StreamWriter() noexcept {
        try {
            close();
        } catch (...) {
            // Never throw from destructor.
        }
    }

    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;

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
            Pbuf_[2 * i] = x[i];
            Pbuf_[2 * i + 1] = y[i];

            Vbuf_[2 * i] = vx[i];
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
        file_.close();

        closed_ = true;
    }

private:
    void extendDatasets(hsize_t newSize) {
        {
            std::array<hsize_t, 3> size = {
                newSize,
                static_cast<hsize_t>(np_),
                2
            };

            pos_.extend(size.data());
            vel_.extend(size.data());
        }

        {
            std::array<hsize_t, 3> size = {
                newSize,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            screen_.extend(size.data());
        }

        {
            std::array<hsize_t, 1> size = {
                newSize
            };

            step_.extend(size.data());
        }
    }

    void shrinkToFit() {
        extendDatasets(currentFrame_);
    }

    static void writeDoubleFrame(
        H5::DataSet& dataset,
        const double* data,
        hsize_t frame,
        hsize_t dim1,
        hsize_t dim2
    ) {
        H5::DataSpace filespace = dataset.getSpace();

        hsize_t start[3] = {
            frame,
            0,
            0
        };

        hsize_t count[3] = {
            1,
            dim1,
            dim2
        };

        filespace.selectHyperslab(H5S_SELECT_SET, count, start);

        H5::DataSpace memspace(3, count);

        dataset.write(
            data,
            H5::PredType::NATIVE_DOUBLE,
            memspace,
            filespace
        );
    }

    static void writeScreenFrame(
        H5::DataSet& dataset,
        const unsigned long long* data,
        hsize_t frame,
        hsize_t ny,
        hsize_t nx
    ) {
        H5::DataSpace filespace = dataset.getSpace();

        hsize_t start[3] = {
            frame,
            0,
            0
        };

        hsize_t count[3] = {
            1,
            ny,
            nx
        };

        filespace.selectHyperslab(H5S_SELECT_SET, count, start);

        H5::DataSpace memspace(3, count);

        dataset.write(
            data,
            H5::PredType::NATIVE_ULLONG,
            memspace,
            filespace
        );
    }

    static void writeStep(
        H5::DataSet& dataset,
        long long stepValue,
        hsize_t frame
    ) {
        H5::DataSpace filespace = dataset.getSpace();

        hsize_t start[1] = {
            frame
        };

        hsize_t count[1] = {
            1
        };

        filespace.selectHyperslab(H5S_SELECT_SET, count, start);

        H5::DataSpace memspace(1, count);

        dataset.write(
            &stepValue,
            H5::PredType::NATIVE_LLONG,
            memspace,
            filespace
        );
    }

    H5::H5File file_;

    H5::DataSet pos_;
    H5::DataSet vel_;
    H5::DataSet screen_;
    H5::DataSet step_;

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


// ============================================================
// CUDA LAUNCH HELPERS
// ============================================================

void launchMandelbrot(
    unsigned long long* d_values,
    const Grid& gen,
    std::size_t maxIters
) {
    const dim3 block(16, 16);

    const dim3 grid(
        static_cast<unsigned>((gen.nx + block.x - 1) / block.x),
        static_cast<unsigned>((gen.ny + block.y - 1) / block.y)
    );

    mandelbrotKernel<<<grid, block>>>(
        d_values,
        gen.nx,
        gen.ny,
        gen.xs,
        gen.xe,
        gen.ys,
        gen.ye,
        maxIters
    );

    CUDA_CHECK_LAST();
}


void launchComputeForces(
    const double* d_x,
    const double* d_y,
    const double* d_w,
    double* d_fx,
    double* d_fy,
    std::size_t N
) {
    constexpr int threads = BLOCK_SIZE;

    const unsigned blocks =
        static_cast<unsigned>((N + threads - 1) / threads);

    computeForcesKernelTiled<BLOCK_SIZE><<<blocks, threads>>>(
        d_x,
        d_y,
        d_w,
        d_fx,
        d_fy,
        N
    );

    CUDA_CHECK_LAST();
}


void launchHalfKickDrift(
    double* d_x,
    double* d_y,
    double* d_vx,
    double* d_vy,
    const double* d_fx,
    const double* d_fy,
    const double* d_w,
    double dt,
    std::size_t N
) {
    constexpr int threads = BLOCK_SIZE;

    const unsigned blocks =
        static_cast<unsigned>((N + threads - 1) / threads);

    halfKickDriftKernel<<<blocks, threads>>>(
        d_x,
        d_y,
        d_vx,
        d_vy,
        d_fx,
        d_fy,
        d_w,
        dt,
        N
    );

    CUDA_CHECK_LAST();
}


void launchHalfKick(
    double* d_vx,
    double* d_vy,
    const double* d_fx_new,
    const double* d_fy_new,
    const double* d_w,
    double dt,
    std::size_t N
) {
    constexpr int threads = BLOCK_SIZE;

    const unsigned blocks =
        static_cast<unsigned>((N + threads - 1) / threads);

    halfKickKernel<<<blocks, threads>>>(
        d_vx,
        d_vy,
        d_fx_new,
        d_fy_new,
        d_w,
        dt,
        N
    );

    CUDA_CHECK_LAST();
}


void launchBuildScreen(
    unsigned long long* d_screen,
    const double* d_x,
    const double* d_y,
    const double* d_w,
    const Grid& screen,
    double wmin,
    double wr,
    std::size_t N
) {
    constexpr int threads = BLOCK_SIZE;

    const unsigned blocks =
        static_cast<unsigned>((N + threads - 1) / threads);

    buildScreenKernel<<<blocks, threads>>>(
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
    H5::Exception::dontPrint();

    try {
        const std::string inputFile =
            (argc > 1) ? argv[1] : "Particles.inp";

        const std::string outputFile =
            (argc > 2) ? argv[2] : "particles.h5";

        Grid gen;
        Grid screen;

        Config cfg = readInput(inputFile, gen, screen);

        if (argc > 3) {
            const long long outEvery =
                parseStrictLongLong(argv[3], "outputEvery");

            if (outEvery <= 0) {
                throw std::runtime_error("outputEvery must be > 0");
            }

            cfg.outputEvery = static_cast<std::size_t>(outEvery);
        }

        const std::size_t genSize = safeGridSize(gen.nx, gen.ny);
        const std::size_t screenSize = safeGridSize(screen.nx, screen.ny);

        const std::size_t genBytes =
            safeMul(genSize, sizeof(unsigned long long), "generating field bytes");

        const std::size_t screenBytes =
            safeMul(screenSize, sizeof(unsigned long long), "screen bytes");

        std::cout << "Input file:                 " << inputFile << "\n";
        std::cout << "HDF5 output:                " << outputFile << "\n";
        std::cout << "Generating grid:            " << gen.nx << " x " << gen.ny << "\n";
        std::cout << "Screen grid:                " << screen.nx << " x " << screen.ny << "\n";
        std::cout << "Max iterations:             " << cfg.maxIters << "\n";
        std::cout << "Steps:                      " << cfg.maxSteps << "\n";
        std::cout << "Output every:               " << cfg.outputEvery << "\n";

        // ----------------------------------------------------
        // 1. Generate Mandelbrot field on GPU
        // ----------------------------------------------------
        DeviceBuffer<unsigned long long> d_gen_values(genSize);

        CudaEvent mandelStart;
        CudaEvent mandelStop;

        CUDA_CHECK(cudaEventRecord(mandelStart.get()));

        launchMandelbrot(
            d_gen_values.get(),
            gen,
            cfg.maxIters
        );

        CUDA_CHECK(cudaEventRecord(mandelStop.get()));
        CUDA_CHECK(cudaEventSynchronize(mandelStop.get()));

        const float mandelMs =
            elapsedMilliseconds(mandelStart, mandelStop);

        CUDA_CHECK(cudaMemcpy(
            gen.values.data(),
            d_gen_values.get(),
            genBytes,
            cudaMemcpyDeviceToHost
        ));

        // ----------------------------------------------------
        // 2. Generate particles on host
        // ----------------------------------------------------
        auto particleStartWall = std::chrono::steady_clock::now();

        Particles P = generateParticles(gen, screen);

        auto particleStopWall = std::chrono::steady_clock::now();

        const double particleGenerationWall =
            std::chrono::duration<double>(
                particleStopWall - particleStartWall
            ).count();

        const std::size_t N = P.n;

        const std::size_t particleBytes =
            safeMul(N, sizeof(double), "particle bytes");

        std::cout << "Particles:                  " << N << "\n";

        // ----------------------------------------------------
        // 3. Allocate device buffers
        // ----------------------------------------------------
        DeviceBuffer<double> d_x(N);
        DeviceBuffer<double> d_y(N);
        DeviceBuffer<double> d_vx(N);
        DeviceBuffer<double> d_vy(N);
        DeviceBuffer<double> d_w(N);

        DeviceBuffer<double> d_fx(N);
        DeviceBuffer<double> d_fy(N);
        DeviceBuffer<double> d_fx_new(N);
        DeviceBuffer<double> d_fy_new(N);

        DeviceBuffer<unsigned long long> d_screen(screenSize);

        // ----------------------------------------------------
        // 4. Allocate pinned host staging buffers
        // ----------------------------------------------------
        PinnedHostBuffer<double> h_x(N);
        PinnedHostBuffer<double> h_y(N);
        PinnedHostBuffer<double> h_vx(N);
        PinnedHostBuffer<double> h_vy(N);
        PinnedHostBuffer<unsigned long long> h_screen(screenSize);

        // ----------------------------------------------------
        // 5. Copy initial particle state to GPU
        // ----------------------------------------------------
        CUDA_CHECK(cudaMemcpy(d_x.get(), P.x.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y.get(), P.y.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx.get(), P.vx.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy.get(), P.vy.data(), particleBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w.get(), P.w.data(), particleBytes, cudaMemcpyHostToDevice));

        const auto [wminIt, wmaxIt] =
            std::minmax_element(P.w.begin(), P.w.end());

        const double wmin = *wminIt;
        const double wmax = *wmaxIt;
        const double wr = std::max(wmax - wmin, 1.0);

        // ----------------------------------------------------
        // 6. Initial force computation
        // ----------------------------------------------------
        CudaEvent initForceStart;
        CudaEvent initForceStop;

        CUDA_CHECK(cudaEventRecord(initForceStart.get()));

        launchComputeForces(
            d_x.get(),
            d_y.get(),
            d_w.get(),
            d_fx.get(),
            d_fy.get(),
            N
        );

        CUDA_CHECK(cudaEventRecord(initForceStop.get()));
        CUDA_CHECK(cudaEventSynchronize(initForceStop.get()));

        const float initForceMs =
            elapsedMilliseconds(initForceStart, initForceStop);

        // ----------------------------------------------------
        // 7. Simulation loop
        // ----------------------------------------------------
        H5StreamWriter h5(
            outputFile,
            N,
            screen.nx,
            screen.ny
        );

        CudaEvent loopGpuStart;
        CudaEvent loopGpuStop;

        auto loopWallStart = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaEventRecord(loopGpuStart.get()));

        std::size_t outputFrames = 0;

        // Guard against accidentally writing the same step twice.
        bool hasLastWrittenStep = false;
        std::size_t lastWrittenStep = 0;

        auto writeOutputFrame = [&](std::size_t step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            CUDA_CHECK(cudaMemset(
                d_screen.get(),
                0,
                screenBytes
            ));

            launchBuildScreen(
                d_screen.get(),
                d_x.get(),
                d_y.get(),
                d_w.get(),
                screen,
                wmin,
                wr,
                N
            );

            CUDA_CHECK(cudaMemcpy(
                h_screen.data(),
                d_screen.get(),
                screenBytes,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaMemcpy(
                h_x.data(),
                d_x.get(),
                particleBytes,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaMemcpy(
                h_y.data(),
                d_y.get(),
                particleBytes,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaMemcpy(
                h_vx.data(),
                d_vx.get(),
                particleBytes,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaMemcpy(
                h_vy.data(),
                d_vy.get(),
                particleBytes,
                cudaMemcpyDeviceToHost
            ));

            h5.writeFrame(
                step,
                h_x.data(),
                h_y.data(),
                h_vx.data(),
                h_vy.data(),
                h_screen.data()
            );

            hasLastWrittenStep = true;
            lastWrittenStep = step;

            ++outputFrames;
        };

        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            // Save the state before advancing, as in your original CUDA logic.
            if ((step % cfg.outputEvery) == 0) {
                writeOutputFrame(step);
            }

            launchHalfKickDrift(
                d_x.get(),
                d_y.get(),
                d_vx.get(),
                d_vy.get(),
                d_fx.get(),
                d_fy.get(),
                d_w.get(),
                cfg.dt,
                N
            );

            launchComputeForces(
                d_x.get(),
                d_y.get(),
                d_w.get(),
                d_fx_new.get(),
                d_fy_new.get(),
                N
            );

            launchHalfKick(
                d_vx.get(),
                d_vy.get(),
                d_fx_new.get(),
                d_fy_new.get(),
                d_w.get(),
                cfg.dt,
                N
            );

            swap(d_fx, d_fx_new);
            swap(d_fy, d_fy_new);
        }

        // Always save the final state after cfg.maxSteps integrations.
        // The duplicate guard inside writeOutputFrame prevents accidental duplication.
        writeOutputFrame(cfg.maxSteps);

        CUDA_CHECK(cudaEventRecord(loopGpuStop.get()));
        CUDA_CHECK(cudaEventSynchronize(loopGpuStop.get()));

        CUDA_CHECK(cudaDeviceSynchronize());

        h5.close();

        auto loopWallStop = std::chrono::steady_clock::now();
        const float loopGpuMs =
            elapsedMilliseconds(loopGpuStart, loopGpuStop);

        const double loopWallSeconds =
            std::chrono::duration<double>(
                loopWallStop - loopWallStart
            ).count();

        // ----------------------------------------------------
        // 8. Reporting
        // ----------------------------------------------------
        const double interactions =
            static_cast<double>(N)
            * static_cast<double>(N - 1)
            * static_cast<double>(cfg.maxSteps);

        const double gigaInteractions =
            interactions / 1.0e9;

        const double loopGpuSeconds =
            static_cast<double>(loopGpuMs) / 1000.0;

        std::cout << "Simulation completed successfully.\n";
        std::cout << "Output frames:              " << outputFrames << "\n";
        std::cout << "Mandelbrot GPU time:        " << (mandelMs / 1000.0f) << " s\n";
        std::cout << "Particle generation wall:   " << particleGenerationWall << " s\n";
        std::cout << "Initial force GPU time:     " << (initForceMs / 1000.0f) << " s\n";
        std::cout << "Timed GPU pipeline time:    " << loopGpuSeconds << " s\n";
        std::cout << "Wall time including HDF5:   " << loopWallSeconds << " s\n";

        if (cfg.maxSteps > 0 && loopGpuSeconds > 0.0) {
            std::cout << "GPU pipeline performance:   "
                      << gigaInteractions / loopGpuSeconds
                      << " GInteractions/s\n";
        }

        if (cfg.maxSteps > 0 && loopWallSeconds > 0.0) {
            std::cout << "End-to-end performance:     "
                      << gigaInteractions / loopWallSeconds
                      << " GInteractions/s\n";
        }

        return EXIT_SUCCESS;
    }
    catch (const H5::Exception& e) {
        std::cerr << "HDF5 ERROR: " << e.getDetailMsg() << "\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "ERROR: unknown exception\n";
        return EXIT_FAILURE;
    }
}
