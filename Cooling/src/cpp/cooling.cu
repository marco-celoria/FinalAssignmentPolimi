#include <H5Cpp.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
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

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::ostringstream oss__;                                            \
            oss__ << "CUDA ERROR: " << cudaGetErrorString(err__)                 \
                  << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(oss__.str());                               \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_LAST()                                                        \
    do {                                                                         \
        cudaError_t err__ = cudaGetLastError();                                  \
        if (err__ != cudaSuccess) {                                              \
            std::ostringstream oss__;                                            \
            oss__ << "CUDA KERNEL LAUNCH ERROR: "                                \
                  << cudaGetErrorString(err__)                                   \
                  << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(oss__.str());                               \
        }                                                                        \
    } while (0)


// ============================================================
// RAII HELPERS
// ============================================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
        : ptr_(nullptr), count_(count)
    {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&ptr_),
                count_ * sizeof(T)
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
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    friend void swap(DeviceBuffer& a, DeviceBuffer& b) noexcept {
        std::swap(a.ptr_, b.ptr_);
        std::swap(a.count_, b.count_);
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }

    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};


template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(std::size_t count)
        : ptr_(nullptr), count_(count)
    {
        if (count_ > 0) {
            CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&ptr_),
                count_ * sizeof(T),
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
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }

    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};


class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&ev_));
    }

    ~CudaEvent() {
        if (ev_) {
            cudaEventDestroy(ev_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept { return ev_; }

private:
    cudaEvent_t ev_{nullptr};
};


// ============================================================
// DATA STRUCTURES
// ============================================================

struct MeasuredPoint {
    double x{};
    double y{};
    double v{};
};

struct Config {
    std::size_t nx{};
    std::size_t ny{};

    double Sreal{};
    double Simag{};
    double Dreal{};
    double Dimag{};

    int maxIters{};
    int steps{};
    int outputEvery{1};

    std::vector<MeasuredPoint> measured;
};

struct DomainMap {
    double x0{};
    double y0{};
    double dx{};
    double dy{};
};

struct CoolingCoeffs {
    double dd{};
    double hx{};
    double hy{};
    double dgx{};
    double dgy{};
    double CX{};
    double CY{};
};

struct Stats {
    double minv{};
    double mean{};
    double maxv{};
    double stddev{};
};


// ============================================================
// UTILITY
// ============================================================

inline std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) noexcept {
    return i + j * nx;
}

__host__ __device__
inline int idx2D_dev(int i, int j, int nx) noexcept {
    return i + j * nx;
}

std::size_t safeGridSize(std::size_t nx, std::size_t ny) {
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument("Grid dimensions must be > 0");
    }

    if (nx > std::numeric_limits<std::size_t>::max() / ny) {
        throw std::overflow_error("Grid size overflow: nx * ny exceeds size_t range");
    }

    return nx * ny;
}

int parseStrictInt(const std::string& s, const std::string& what) {
    int v = 0;

    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);

    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        throw std::runtime_error("Invalid " + what + ": '" + s + "'");
    }

    return v;
}

void validateCudaLaunchConfig(dim3 block2d, int block1d) {
    const unsigned threads2d = block2d.x * block2d.y * block2d.z;

    if (threads2d == 0 || threads2d > 1024) {
        throw std::runtime_error("Invalid 2D CUDA block size");
    }

    if (block1d <= 0 || block1d > 1024) {
        throw std::runtime_error("Invalid 1D CUDA block size");
    }
}


// ============================================================
// INPUT PARSER
// ============================================================

Config readInput(const std::string& fname) {
    std::ifstream in(fname);

    if (!in) {
        throw std::runtime_error("Cannot open input file: " + fname);
    }

    std::vector<std::string> tokens;
    std::string line;

    while (std::getline(in, line)) {
        const auto commentPos = line.find('#');

        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }

        std::istringstream iss(line);
        std::string tok;

        while (iss >> tok) {
            tokens.push_back(tok);
        }
    }

    if (tokens.empty()) {
        throw std::runtime_error(
            "Input file is empty or contains no numeric tokens: " + fname
        );
    }

    std::size_t pos = 0;

    auto nextInt = [&]() -> int {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing integer token");
        }

        const std::string& s = tokens.at(pos++);

        try {
            std::size_t used = 0;
            int v = std::stoi(s, &used);

            if (used != s.size()) {
                throw std::runtime_error("");
            }

            return v;
        } catch (...) {
            throw std::runtime_error(
                "Malformed input: invalid integer token '" + s + "'"
            );
        }
    };

    auto nextDouble = [&]() -> double {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing floating-point token");
        }

        const std::string& s = tokens.at(pos++);

        try {
            std::size_t used = 0;
            double v = std::stod(s, &used);

            if (used != s.size()) {
                throw std::runtime_error("");
            }

            return v;
        } catch (...) {
            throw std::runtime_error(
                "Malformed input: invalid floating-point token '" + s + "'"
            );
        }
    };

    Config cfg{};

    const int rawNx = nextInt();
    const int rawNy = nextInt();

    if (rawNx < 3 || rawNy < 3) {
        throw std::runtime_error("Grid dimensions must be at least 3 x 3");
    }

    cfg.nx = static_cast<std::size_t>(rawNx);
    cfg.ny = static_cast<std::size_t>(rawNy);

    const int nMeasured = nextInt();

    if (nMeasured < 0) {
        throw std::runtime_error("Number of measured points cannot be negative");
    }

    cfg.measured.resize(static_cast<std::size_t>(nMeasured));

    for (int i = 0; i < nMeasured; ++i) {
        auto& m = cfg.measured[static_cast<std::size_t>(i)];
        m.x = nextDouble();
        m.y = nextDouble();
        m.v = nextDouble();
    }

    cfg.Sreal = nextDouble();
    cfg.Simag = nextDouble();
    cfg.Dreal = nextDouble();
    cfg.Dimag = nextDouble();

    cfg.maxIters = nextInt();
    cfg.steps = nextInt();

    if (cfg.maxIters <= 0) {
        throw std::runtime_error("maxIters must be > 0");
    }

    if (cfg.steps < 0) {
        throw std::runtime_error("steps must be >= 0");
    }

    if (cfg.Dreal <= 0.0 || cfg.Dimag <= 0.0) {
        throw std::runtime_error("Domain extents Dreal and Dimag must be > 0");
    }

    if (pos != tokens.size()) {
        throw std::runtime_error("Malformed input: unexpected extra tokens at end of file");
    }

    cfg.outputEvery = 1;

    return cfg;
}


// ============================================================
// PHYSICS HELPERS
// ============================================================

DomainMap buildDomainMap(const Config& cfg) {
    DomainMap map{};

    map.x0 = cfg.Sreal;
    map.y0 = cfg.Simag;
    map.dx = cfg.Dreal / static_cast<double>(cfg.nx - 1);
    map.dy = cfg.Dimag / static_cast<double>(cfg.ny - 1);

    return map;
}

inline double analyticalField(double x, double y) noexcept {
    return (x * x * x + y * y * y) / 6.0;
}

double computeDiscrepancy(const Config& cfg) {
    if (cfg.measured.empty()) {
        return 0.0;
    }

    long double sum = 0.0L;

    for (const auto& m : cfg.measured) {
        sum += static_cast<long double>(m.v - analyticalField(m.x, m.y));
    }

    return static_cast<double>(
        sum / static_cast<long double>(cfg.measured.size())
    );
}

CoolingCoeffs buildCoolingCoeffs(double dx, double dy, double dd = 100.0) {
    if (dx <= 0.0 || dy <= 0.0) {
        throw std::invalid_argument("buildCoolingCoeffs: dx and dy must be > 0");
    }

    if (dd <= 0.0) {
        throw std::invalid_argument("buildCoolingCoeffs: dd must be > 0");
    }

    CoolingCoeffs c{};

    c.dd = dd;
    c.hx = dx;
    c.hy = dy;

    const double hx2 = c.hx * c.hx;
    const double hy2 = c.hy * c.hy;

    c.dgx = -2.0 * (1.0 + c.dd * c.hx / (hx2 + c.dd));
    c.dgy = -2.0 * (1.0 + c.dd * c.hy / (hy2 + c.dd));

    c.CX = (c.hx + c.dd * std::exp(c.hx)) / (15.0 * c.dd + c.hx);
    c.CY = (c.hy + c.dd * std::exp(c.hy)) / (15.0 * c.dd + c.hy);

    return c;
}


// ============================================================
// HDF5 WRITER
// ============================================================

class H5Writer {
public:
    H5Writer(
        const std::string& fname,
        std::size_t nx,
        std::size_t ny,
        std::size_t batch = 32,
        std::size_t tileY = 256,
        std::size_t tileX = 256
    )
        : file_(fname, H5F_ACC_TRUNC),
          nx_(nx),
          ny_(ny),
          batch_(batch),
          frame_(0),
          capacity_(batch),
          closed_(false)
    {
        if (nx_ == 0 || ny_ == 0) {
            throw std::invalid_argument("H5Writer: nx and ny must be > 0");
        }

        if (batch_ == 0) {
            throw std::invalid_argument("H5Writer: batch must be > 0");
        }

        if (tileY == 0 || tileX == 0) {
            throw std::invalid_argument("H5Writer: tile sizes must be > 0");
        }

        const hsize_t chunkY =
            static_cast<hsize_t>(std::min<std::size_t>(ny_, tileY));

        const hsize_t chunkX =
            static_cast<hsize_t>(std::min<std::size_t>(nx_, tileX));

        // /field dataset: [frame, y, x]
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

            hsize_t chunks[3] = {
                1,
                chunkY,
                chunkX
            };

            prop.setChunk(3, chunks);

            field_ = file_.createDataSet(
                "/field",
                H5::PredType::NATIVE_DOUBLE,
                space,
                prop
            );
        }

        // /step dataset
        {
            hsize_t dims[1] = {0};
            hsize_t maxdims[1] = {H5S_UNLIMITED};

            H5::DataSpace space(1, dims, maxdims);
            H5::DSetCreatPropList prop;

            hsize_t chunks[1] = {static_cast<hsize_t>(batch_)};
            prop.setChunk(1, chunks);

            step_ = file_.createDataSet(
                "/step",
                H5::PredType::NATIVE_INT,
                space,
                prop
            );
        }

        extend(capacity_);
    }

    ~H5Writer() {
        try {
            close();
        } catch (...) {
            // Never throw from destructor.
        }
    }

    H5Writer(const H5Writer&) = delete;
    H5Writer& operator=(const H5Writer&) = delete;

    void write(int stepNumber, const double* field, std::size_t count) {
        if (closed_) {
            throw std::runtime_error("H5Writer: write() called after close()");
        }

        if (!field) {
            throw std::runtime_error("H5Writer: null field pointer");
        }

        if (count != safeGridSize(nx_, ny_)) {
            throw std::runtime_error("H5Writer: field size mismatch");
        }

        if (frame_ >= capacity_) {
            capacity_ += batch_;
            extend(capacity_);
        }

        // Write /field[frame,:,:]
        {
            H5::DataSpace filespace = field_.getSpace();

            hsize_t start[3] = {
                static_cast<hsize_t>(frame_),
                0,
                0
            };

            hsize_t count3[3] = {
                1,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count3, start);

            H5::DataSpace memspace(3, count3);

            field_.write(
                field,
                H5::PredType::NATIVE_DOUBLE,
                memspace,
                filespace
            );
        }

        // Write /step[frame]
        {
            H5::DataSpace filespace = step_.getSpace();

            hsize_t start[1] = {
                static_cast<hsize_t>(frame_)
            };

            hsize_t count1[1] = {1};

            filespace.selectHyperslab(H5S_SELECT_SET, count1, start);

            H5::DataSpace memspace(1, count1);

            int value = stepNumber;

            step_.write(
                &value,
                H5::PredType::NATIVE_INT,
                memspace,
                filespace
            );
        }

        ++frame_;
    }

    void close() {
        if (closed_) {
            return;
        }

        closed_ = true;

        if (frame_ != capacity_) {
            extend(frame_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);

        field_.close();
        step_.close();
        file_.close();
    }

private:
    void extend(std::size_t n) {
        {
            hsize_t dims[3] = {
                static_cast<hsize_t>(n),
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            field_.extend(dims);
        }

        {
            hsize_t dims[1] = {
                static_cast<hsize_t>(n)
            };

            step_.extend(dims);
        }
    }

    H5::H5File file_;
    H5::DataSet field_;
    H5::DataSet step_;

    std::size_t nx_{};
    std::size_t ny_{};
    std::size_t batch_{};
    std::size_t frame_{};
    std::size_t capacity_{};

    bool closed_{false};
};


// ============================================================
// CUDA KERNELS
// ============================================================

__global__
void computeWeightKernel(
    int* weight,
    int nx,
    int ny,
    double x0,
    double y0,
    double dx,
    double dy,
    int maxIters
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    const int p = idx2D_dev(i, j, nx);

    const double ca = x0 + dx * static_cast<double>(i);
    const double cb = y0 + dy * static_cast<double>(j);

    double za = 0.0;
    double zb = 0.0;

    int it = 0;

    for (; it < maxIters; ++it) {
        if (za * za + zb * zb > 4.0) {
            break;
        }

        const double tmp = za * za - zb * zb + ca;
        zb = 2.0 * za * zb + cb;
        za = tmp;
    }

    weight[p] = it;
}


__global__
void initializeFieldKernel(
    double* u,
    const int* weight,
    int nx,
    int ny,
    double x0,
    double y0,
    double dx,
    double dy,
    double discrepancy,
    int wmin,
    int wmax
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    const int p = idx2D_dev(i, j, nx);

    const double x = x0 + dx * static_cast<double>(i);
    const double y = y0 + dy * static_cast<double>(j);

    const double denom =
        (wmax > wmin) ? static_cast<double>(wmax - wmin) : 1.0;

    const double F = (x * x * x + y * y * y) / 6.0;
    const double wnorm = static_cast<double>(weight[p] - wmin) / denom;

    u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm;
}


__global__
void updateInteriorKernel(
    const double* __restrict__ u1,
    double* __restrict__ u2,
    int nx,
    int ny,
    double dgx,
    double dgy,
    double CX,
    double CY
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) {
        return;
    }

    const int p = idx2D_dev(i, j, nx);

    u2[p] =
        CX * (
            u1[p - 1]
            + u1[p + 1]
            + (dgx + 0.5 / CX) * u1[p]
        )
        + CY * (
            u1[p - nx]
            + u1[p + nx]
            + (dgy + 0.5 / CY) * u1[p]
        );
}


// Left/right edges are updated first.
// Top/bottom then copies from already-updated edge-adjacent values.
__global__
void applyBoundaryLRKernel(double* u, int nx, int ny) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < 1 || j >= ny - 1) {
        return;
    }

    const int row = j * nx;

    u[row] = u[row + 1];
    u[row + nx - 1] = u[row + nx - 2];
}


__global__
void applyBoundaryTBKernel(double* u, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx) {
        return;
    }

    u[i] = u[nx + i];
    u[(ny - 1) * nx + i] = u[(ny - 2) * nx + i];
}


// ============================================================
// CUDA LAUNCH HELPERS
// ============================================================

void launchComputeWeight(
    int* d_weight,
    int nx,
    int ny,
    double x0,
    double y0,
    double dx,
    double dy,
    int maxIters,
    dim3 block2d
) {
    const dim3 grid2d(
        static_cast<unsigned>((nx + static_cast<int>(block2d.x) - 1)
                              / static_cast<int>(block2d.x)),
        static_cast<unsigned>((ny + static_cast<int>(block2d.y) - 1)
                              / static_cast<int>(block2d.y))
    );

    computeWeightKernel<<<grid2d, block2d>>>(
        d_weight,
        nx,
        ny,
        x0,
        y0,
        dx,
        dy,
        maxIters
    );

    CUDA_CHECK_LAST();
}


void launchInitializeField(
    double* d_u,
    const int* d_weight,
    int nx,
    int ny,
    double x0,
    double y0,
    double dx,
    double dy,
    double discrepancy,
    int wmin,
    int wmax,
    dim3 block2d
) {
    const dim3 grid2d(
        static_cast<unsigned>((nx + static_cast<int>(block2d.x) - 1)
                              / static_cast<int>(block2d.x)),
        static_cast<unsigned>((ny + static_cast<int>(block2d.y) - 1)
                              / static_cast<int>(block2d.y))
    );

    initializeFieldKernel<<<grid2d, block2d>>>(
        d_u,
        d_weight,
        nx,
        ny,
        x0,
        y0,
        dx,
        dy,
        discrepancy,
        wmin,
        wmax
    );

    CUDA_CHECK_LAST();
}


void launchUpdateField(
    const double* d_u1,
    double* d_u2,
    int nx,
    int ny,
    double dgx,
    double dgy,
    double CX,
    double CY,
    dim3 block2d,
    int block1d
) {
    const dim3 grid2d(
        static_cast<unsigned>((nx + static_cast<int>(block2d.x) - 1)
                              / static_cast<int>(block2d.x)),
        static_cast<unsigned>((ny + static_cast<int>(block2d.y) - 1)
                              / static_cast<int>(block2d.y))
    );

    updateInteriorKernel<<<grid2d, block2d>>>(
        d_u1,
        d_u2,
        nx,
        ny,
        dgx,
        dgy,
        CX,
        CY
    );

    CUDA_CHECK_LAST();

    const int gridY = (ny + block1d - 1) / block1d;

    applyBoundaryLRKernel<<<gridY, block1d>>>(
        d_u2,
        nx,
        ny
    );

    CUDA_CHECK_LAST();

    const int gridX = (nx + block1d - 1) / block1d;

    applyBoundaryTBKernel<<<gridX, block1d>>>(
        d_u2,
        nx,
        ny
    );

    CUDA_CHECK_LAST();
}


// ============================================================
// STATISTICS
// ============================================================

Stats computeStatsCPUReference(const double* u, std::size_t N) {
    if (!u || N == 0) {
        throw std::runtime_error("computeStatsCPUReference: empty field");
    }

    const auto minmax = std::minmax_element(u, u + N);

    long double sum = 0.0L;

    for (std::size_t i = 0; i < N; ++i) {
        sum += static_cast<long double>(u[i]);
    }

    const long double mean = sum / static_cast<long double>(N);

    long double ssd = 0.0L;

    for (std::size_t i = 0; i < N; ++i) {
        const long double d = static_cast<long double>(u[i]) - mean;
        ssd += d * d;
    }

    Stats s{};

    s.minv = *minmax.first;
    s.maxv = *minmax.second;
    s.mean = static_cast<double>(mean);
    s.stddev = static_cast<double>(
        std::sqrt(ssd / static_cast<long double>(N))
    );

    return s;
}


struct SquareDeviationFunctor {
    double mean;

    __host__ __device__
    explicit SquareDeviationFunctor(double m)
        : mean(m)
    {}

    __host__ __device__
    double operator()(const double& x) const {
        const double delta = x - mean;
        return delta * delta;
    }
};


Stats computeStatsGPU(double* d_ptr, std::size_t N) {
    if (!d_ptr || N == 0) {
        throw std::runtime_error("computeStatsGPU: empty field");
    }

    thrust::device_ptr<double> t_ptr(d_ptr);

    Stats s{};

    auto minmax_pair = thrust::minmax_element(t_ptr, t_ptr + N);

    s.minv = *minmax_pair.first;
    s.maxv = *minmax_pair.second;

    const double sum = thrust::reduce(
        t_ptr,
        t_ptr + N,
        0.0,
        thrust::plus<double>()
    );

    s.mean = sum / static_cast<double>(N);

    const double sum_squared_diff = thrust::transform_reduce(
        t_ptr,
        t_ptr + N,
        SquareDeviationFunctor(s.mean),
        0.0,
        thrust::plus<double>()
    );

    s.stddev = std::sqrt(sum_squared_diff / static_cast<double>(N));

    return s;
}


void writeStatsHeader(std::ostream& os) {
    os << "Step;Min;Mean;Max;Std_dev\n";
}


void writeStatsLine(std::ostream& os, int step, const Stats& s) {
    os << step << ';'
       << std::setprecision(15) << s.minv << ';'
       << std::setprecision(15) << s.mean << ';'
       << std::setprecision(15) << s.maxv << ';'
       << std::setprecision(15) << s.stddev << '\n';
}


// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    H5::Exception::dontPrint();

    try {
        const std::string inputFile = (argc > 1) ? argv[1] : "Cooling.inp";
        const std::string h5File = (argc > 2) ? argv[2] : "cooling.h5";
        const std::string csvFile = (argc > 3) ? argv[3] : "Statistics.csv";

        Config cfg = readInput(inputFile);

        if (argc > 4) {
            cfg.outputEvery = parseStrictInt(argv[4], "outputEvery");
        }

        if (cfg.outputEvery <= 0) {
            throw std::invalid_argument("outputEvery must be > 0");
        }

        const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

        if (cfg.nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            cfg.ny > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            N > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "Grid exceeds 32-bit CUDA indexing range used by this implementation"
            );
        }

        const int nx_i = static_cast<int>(cfg.nx);
        const int ny_i = static_cast<int>(cfg.ny);

        const DomainMap map = buildDomainMap(cfg);
        const CoolingCoeffs cooling = buildCoolingCoeffs(map.dx, map.dy, 100.0);
        const double discrepancy = computeDiscrepancy(cfg);

        const dim3 block2d(16, 16);
        constexpr int block1d = 256;

        validateCudaLaunchConfig(block2d, block1d);

        DeviceBuffer<int> d_weight(N);
        DeviceBuffer<double> d_uCurr(N);
        DeviceBuffer<double> d_uNext(N);

        PinnedHostBuffer<double> uHost(N);

        std::ofstream csv(csvFile);

        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + csvFile);
        }

        writeStatsHeader(csv);

        std::cout << "Input file:                   " << inputFile << '\n';
        std::cout << "HDF5 output:                  " << h5File << '\n';
        std::cout << "CSV output:                   " << csvFile << '\n';
        std::cout << "Grid:                         " << cfg.nx << " x " << cfg.ny << '\n';
        std::cout << "Measured points:              " << cfg.measured.size() << '\n';
        std::cout << "Max iterations:               " << cfg.maxIters << '\n';
        std::cout << "Time steps:                   " << cfg.steps << '\n';
        std::cout << "Snapshot every:               " << cfg.outputEvery << " step(s)\n";
        std::cout << "CUDA block2d:                 " << block2d.x << " x " << block2d.y << '\n';
        std::cout << "CUDA block1d:                 " << block1d << "\n\n";

        CudaEvent weightStart;
        CudaEvent weightStop;

        CudaEvent initStart;
        CudaEvent initStop;

        CudaEvent dynBatchStart;
        CudaEvent dynBatchStop;

        float tWeightMs = 0.0f;
        float tInitMs = 0.0f;
        float tDynUpdateKernelMsAccum = 0.0f;

        // ----------------------------------------------------
        // Stage 1: weight field
        // ----------------------------------------------------
        CUDA_CHECK(cudaEventRecord(weightStart.get()));

        launchComputeWeight(
            d_weight.get(),
            nx_i,
            ny_i,
            map.x0,
            map.y0,
            map.dx,
            map.dy,
            cfg.maxIters,
            block2d
        );

        CUDA_CHECK(cudaEventRecord(weightStop.get()));
        CUDA_CHECK(cudaEventSynchronize(weightStop.get()));
        CUDA_CHECK(cudaEventElapsedTime(
            &tWeightMs,
            weightStart.get(),
            weightStop.get()
        ));

        thrust::device_ptr<int> w_ptr(d_weight.get());

        auto minmax_pair = thrust::minmax_element(w_ptr, w_ptr + N);

        const int wmin = *minmax_pair.first;
        const int wmax = *minmax_pair.second;

        // ----------------------------------------------------
        // Stage 2: initialization
        // ----------------------------------------------------
        CUDA_CHECK(cudaEventRecord(initStart.get()));

        launchInitializeField(
            d_uCurr.get(),
            d_weight.get(),
            nx_i,
            ny_i,
            map.x0,
            map.y0,
            map.dx,
            map.dy,
            discrepancy,
            wmin,
            wmax,
            block2d
        );

        CUDA_CHECK(cudaEventRecord(initStop.get()));
        CUDA_CHECK(cudaEventSynchronize(initStop.get()));
        CUDA_CHECK(cudaEventElapsedTime(
            &tInitMs,
            initStart.get(),
            initStop.get()
        ));

        // ----------------------------------------------------
        // Stage 3: dynamics + output
        // ----------------------------------------------------
        auto dynWallStart = std::chrono::steady_clock::now();

        H5Writer writer(h5File, cfg.nx, cfg.ny, 32);

        // Step 0 output.
        CUDA_CHECK(cudaMemcpy(
            uHost.data(),
            d_uCurr.get(),
            N * sizeof(double),
            cudaMemcpyDeviceToHost
        ));

        writer.write(0, uHost.data(), N);

        Stats s0 = computeStatsGPU(d_uCurr.get(), N);
        writeStatsLine(csv, 0, s0);

        if (cfg.steps > 0) {
            CUDA_CHECK(cudaEventRecord(dynBatchStart.get()));
        }

        for (int step = 1; step <= cfg.steps; ++step) {
            launchUpdateField(
                d_uCurr.get(),
                d_uNext.get(),
                nx_i,
                ny_i,
                cooling.dgx,
                cooling.dgy,
                cooling.CX,
                cooling.CY,
                block2d,
                block1d
            );

            swap(d_uCurr, d_uNext);

            if ((step % cfg.outputEvery) == 0 || step == cfg.steps) {
                CUDA_CHECK(cudaEventRecord(dynBatchStop.get()));
                CUDA_CHECK(cudaEventSynchronize(dynBatchStop.get()));

                float batchMs = 0.0f;

                CUDA_CHECK(cudaEventElapsedTime(
                    &batchMs,
                    dynBatchStart.get(),
                    dynBatchStop.get()
                ));

                tDynUpdateKernelMsAccum += batchMs;

                CUDA_CHECK(cudaMemcpy(
                    uHost.data(),
                    d_uCurr.get(),
                    N * sizeof(double),
                    cudaMemcpyDeviceToHost
                ));

                writer.write(step, uHost.data(), N);

                Stats s = computeStatsGPU(d_uCurr.get(), N);
                writeStatsLine(csv, step, s);

                if (step < cfg.steps) {
                    CUDA_CHECK(cudaEventRecord(dynBatchStart.get()));
                }
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        writer.close();

        auto dynWallStop = std::chrono::steady_clock::now();

        const double tDynWall =
            std::chrono::duration<double>(dynWallStop - dynWallStart).count();

        std::cout << "Weight field GPU kernel time:       "
                  << (tWeightMs / 1e3f) << " s\n";

        std::cout << "Init field GPU kernel time:         "
                  << (tInitMs / 1e3f) << " s\n";

        std::cout << "Dynamics update kernels only:       "
                  << (tDynUpdateKernelMsAccum / 1e3f) << " s\n";

        std::cout << "Dynamics loop wall incl. copy/stats/I/O: "
                  << tDynWall << " s\n";

        if (cfg.steps > 0) {
            const double updates =
                static_cast<double>(cfg.nx - 2)
                * static_cast<double>(cfg.ny - 2)
                * static_cast<double>(cfg.steps);

            if (tDynUpdateKernelMsAccum > 0.0f) {
                const double gpuKernelSeconds =
                    static_cast<double>(tDynUpdateKernelMsAccum) / 1e3;

                std::cout << "Performance, update kernels only:   "
                          << updates / gpuKernelSeconds / 1e9
                          << " GLUP/s\n";
            }

            if (tDynWall > 0.0) {
                std::cout << "Performance, end-to-end loop:       "
                          << updates / tDynWall / 1e9
                          << " GLUP/s\n";
            }
        }

        std::cout << "Mean discrepancy:                   "
                  << std::setprecision(15) << discrepancy << '\n';

        std::cout << "\nSimulation completed successfully.\n";

        return 0;
    }
    catch (const H5::Exception& e) {
        std::cerr << "HDF5 ERROR: " << e.getDetailMsg() << '\n';
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << '\n';
        return 1;
    }
    catch (...) {
        std::cerr << "CRITICAL ERROR: Unknown failure\n";
        return 1;
    }
}
