/*
================================================================================
Cooling Field Solver - CUDA Reference Implementation
================================================================================

Instructor/reference CUDA implementation of the serial C++17 baseline.

The numerical model, input format, CSV statistics, and
HDF5 behavior are intentionally kept aligned with the serial baseline.

This version offloads the main computational kernels to CUDA. HDF5 output
remains serial and optional. Snapshot statistics are computed on the host after
copying the requested output frame from device to host. This keeps the CSV
validation format directly comparable with the serial and OpenMP versions.

Official performance mode:

  ./cooling_cuda input_final.in none output_final_cuda.csv 0

or with explicit CUDA device selection:

  ./cooling_cuda --device 0 input_final.in none output_final_cuda.csv 0

Compile without HDF5:

  nvcc -O3 -std=c++17 cooling.cu -o cooling_cuda

Compile with HDF5:

  nvcc -O3 -std=c++17 -DUSE_HDF5 cooling.cu -o cooling_cuda \
      -lhdf5_cpp -lhdf5

Depending on the cluster configuration, students may need an HDF5 compiler
wrapper, modules, or explicit include/library paths.

Input file format, after removing comments beginning with '#':

  gridWidth
  gridHeight
  numberOfMeasuredPoints
  x y value        repeated numberOfMeasuredPoints times
  domainStartX
  domainStartY
  domainWidth
  domainHeight
  maxFractalIterations
  timeSteps
  outputEvery      optional; 0 means final step only

Command line:

  ./cooling_cuda [options] [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]

Options:

  --device N       Select CUDA device, default 0
  --block-x N      CUDA 2D block x dimension, default 16
  --block-y N      CUDA 2D block y dimension, default 16
  --block1d N      CUDA 1D block size for boundary kernels, default 256
  --no-hdf5        Disable HDF5 output
  --help, -h       Print usage

Examples:

  ./cooling_cuda --device 0 input_final.in none Statistics_cuda.csv 0
  ./cooling_cuda --device 0 input_medium.in output.h5 Statistics_cuda.csv 50

================================================================================
*/

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
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

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::ostringstream oss__;                                            \
            oss__ << "CUDA ERROR: " << cudaGetErrorString(err__)                 \
                  << " at " << __FILE__ << ":" << __LINE__;                     \
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
                  << " at " << __FILE__ << ":" << __LINE__;                     \
            throw std::runtime_error(oss__.str());                               \
        }                                                                        \
    } while (0)

struct SamplePoint {
    double x{};
    double y{};
    double value{};
};

struct SimulationConfig {
    std::size_t gridWidth{};
    std::size_t gridHeight{};
    double domainStartX{};
    double domainStartY{};
    double domainWidth{};
    double domainHeight{};
    int maxFractalIterations{};
    int timeSteps{};
    int outputEvery{0};
    std::vector<SamplePoint> measuredPoints;
};

struct GridMapping {
    double x0{};
    double y0{};
    double dx{};
    double dy{};
};

struct UpdateCoefficients {
    double damping{};
    double stepX{};
    double stepY{};
    double coeffX{};
    double coeffY{};
    double laplaceX{};
    double laplaceY{};
};

struct FieldStatistics {
    double minValue{};
    double meanValue{};
    double maxValue{};
    double stdDev{};
    double l2Norm{};
    double checksum{};
};

struct CommandLineOptions {
    std::string inputFile{"input_final.in"};
    std::string h5File{"none"};
    std::string csvFile{"Statistics.csv"};

    bool writeHdf5{false};
    bool overrideOutputEvery{false};
    int outputEvery{0};

    int deviceId{0};
    int blockX{16};
    int blockY{16};
    int block1d{256};
};

struct ScopedTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point start{clock::now()};

    double elapsedSeconds() const {
        return std::chrono::duration<double>(clock::now() - start).count();
    }
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

    cudaEvent_t get() const noexcept {
        return event_;
    }

private:
    cudaEvent_t event_{nullptr};
};

class CudaEventTimer {
public:
    void start() {
        CUDA_CHECK(cudaEventRecord(start_.get()));
        running_ = true;
    }

    double stopSeconds() {
        if (!running_) {
            return 0.0;
        }

        CUDA_CHECK(cudaEventRecord(stop_.get()));
        CUDA_CHECK(cudaEventSynchronize(stop_.get()));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_.get(), stop_.get()));

        running_ = false;
        return static_cast<double>(milliseconds) / 1.0e3;
    }

private:
    CudaEvent start_;
    CudaEvent stop_;
    bool running_{false};
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count)
        : count_(count)
    {
        if (count_ > 0) {
            if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::overflow_error("DeviceBuffer allocation size overflow");
            }

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
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

    T* get() noexcept {
        return ptr_;
    }

    const T* get() const noexcept {
        return ptr_;
    }

    std::size_t size() const noexcept {
        return count_;
    }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(std::size_t count)
        : count_(count)
    {
        if (count_ > 0) {
            if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                throw std::overflow_error("PinnedHostBuffer allocation size overflow");
            }

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

    T* data() noexcept {
        return ptr_;
    }

    const T* data() const noexcept {
        return ptr_;
    }

    std::size_t size() const noexcept {
        return count_;
    }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

inline std::size_t linearIndex(std::size_t i, std::size_t j, std::size_t width) noexcept {
    return i + j * width;
}

__host__ __device__
inline int linearIndexDev(int i, int j, int width) noexcept {
    return i + j * width;
}

std::size_t checkedGridSize(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Grid dimensions must be greater than zero");
    }

    if (width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("Grid size overflow");
    }

    return width * height;
}

int parseStrictInt(const std::string& text, const std::string& what) {
    int value = 0;

    const char* begin = text.data();
    const char* end = text.data() + text.size();

    const auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid " + what + ": '" + text + "'");
    }

    return value;
}

int parseStrictPositiveInt(const std::string& text, const std::string& what) {
    const int value = parseStrictInt(text, what);

    if (value <= 0) {
        throw std::runtime_error(what + " must be positive");
    }

    return value;
}

bool beginsWithDoubleDash(const std::string& text) {
    return text.rfind("--", 0) == 0;
}

bool isNoHdf5Token(const std::string& text) {
    return text == "none" || text == "NONE" || text == "-" || text == "--no-hdf5";
}

void ensureParentDirectoryExists(const std::string& fileName) {
    if (fileName.empty() || isNoHdf5Token(fileName)) {
        return;
    }

    const fs::path path(fileName);
    const fs::path parent = path.parent_path();

    if (parent.empty()) {
        return;
    }

    std::error_code ec;

    if (fs::exists(parent, ec)) {
        if (!fs::is_directory(parent, ec)) {
            throw std::runtime_error("Output parent exists but is not a directory: " + parent.string());
        }

        return;
    }

    if (!fs::create_directories(parent, ec) && ec) {
        throw std::runtime_error("Cannot create output directory '" + parent.string() + "': " + ec.message());
    }
}

void selectCudaDevice(int deviceId) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount <= 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }

    if (deviceId < 0 || deviceId >= deviceCount) {
        throw std::runtime_error(
            "Invalid CUDA device id " + std::to_string(deviceId) +
            "; available devices: 0.." + std::to_string(deviceCount - 1)
        );
    }

    CUDA_CHECK(cudaSetDevice(deviceId));
}

void validateCudaLaunchConfig(dim3 block2d, int block1d) {
    const unsigned threads2d = block2d.x * block2d.y * block2d.z;

    if (threads2d == 0 || threads2d > 1024) {
        throw std::runtime_error("Invalid CUDA 2D block size: total threads must be in [1, 1024]");
    }

    if (block2d.x == 0 || block2d.y == 0 || block2d.z == 0) {
        throw std::runtime_error("Invalid CUDA 2D block size: dimensions must be positive");
    }

    if (block1d <= 0 || block1d > 1024) {
        throw std::runtime_error("Invalid CUDA 1D block size: must be in [1, 1024]");
    }
}

SimulationConfig readConfigurationFile(const std::string& fileName) {
    std::ifstream input(fileName);

    if (!input) {
        throw std::runtime_error("Cannot open input file: " + fileName);
    }

    std::vector<std::string> tokens;
    std::string line;

    while (std::getline(input, line)) {
        const auto commentPos = line.find('#');

        if (commentPos != std::string::npos) {
            line.erase(commentPos);
        }

        std::istringstream iss(line);
        std::string token;

        while (iss >> token) {
            tokens.push_back(token);
        }
    }

    if (tokens.empty()) {
        throw std::runtime_error("Input file contains no tokens: " + fileName);
    }

    std::size_t pos = 0;

    auto nextInt = [&]() -> int {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing integer token");
        }

        return parseStrictInt(tokens[pos++], "integer token");
    };

    auto nextDouble = [&]() -> double {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing floating-point token");
        }

        const std::string token = tokens[pos++];

        try {
            std::size_t used = 0;
            const double value = std::stod(token, &used);

            if (used != token.size()) {
                throw std::runtime_error("invalid trailing characters");
            }

            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite value");
            }

            return value;
        } catch (...) {
            throw std::runtime_error("Malformed input: invalid floating-point token '" + token + "'");
        }
    };

    SimulationConfig cfg{};

    const int rawWidth = nextInt();
    const int rawHeight = nextInt();

    if (rawWidth < 3 || rawHeight < 3) {
        throw std::runtime_error("Grid dimensions must be at least 3 x 3");
    }

    cfg.gridWidth = static_cast<std::size_t>(rawWidth);
    cfg.gridHeight = static_cast<std::size_t>(rawHeight);

    const int measuredCount = nextInt();

    if (measuredCount < 0) {
        throw std::runtime_error("Number of measured points cannot be negative");
    }

    cfg.measuredPoints.resize(static_cast<std::size_t>(measuredCount));

    for (int i = 0; i < measuredCount; ++i) {
        auto& p = cfg.measuredPoints[static_cast<std::size_t>(i)];
        p.x = nextDouble();
        p.y = nextDouble();
        p.value = nextDouble();
    }

    cfg.domainStartX = nextDouble();
    cfg.domainStartY = nextDouble();
    cfg.domainWidth = nextDouble();
    cfg.domainHeight = nextDouble();
    cfg.maxFractalIterations = nextInt();
    cfg.timeSteps = nextInt();

    if (cfg.domainWidth <= 0.0 || cfg.domainHeight <= 0.0) {
        throw std::runtime_error("domainWidth and domainHeight must be positive");
    }

    if (cfg.maxFractalIterations <= 0) {
        throw std::runtime_error("maxFractalIterations must be positive");
    }

    if (cfg.timeSteps < 0) {
        throw std::runtime_error("timeSteps must be non-negative");
    }

    if (pos < tokens.size()) {
        cfg.outputEvery = nextInt();

        if (cfg.outputEvery < 0) {
            throw std::runtime_error("outputEvery must be non-negative");
        }
    }

    if (pos != tokens.size()) {
        throw std::runtime_error("Malformed input: unexpected extra tokens at end of file");
    }

    return cfg;
}

GridMapping buildGridMapping(const SimulationConfig& cfg) {
    GridMapping mapping{};
    mapping.x0 = cfg.domainStartX;
    mapping.y0 = cfg.domainStartY;
    mapping.dx = cfg.domainWidth / static_cast<double>(cfg.gridWidth - 1);
    mapping.dy = cfg.domainHeight / static_cast<double>(cfg.gridHeight - 1);
    return mapping;
}

inline double analyticalReferenceField(double x, double y) noexcept {
    return (x * x * x + y * y * y) / 6.0;
}

double computeMeanDiscrepancy(const SimulationConfig& cfg) {
    if (cfg.measuredPoints.empty()) {
        return 0.0;
    }

    double sum = 0.0;

    for (const auto& p : cfg.measuredPoints) {
        sum += p.value - analyticalReferenceField(p.x, p.y);
    }

    return sum / static_cast<double>(cfg.measuredPoints.size());
}

UpdateCoefficients buildUpdateCoefficients(double dx, double dy, double damping = 100.0) {
    if (dx <= 0.0 || dy <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }

    if (damping <= 0.0) {
        throw std::invalid_argument("Damping parameter must be positive");
    }

    UpdateCoefficients c{};
    c.damping = damping;
    c.stepX = dx;
    c.stepY = dy;
    c.laplaceX = -2.0 * (1.0 + c.damping * c.stepX / (c.stepX * c.stepX + c.damping));
    c.laplaceY = -2.0 * (1.0 + c.damping * c.stepY / (c.stepY * c.stepY + c.damping));
    c.coeffX = (c.stepX + c.damping * std::exp(c.stepX)) / (15.0 * c.damping + c.stepX);
    c.coeffY = (c.stepY + c.damping * std::exp(c.stepY)) / (15.0 * c.damping + c.stepY);
    return c;
}

#ifdef USE_HDF5

class TimeSeriesWriter {
public:
    TimeSeriesWriter(
        const std::string& fileName,
        std::size_t width,
        std::size_t height,
        std::size_t batch = 32,
        std::size_t tileY = 256,
        std::size_t tileX = 256
    )
        : file_(fileName, H5F_ACC_TRUNC),
          width_(width),
          height_(height),
          batch_(batch),
          frameCount_(0),
          capacity_(batch),
          closed_(false)
    {
        if (width_ == 0 || height_ == 0) {
            throw std::invalid_argument("TimeSeriesWriter: invalid dimensions");
        }

        if (batch_ == 0) {
            throw std::invalid_argument("TimeSeriesWriter: batch must be positive");
        }

        if (tileY == 0 || tileX == 0) {
            throw std::invalid_argument("TimeSeriesWriter: tile sizes must be positive");
        }

        const hsize_t chunkY = static_cast<hsize_t>(std::min<std::size_t>(height_, tileY));
        const hsize_t chunkX = static_cast<hsize_t>(std::min<std::size_t>(width_, tileX));

        {
            hsize_t dims[3] = {
                0,
                static_cast<hsize_t>(height_),
                static_cast<hsize_t>(width_)
            };

            hsize_t maxdims[3] = {
                H5S_UNLIMITED,
                static_cast<hsize_t>(height_),
                static_cast<hsize_t>(width_)
            };

            H5::DataSpace space(3, dims, maxdims);
            H5::DSetCreatPropList props;

            hsize_t chunks[3] = {1, chunkY, chunkX};
            props.setChunk(3, chunks);

            fieldDataset_ = file_.createDataSet(
                "/field",
                H5::PredType::NATIVE_DOUBLE,
                space,
                props
            );
        }

        {
            hsize_t dims[1] = {0};
            hsize_t maxdims[1] = {H5S_UNLIMITED};

            H5::DataSpace space(1, dims, maxdims);
            H5::DSetCreatPropList props;

            hsize_t chunks[1] = {static_cast<hsize_t>(batch_)};
            props.setChunk(1, chunks);

            stepDataset_ = file_.createDataSet(
                "/step",
                H5::PredType::NATIVE_INT,
                space,
                props
            );
        }

        extend(capacity_);
    }

    ~TimeSeriesWriter() {
        try {
            close();
        } catch (...) {
        }
    }

    TimeSeriesWriter(const TimeSeriesWriter&) = delete;
    TimeSeriesWriter& operator=(const TimeSeriesWriter&) = delete;

    void writeFrame(int stepNumber, const double* field) {
        if (closed_) {
            throw std::runtime_error("TimeSeriesWriter: write after close");
        }

        if (!field) {
            throw std::runtime_error("TimeSeriesWriter: null field pointer");
        }

        if (frameCount_ >= capacity_) {
            capacity_ += batch_;
            extend(capacity_);
        }

        {
            H5::DataSpace filespace = fieldDataset_.getSpace();

            hsize_t start[3] = {
                static_cast<hsize_t>(frameCount_),
                0,
                0
            };

            hsize_t count[3] = {
                1,
                static_cast<hsize_t>(height_),
                static_cast<hsize_t>(width_)
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(3, count);

            fieldDataset_.write(
                field,
                H5::PredType::NATIVE_DOUBLE,
                memspace,
                filespace
            );
        }

        {
            H5::DataSpace filespace = stepDataset_.getSpace();

            hsize_t start[1] = {
                static_cast<hsize_t>(frameCount_)
            };

            hsize_t count[1] = {1};

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(1, count);

            int value = stepNumber;

            stepDataset_.write(
                &value,
                H5::PredType::NATIVE_INT,
                memspace,
                filespace
            );
        }

        ++frameCount_;
    }

    void close() {
        if (closed_) {
            return;
        }

        if (frameCount_ != capacity_) {
            extend(frameCount_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);
        fieldDataset_.close();
        stepDataset_.close();
        file_.close();

        closed_ = true;
    }

private:
    void extend(std::size_t newSize) {
        hsize_t fieldDims[3] = {
            static_cast<hsize_t>(newSize),
            static_cast<hsize_t>(height_),
            static_cast<hsize_t>(width_)
        };

        fieldDataset_.extend(fieldDims);

        hsize_t stepDims[1] = {
            static_cast<hsize_t>(newSize)
        };

        stepDataset_.extend(stepDims);
    }

    H5::H5File file_;
    H5::DataSet fieldDataset_;
    H5::DataSet stepDataset_;

    std::size_t width_{};
    std::size_t height_{};
    std::size_t batch_{};
    std::size_t frameCount_{};
    std::size_t capacity_{};

    bool closed_{false};
};

#else

class TimeSeriesWriter {
public:
    TimeSeriesWriter(
        const std::string&,
        std::size_t,
        std::size_t,
        std::size_t = 32,
        std::size_t = 256,
        std::size_t = 256
    )
    {
        throw std::runtime_error(
            "This executable was built without HDF5 support. "
            "Recompile with -DUSE_HDF5 to enable HDF5 output."
        );
    }

    void writeFrame(int, const double*) {}
    void close() {}
};

#endif

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

    const int p = linearIndexDev(i, j, nx);

    const double cReal = x0 + dx * static_cast<double>(i);
    const double cImag = y0 + dy * static_cast<double>(j);

    double zReal = 0.0;
    double zImag = 0.0;

    int iter = 0;

    for (; iter < maxIters; ++iter) {
        if (zReal * zReal + zImag * zImag > 4.0) {
            break;
        }

        const double tmp = zReal * zReal - zImag * zImag + cReal;
        zImag = 2.0 * zReal * zImag + cImag;
        zReal = tmp;
    }

    weight[p] = iter;
}

__global__
void initializeFieldKernel(
    double* field,
    const int* weight,
    int nx,
    int ny,
    double x0,
    double y0,
    double dx,
    double dy,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) {
        return;
    }

    const int p = linearIndexDev(i, j, nx);

    const double x = x0 + dx * static_cast<double>(i);
    const double y = y0 + dy * static_cast<double>(j);

    const double denom = (maxWeight > minWeight)
        ? static_cast<double>(maxWeight - minWeight)
        : 1.0;

    const double reference = (x * x * x + y * y * y) / 6.0;
    const double normalizedWeight =
        static_cast<double>(weight[p] - minWeight) / denom;

    field[p] =
        293.16 +
        80.0 *
        (meanDiscrepancy + reference) *
        normalizedWeight;
}

__global__
void updateInteriorKernel(
    const double* __restrict__ current,
    double* __restrict__ next,
    int nx,
    int ny,
    double laplaceX,
    double laplaceY,
    double coeffX,
    double coeffY
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) {
        return;
    }

    const int p = linearIndexDev(i, j, nx);

    next[p] =
        coeffX *
        (
            current[p - 1] +
            current[p + 1] +
            (laplaceX + 0.5 / coeffX) * current[p]
        )
        +
        coeffY *
        (
            current[p - nx] +
            current[p + nx] +
            (laplaceY + 0.5 / coeffY) * current[p]
        );
}

__global__
void applyBoundaryLeftRightKernel(double* field, int nx, int ny) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < 1 || j >= ny - 1) {
        return;
    }

    const int row = j * nx;

    field[row] = field[row + 1];
    field[row + nx - 1] = field[row + nx - 2];
}

__global__
void applyBoundaryTopBottomKernel(double* field, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nx) {
        return;
    }

    field[i] = field[nx + i];
    field[(ny - 1) * nx + i] = field[(ny - 2) * nx + i];
}

dim3 gridFor2D(int nx, int ny, dim3 block2d) {
    return dim3(
        static_cast<unsigned>((nx + static_cast<int>(block2d.x) - 1) / static_cast<int>(block2d.x)),
        static_cast<unsigned>((ny + static_cast<int>(block2d.y) - 1) / static_cast<int>(block2d.y))
    );
}

void launchComputeWeights(
    int* d_weight,
    int nx,
    int ny,
    const GridMapping& mapping,
    int maxIters,
    dim3 block2d
) {
    computeWeightKernel<<<gridFor2D(nx, ny, block2d), block2d>>>(
        d_weight,
        nx,
        ny,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        maxIters
    );

    CUDA_CHECK_LAST();
}

void launchInitializeField(
    double* d_field,
    const int* d_weight,
    int nx,
    int ny,
    const GridMapping& mapping,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight,
    dim3 block2d
) {
    initializeFieldKernel<<<gridFor2D(nx, ny, block2d), block2d>>>(
        d_field,
        d_weight,
        nx,
        ny,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        meanDiscrepancy,
        minWeight,
        maxWeight
    );

    CUDA_CHECK_LAST();
}

void launchAdvanceOneStep(
    const double* d_current,
    double* d_next,
    int nx,
    int ny,
    const UpdateCoefficients& coeffs,
    dim3 block2d,
    int block1d
) {
    updateInteriorKernel<<<gridFor2D(nx, ny, block2d), block2d>>>(
        d_current,
        d_next,
        nx,
        ny,
        coeffs.laplaceX,
        coeffs.laplaceY,
        coeffs.coeffX,
        coeffs.coeffY
    );

    CUDA_CHECK_LAST();

    const int gridY = (ny + block1d - 1) / block1d;

    applyBoundaryLeftRightKernel<<<gridY, block1d>>>(
        d_next,
        nx,
        ny
    );

    CUDA_CHECK_LAST();

    const int gridX = (nx + block1d - 1) / block1d;

    applyBoundaryTopBottomKernel<<<gridX, block1d>>>(
        d_next,
        nx,
        ny
    );

    CUDA_CHECK_LAST();
}

void advanceTemperatureFieldSteps(
    DeviceBuffer<double>& currentField,
    DeviceBuffer<double>& nextField,
    int numberOfSteps,
    int nx,
    int ny,
    const UpdateCoefficients& coeffs,
    dim3 block2d,
    int block1d
) {
    for (int step = 0; step < numberOfSteps; ++step) {
        launchAdvanceOneStep(
            currentField.get(),
            nextField.get(),
            nx,
            ny,
            coeffs,
            block2d,
            block1d
        );

        swap(currentField, nextField);
    }
}

std::pair<int, int> computeWeightRangeOnDevice(int* d_weight, std::size_t count) {
    if (!d_weight || count == 0) {
        throw std::runtime_error("computeWeightRangeOnDevice: empty device field");
    }

    thrust::device_ptr<int> first(d_weight);
    thrust::device_ptr<int> last = first + count;

    const auto minmaxPair = thrust::minmax_element(first, last);

    const int minWeight = *minmaxPair.first;
    const int maxWeight = *minmaxPair.second;

    return {minWeight, maxWeight};
}

FieldStatistics computeFieldStatistics(const double* field, std::size_t count) {
    if (!field || count == 0) {
        throw std::runtime_error("computeFieldStatistics: empty field");
    }

    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sumSquares = 0.0;
    double checksum = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double value = field[i];

        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
        sumSquares += value * value;
        checksum += value * static_cast<double>((i % 1009U) + 1U);
    }

    const double mean = sum / static_cast<double>(count);

    double sumSquaredDiff = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const double diff = field[i] - mean;
        sumSquaredDiff += diff * diff;
    }

    FieldStatistics stats{};
    stats.minValue = minValue;
    stats.meanValue = mean;
    stats.maxValue = maxValue;
    stats.stdDev = std::sqrt(sumSquaredDiff / static_cast<double>(count));
    stats.l2Norm = std::sqrt(sumSquares);
    stats.checksum = checksum;

    return stats;
}

void writeStatisticsHeader(std::ostream& out) {
    out << "Step;Min;Mean;Max;Std_dev;L2_norm;Checksum\n";
}

void writeStatisticsRow(std::ostream& out, int step, const FieldStatistics& stats) {
    out
        << step << ';'
        << std::setprecision(15) << stats.minValue << ';'
        << std::setprecision(15) << stats.meanValue << ';'
        << std::setprecision(15) << stats.maxValue << ';'
        << std::setprecision(15) << stats.stdDev << ';'
        << std::setprecision(15) << stats.l2Norm << ';'
        << std::setprecision(15) << stats.checksum << '\n';
}

CommandLineOptions parseCommandLineArguments(int argc, char** argv) {
    CommandLineOptions options{};
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--device requires a value");
            }
            options.deviceId = parseStrictInt(argv[++i], "device");
            continue;
        }
         
        if (arg == "--block-x") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--block-x requires a value");
            }
            options.blockX = parseStrictPositiveInt(argv[++i], "block-x");
            continue;
        }
         
        if (arg == "--block-y") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--block-y requires a value");
            }
            options.blockY = parseStrictPositiveInt(argv[++i], "block-y");
            continue;
        }
          
        if (arg == "--block1d") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--block1d requires a value");
            }
            options.block1d = parseStrictPositiveInt(argv[++i], "block1d");
            continue;
        }
        
        if (arg == "--no-hdf5") {
            options.writeHdf5 = false;
            options.h5File = "none";
            continue;
        }
        
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  " << argv[0]
                << " [options] [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]\n\n"
                << "Options:\n"
                << "  --device N       Select CUDA device, default 0\n"
                << "  --block-x N      CUDA 2D block x dimension, default 16\n"
                << "  --block-y N      CUDA 2D block y dimension, default 16\n"
                << "  --block1d N      CUDA 1D block size, default 256\n"
                << "  --no-hdf5        Disable HDF5 output\n"
                << "  --help, -h       Print this help message\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " --device 0 input_final.in none Statistics_cuda.csv 0\n"
                << "  " << argv[0] << " --device 0 input_medium.in output.h5 Statistics_cuda.csv 50\n\n"
                << "Default official grading mode disables HDF5.\n";

            std::exit(0);
        }

        if (beginsWithDoubleDash(arg)) {
            throw std::runtime_error("Unknown option: " + arg);
        }

        positional.push_back(arg);
    }

    if (positional.size() > 4) {
        throw std::runtime_error("Too many positional arguments");
    }

    if (positional.size() >= 1) {
        options.inputFile = positional[0];
    }

    if (positional.size() >= 2) {
        options.h5File = positional[1];
        options.writeHdf5 = !isNoHdf5Token(options.h5File);
    }

    if (positional.size() >= 3) {
        options.csvFile = positional[2];
    }

    if (positional.size() >= 4) {
        options.outputEvery = parseStrictInt(positional[3], "outputEvery");
        options.overrideOutputEvery = true;

        if (options.outputEvery < 0) {
            throw std::runtime_error("outputEvery must be non-negative");
        }
    }

    return options;
}

bool shouldWriteStep(int step, int finalStep, int outputEvery) {
    if (step == finalStep) {
        return true;
    }

    if (outputEvery <= 0) {
        return false;
    }

    if (step == 0) {
        return true;
    }

    return (step % outputEvery) == 0;
}

std::vector<int> buildOutputSchedule(int finalStep, int outputEvery) {
    std::vector<int> steps;

    if (outputEvery > 0) {
        steps.push_back(0);

        for (int step = outputEvery; step < finalStep; step += outputEvery) {
            steps.push_back(step);
        }
    }

    if (steps.empty() || steps.back() != finalStep) {
        steps.push_back(finalStep);
    }

    return steps;
}

void printRunHeader(
    const CommandLineOptions& cli,
    const SimulationConfig& cfg,
    bool hdf5Compiled,
    const cudaDeviceProp& prop,
    dim3 block2d
) {
    std::cout << "Input file:                    " << cli.inputFile << '\n';
    std::cout << "CSV output:                    " << cli.csvFile << '\n';
    std::cout << "HDF5 compiled:                 " << (hdf5Compiled ? "yes" : "no") << '\n';
    std::cout << "HDF5 output:                   " << (cli.writeHdf5 ? cli.h5File : "disabled") << '\n';
    std::cout << "Official grading mode:         " << (!cli.writeHdf5 ? "yes" : "no") << '\n';
    std::cout << "Grid:                          " << cfg.gridWidth << " x " << cfg.gridHeight << '\n';
    std::cout << "Measured points:               " << cfg.measuredPoints.size() << '\n';
    std::cout << "Max fractal iterations:        " << cfg.maxFractalIterations << '\n';
    std::cout << "Time steps:                    " << cfg.timeSteps << '\n';

    if (cfg.outputEvery == 0) {
        std::cout << "Snapshot/statistics policy:     final step only\n";
    } else {
        std::cout << "Snapshot/statistics policy:     step 0, every "
                  << cfg.outputEvery << " step(s), and final step\n";
    }

    std::cout << "CUDA device:                    " << cli.deviceId << " - " << prop.name << '\n';
    std::cout << "CUDA compute capability:        " << prop.major << '.' << prop.minor << '\n';
    std::cout << "CUDA global memory:             "
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "CUDA block2d:                   " << block2d.x << " x " << block2d.y << '\n';
    std::cout << "CUDA block1d:                   " << cli.block1d << '\n';
    std::cout << '\n';
}

int main(int argc, char** argv) {
#ifdef USE_HDF5
    H5::Exception::dontPrint();
#endif

    try {
        const CommandLineOptions cli = parseCommandLineArguments(argc, argv);

        selectCudaDevice(cli.deviceId);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, cli.deviceId));

        const dim3 block2d(
            static_cast<unsigned>(cli.blockX),
            static_cast<unsigned>(cli.blockY),
            1
        );

        validateCudaLaunchConfig(block2d, cli.block1d);

        SimulationConfig cfg = readConfigurationFile(cli.inputFile);

        if (cli.overrideOutputEvery) {
            cfg.outputEvery = cli.outputEvery;
        }

#ifndef USE_HDF5
        if (cli.writeHdf5) {
            throw std::runtime_error(
                "HDF5 output requested, but executable was built without HDF5 support. "
                "Use 'none' for the HDF5 argument or rebuild with -DUSE_HDF5."
            );
        }
#endif

        ensureParentDirectoryExists(cli.csvFile);

        if (cli.writeHdf5) {
            ensureParentDirectoryExists(cli.h5File);
        }

        const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);

        if (cfg.gridWidth > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            cfg.gridHeight > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            totalCells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Grid exceeds 32-bit CUDA indexing range used by this implementation");
        }

        const int nx = static_cast<int>(cfg.gridWidth);
        const int ny = static_cast<int>(cfg.gridHeight);

        const GridMapping mapping = buildGridMapping(cfg);
        const UpdateCoefficients coeffs = buildUpdateCoefficients(mapping.dx, mapping.dy, 100.0);
        const double meanDiscrepancy = computeMeanDiscrepancy(cfg);

#ifdef USE_HDF5
        constexpr bool hdf5Compiled = true;
#else
        constexpr bool hdf5Compiled = false;
#endif

        printRunHeader(cli, cfg, hdf5Compiled, prop, block2d);

        std::ofstream csv(cli.csvFile);

        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + cli.csvFile);
        }

        writeStatisticsHeader(csv);

        DeviceBuffer<int> d_weight(totalCells);
        DeviceBuffer<double> d_currentField(totalCells);
        DeviceBuffer<double> d_nextField(totalCells);
        PinnedHostBuffer<double> hostField(totalCells);
        
	ScopedTimer totalTimer;

        CudaEventTimer gpuTimer;

        gpuTimer.start();
        launchComputeWeights(
            d_weight.get(),
            nx,
            ny,
            mapping,
            cfg.maxFractalIterations,
            block2d
        );
        const double weightKernelTime = gpuTimer.stopSeconds();

        ScopedTimer weightRangeTimer;
        const auto [minWeight, maxWeight] = computeWeightRangeOnDevice(
            d_weight.get(),
            totalCells
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        const double weightRangeTime = weightRangeTimer.elapsedSeconds();

        gpuTimer.start();
        launchInitializeField(
            d_currentField.get(),
            d_weight.get(),
            nx,
            ny,
            mapping,
            meanDiscrepancy,
            minWeight,
            maxWeight,
            block2d
        );
        const double initKernelTime = gpuTimer.stopSeconds();

        std::unique_ptr<TimeSeriesWriter> writer;

        if (cli.writeHdf5) {
            writer = std::make_unique<TimeSeriesWriter>(
                cli.h5File,
                cfg.gridWidth,
                cfg.gridHeight,
                32,
                256,
                256
            );
        }

        double pureDynamicsKernelTime = 0.0;
        double deviceToHostCopyTime = 0.0;
        double statisticsTime = 0.0;
        double csvTime = 0.0;
        double hdf5Time = 0.0;

        int outputFrames = 0;

        FieldStatistics finalStats{};

        auto copyCurrentFieldToHost = [&]() {
            ScopedTimer copyTimer;

            CUDA_CHECK(cudaMemcpy(
                hostField.data(),
                d_currentField.get(),
                totalCells * sizeof(double),
                cudaMemcpyDeviceToHost
            ));

            deviceToHostCopyTime += copyTimer.elapsedSeconds();
        };

        auto writeOutputFrame = [&](int step) {
            copyCurrentFieldToHost();

            ScopedTimer statsTimer;
            const FieldStatistics stats = computeFieldStatistics(
                hostField.data(),
                totalCells
            );
            statisticsTime += statsTimer.elapsedSeconds();

            finalStats = stats;

            ScopedTimer csvTimer;
            writeStatisticsRow(csv, step, stats);
            csvTime += csvTimer.elapsedSeconds();

            if (writer) {
                ScopedTimer hdf5Timer;
                writer->writeFrame(step, hostField.data());
                hdf5Time += hdf5Timer.elapsedSeconds();
            }

            ++outputFrames;
        };

        const std::vector<int> outputSchedule =
            buildOutputSchedule(cfg.timeSteps, cfg.outputEvery);

        ScopedTimer loopTimer;

        int currentStep = 0;

        for (const int targetStep : outputSchedule) {
            if (targetStep < currentStep || targetStep > cfg.timeSteps) {
                throw std::runtime_error("Internal error: invalid output schedule");
            }

            const int stepsToAdvance = targetStep - currentStep;

            if (stepsToAdvance > 0) {
                gpuTimer.start();

                advanceTemperatureFieldSteps(
                    d_currentField,
                    d_nextField,
                    stepsToAdvance,
                    nx,
                    ny,
                    coeffs,
                    block2d,
                    cli.block1d
                );

                pureDynamicsKernelTime += gpuTimer.stopSeconds();
                currentStep = targetStep;
            }

            if (shouldWriteStep(targetStep, cfg.timeSteps, cfg.outputEvery)) {
                writeOutputFrame(targetStep);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        if (writer) {
            writer->close();
        }

        csv.flush();

        const double loopWallTime = loopTimer.elapsedSeconds();
        const double totalWallTime = totalTimer.elapsedSeconds();

        const double updates =
            static_cast<double>(cfg.gridWidth - 2) *
            static_cast<double>(cfg.gridHeight - 2) *
            static_cast<double>(cfg.timeSteps);

        std::cout << "Weight field GPU kernel time:       " << weightKernelTime << " s\n";
        std::cout << "Weight range reduction time:         " << weightRangeTime << " s\n";
        std::cout << "Initialization GPU kernel time:      " << initKernelTime << " s\n";
        std::cout << "Dynamics update kernels only:        " << pureDynamicsKernelTime << " s\n";
        std::cout << "Device-to-host copy time:            " << deviceToHostCopyTime << " s\n";
        std::cout << "Statistics time:                     " << statisticsTime << " s\n";
        std::cout << "CSV write time:                      " << csvTime << " s\n";
        std::cout << "HDF5 write time:                     " << hdf5Time << " s\n";
        std::cout << "Dynamics loop wall incl. copy/I/O:   " << loopWallTime << " s\n";
        std::cout << "Total measured wall time:            " << totalWallTime << " s\n";
        std::cout << "Output frames:                       " << outputFrames << '\n';

        if (cfg.timeSteps > 0 && pureDynamicsKernelTime > 0.0) {
            std::cout << "Performance, update kernels only:    "
                      << updates / pureDynamicsKernelTime / 1.0e9
                      << " GLUP/s\n";
        }

        if (cfg.timeSteps > 0 && loopWallTime > 0.0) {
            std::cout << "Performance, end-to-end loop:        "
                      << updates / loopWallTime / 1.0e9
                      << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:                    "
                  << std::setprecision(15) << meanDiscrepancy << '\n';

        std::cout << "Final min:                           "
                  << std::setprecision(15) << finalStats.minValue << '\n';

        std::cout << "Final mean:                          "
                  << std::setprecision(15) << finalStats.meanValue << '\n';

        std::cout << "Final max:                           "
                  << std::setprecision(15) << finalStats.maxValue << '\n';

        std::cout << "Final std.dev.:                      "
                  << std::setprecision(15) << finalStats.stdDev << '\n';

        std::cout << "Final L2 norm:                       "
                  << std::setprecision(15) << finalStats.l2Norm << '\n';

        std::cout << "Final checksum:                      "
                  << std::setprecision(15) << finalStats.checksum << '\n';

        std::cout << "Weight range:                        "
                  << minWeight << " ... " << maxWeight << '\n';

        std::cout << "\nSimulation completed successfully.\n";

        return 0;

#ifdef USE_HDF5
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 ERROR: " << e.getDetailMsg() << '\n';
        return 1;
#endif
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "CRITICAL ERROR: unknown failure\n";
        return 1;
    }
}
