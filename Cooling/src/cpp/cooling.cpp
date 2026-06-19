/*
================================================================================
Cooling Field Solver - Serial C++17 Baseline
================================================================================

Course final project baseline.

This program is a serial mini-application for parallelization and performance
analysis. Students may develop a parallel version using one or more of:

  - OpenMP
  - MPI
  - CUDA
  - OpenACC
  - hybrid approaches

The application contains:
  - an irregular field-weight computation,
  - a field initialization phase,
  - a 2D iterative stencil update,
  - global statistical reductions,
  - optional HDF5 output.

Official performance grading mode:
  HDF5 output must be disabled.

Recommended official run style:

  ./cooling_serial input_final.in none output_final.csv 0

or simply:

  ./cooling_serial input_final.in

if the default output names are acceptable.

HDF5 support:
  HDF5 is optional at compile time.

  Without HDF5:

    g++ -O3 -std=c++17 -Wall -Wextra -pedantic cooling.cpp \
        -o cooling_serial

  With HDF5:

    g++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 \
        cooling.cpp -o cooling_serial \
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

  ./cooling_serial [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]

Examples:

  ./cooling_serial input_small.in
  ./cooling_serial input_final.in none output_final.csv 0
  ./cooling_serial input_medium.in output.h5 output.csv 50

Rules for student submissions:

  1. The numerical model must not be changed.
  2. The grid size, number of time steps, max iteration count, and input data
     must not be reduced for official measurements.
  3. Students may reorganize data structures, introduce device memory,
     implement MPI domain decomposition, add OpenMP/OpenACC directives,
     write CUDA kernels, change reduction implementations, or change I/O
     implementation.
  4. Students may not remove required computations, skip time steps, omit
     required statistics, hard-code answers, or use precomputed results.
  5. Parallel results are validated against the serial baseline.

================================================================================
*/

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

using index_t = std::ptrdiff_t;
namespace fs = std::filesystem;

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
};

struct ScopedTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point start{clock::now()};

    double elapsedSeconds() const {
        return std::chrono::duration<double>(clock::now() - start).count();
    }
};

inline std::size_t linearIndex(std::size_t i, std::size_t j, std::size_t width) noexcept {
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

        const hsize_t chunkY = static_cast<hsize_t>(std::min<std::size_t>(height_, tileY));
        const hsize_t chunkX = static_cast<hsize_t>(std::min<std::size_t>(width_, tileX));

        {
            hsize_t dims[3] = {0, static_cast<hsize_t>(height_), static_cast<hsize_t>(width_)};
            hsize_t maxdims[3] = {H5S_UNLIMITED, static_cast<hsize_t>(height_), static_cast<hsize_t>(width_)};

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

    void writeFrame(int stepNumber, const std::vector<double>& field) {
        if (closed_) {
            throw std::runtime_error("TimeSeriesWriter: write after close");
        }

        if (field.size() != checkedGridSize(width_, height_)) {
            throw std::runtime_error("TimeSeriesWriter: field size mismatch");
        }

        if (frameCount_ >= capacity_) {
            capacity_ += batch_;
            extend(capacity_);
        }

        {
            H5::DataSpace filespace = fieldDataset_.getSpace();

            hsize_t start[3] = {static_cast<hsize_t>(frameCount_), 0, 0};
            hsize_t count[3] = {
                1,
                static_cast<hsize_t>(height_),
                static_cast<hsize_t>(width_)
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(3, count);

            fieldDataset_.write(
                field.data(),
                H5::PredType::NATIVE_DOUBLE,
                memspace,
                filespace
            );
        }

        {
            H5::DataSpace filespace = stepDataset_.getSpace();

            hsize_t start[1] = {static_cast<hsize_t>(frameCount_)};
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

        hsize_t stepDims[1] = {static_cast<hsize_t>(newSize)};
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

    void writeFrame(int, const std::vector<double>&) {}
    void close() {}
};

#endif

void computeFractalWeights(
    std::vector<int>& weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping
) {
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);

    if (weightField.size() != totalCells) {
        throw std::runtime_error("computeFractalWeights: size mismatch");
    }

    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);

    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, cfg.gridWidth);

            const double cReal = mapping.x0 + mapping.dx * static_cast<double>(i);
            const double cImag = mapping.y0 + mapping.dy * static_cast<double>(j);

            double zReal = 0.0;
            double zImag = 0.0;
            int iter = 0;

            for (; iter < cfg.maxFractalIterations; ++iter) {
                if (zReal * zReal + zImag * zImag > 4.0) {
                    break;
                }

                const double tmp = zReal * zReal - zImag * zImag + cReal;
                zImag = 2.0 * zReal * zImag + cImag;
                zReal = tmp;
            }

            weightField[idx] = iter;
        }
    }
}

std::pair<int, int> computeWeightRange(const std::vector<int>& weightField) {
    if (weightField.empty()) {
        throw std::runtime_error("computeWeightRange: empty field");
    }

    const auto [minIt, maxIt] = std::minmax_element(weightField.begin(), weightField.end());
    return {*minIt, *maxIt};
}

void initializeTemperatureField(
    std::vector<double>& temperature,
    const std::vector<int>& weightField,
    const SimulationConfig& cfg,
    const GridMapping& mapping,
    double meanDiscrepancy,
    int minWeight,
    int maxWeight
) {
    const std::size_t totalCells = checkedGridSize(cfg.gridWidth, cfg.gridHeight);

    if (temperature.size() != totalCells || weightField.size() != totalCells) {
        throw std::runtime_error("initializeTemperatureField: size mismatch");
    }

    const double denom = (maxWeight > minWeight)
        ? static_cast<double>(maxWeight - minWeight)
        : 1.0;

    const index_t width = static_cast<index_t>(cfg.gridWidth);
    const index_t height = static_cast<index_t>(cfg.gridHeight);

    for (index_t j = 0; j < height; ++j) {
        for (index_t i = 0; i < width; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, cfg.gridWidth);

            const double x = mapping.x0 + mapping.dx * static_cast<double>(i);
            const double y = mapping.y0 + mapping.dy * static_cast<double>(j);

            const double normalizedWeight =
                static_cast<double>(weightField[idx] - minWeight) / denom;

            temperature[idx] =
                293.16 +
                80.0 *
                (meanDiscrepancy + analyticalReferenceField(x, y)) *
                normalizedWeight;
        }
    }
}

void updateInterior(
    const double* RESTRICT current,
    double* RESTRICT next,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs
) {
    const index_t w = static_cast<index_t>(width);
    const index_t h = static_cast<index_t>(height);

    for (index_t j = 1; j < h - 1; ++j) {
        for (index_t i = 1; i < w - 1; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t jj = static_cast<std::size_t>(j);
            const std::size_t idx = linearIndex(ii, jj, width);

            next[idx] =
                coeffs.coeffX *
                (
                    current[linearIndex(ii - 1, jj, width)] +
                    current[linearIndex(ii + 1, jj, width)] +
                    (coeffs.laplaceX + 0.5 / coeffs.coeffX) * current[idx]
                )
                +
                coeffs.coeffY *
                (
                    current[linearIndex(ii, jj - 1, width)] +
                    current[linearIndex(ii, jj + 1, width)] +
                    (coeffs.laplaceY + 0.5 / coeffs.coeffY) * current[idx]
                );
        }
    }
}

void applyBoundaryConditions(double* field, std::size_t width, std::size_t height) {
    for (std::size_t j = 1; j < height - 1; ++j) {
        field[linearIndex(0, j, width)] = field[linearIndex(1, j, width)];
        field[linearIndex(width - 1, j, width)] = field[linearIndex(width - 2, j, width)];
    }

    for (std::size_t i = 0; i < width; ++i) {
        field[linearIndex(i, 0, width)] = field[linearIndex(i, 1, width)];
        field[linearIndex(i, height - 1, width)] = field[linearIndex(i, height - 2, width)];
    }
}

void advanceTemperatureField(
    const double* current,
    double* next,
    std::size_t width,
    std::size_t height,
    const UpdateCoefficients& coeffs
) {
    updateInterior(current, next, width, height, coeffs);
    applyBoundaryConditions(next, width, height);
}

FieldStatistics computeFieldStatistics(const std::vector<double>& field) {
    if (field.empty()) {
        throw std::runtime_error("computeFieldStatistics: empty field");
    }

    const std::size_t n = field.size();

    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sumSquares = 0.0;
    double checksum = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double value = field[i];

        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
        sum += value;
        sumSquares += value * value;
        checksum += value * static_cast<double>((i % 1009U) + 1U);
    }

    const double mean = sum / static_cast<double>(n);

    double sumSquaredDiff = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double diff = field[i] - mean;
        sumSquaredDiff += diff * diff;
    }

    FieldStatistics stats{};
    stats.minValue = minValue;
    stats.meanValue = mean;
    stats.maxValue = maxValue;
    stats.stdDev = std::sqrt(sumSquaredDiff / static_cast<double>(n));
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

        if (arg == "--no-hdf5") {
            options.writeHdf5 = false;
            options.h5File = "none";
            continue;
        }

        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  " << argv[0] << " [inputFile] [h5File|none|--no-hdf5] [csvFile] [outputEvery]\n\n"
                << "Examples:\n"
                << "  " << argv[0] << " input_final.in none Statistics.csv 0\n"
                << "  " << argv[0] << " input_medium.in output.h5 Statistics.csv 50\n\n"
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

void printRunHeader(
    const CommandLineOptions& cli,
    const SimulationConfig& cfg,
    bool hdf5Compiled
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

    std::cout << '\n';
}

int main(int argc, char** argv) {
#ifdef USE_HDF5
    H5::Exception::dontPrint();
#endif

    try {
        const CommandLineOptions cli = parseCommandLineArguments(argc, argv);
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
        const GridMapping mapping = buildGridMapping(cfg);
        const UpdateCoefficients coeffs = buildUpdateCoefficients(mapping.dx, mapping.dy, 100.0);
        const double meanDiscrepancy = computeMeanDiscrepancy(cfg);

#ifdef USE_HDF5
        constexpr bool hdf5Compiled = true;
#else
        constexpr bool hdf5Compiled = false;
#endif

        printRunHeader(cli, cfg, hdf5Compiled);

        std::vector<int> weightField(totalCells);
        std::vector<double> currentField(totalCells);
        std::vector<double> nextField(totalCells);

        std::ofstream csv(cli.csvFile);

        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + cli.csvFile);
        }

        writeStatisticsHeader(csv);

        ScopedTimer totalTimer;

        ScopedTimer weightTimer;
        computeFractalWeights(weightField, cfg, mapping);
        const double weightTime = weightTimer.elapsedSeconds();

        ScopedTimer rangeTimer;
        const auto [minWeight, maxWeight] = computeWeightRange(weightField);
        const double weightRangeTime = rangeTimer.elapsedSeconds();

        ScopedTimer initTimer;
        initializeTemperatureField(
            currentField,
            weightField,
            cfg,
            mapping,
            meanDiscrepancy,
            minWeight,
            maxWeight
        );
        const double initTime = initTimer.elapsedSeconds();

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

        double pureDynamicsTime = 0.0;
        double statisticsTime = 0.0;
        double csvTime = 0.0;
        double hdf5Time = 0.0;

        int outputFrames = 0;
        bool hasLastWrittenStep = false;
        int lastWrittenStep = -1;

        FieldStatistics finalStats{};

        auto writeOutputFrame = [&](int step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            ScopedTimer statsTimer;
            const FieldStatistics stats = computeFieldStatistics(currentField);
            statisticsTime += statsTimer.elapsedSeconds();
            finalStats = stats;

            ScopedTimer csvTimer;
            writeStatisticsRow(csv, step, stats);
            csvTime += csvTimer.elapsedSeconds();

            if (writer) {
                ScopedTimer hdf5Timer;
                writer->writeFrame(step, currentField);
                hdf5Time += hdf5Timer.elapsedSeconds();
            }

            ++outputFrames;
            hasLastWrittenStep = true;
            lastWrittenStep = step;
        };

        ScopedTimer loopTimer;

        if (shouldWriteStep(0, cfg.timeSteps, cfg.outputEvery)) {
            writeOutputFrame(0);
        }

        for (int step = 1; step <= cfg.timeSteps; ++step) {
            ScopedTimer stepTimer;
            advanceTemperatureField(
                currentField.data(),
                nextField.data(),
                cfg.gridWidth,
                cfg.gridHeight,
                coeffs
            );
            pureDynamicsTime += stepTimer.elapsedSeconds();

            std::swap(currentField, nextField);

            if (shouldWriteStep(step, cfg.timeSteps, cfg.outputEvery)) {
                writeOutputFrame(step);
            }
        }

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

        std::cout << "Weight field time:             " << weightTime << " s\n";
        std::cout << "Weight range reduction time:   " << weightRangeTime << " s\n";
        std::cout << "Initialization time:           " << initTime << " s\n";
        std::cout << "Pure dynamics compute time:    " << pureDynamicsTime << " s\n";
        std::cout << "Statistics time:               " << statisticsTime << " s\n";
        std::cout << "CSV write time:                " << csvTime << " s\n";
        std::cout << "HDF5 write time:               " << hdf5Time << " s\n";
        std::cout << "Dynamics loop wall time:       " << loopWallTime << " s\n";
        std::cout << "Total measured wall time:      " << totalWallTime << " s\n";
        std::cout << "Output frames:                 " << outputFrames << '\n';

        if (cfg.timeSteps > 0 && pureDynamicsTime > 0.0) {
            std::cout << "Pure dynamics performance:     "
                      << updates / pureDynamicsTime / 1.0e9
                      << " GLUP/s\n";
        }

        if (cfg.timeSteps > 0 && loopWallTime > 0.0) {
            std::cout << "Loop end-to-end performance:   "
                      << updates / loopWallTime / 1.0e9
                      << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:              "
                  << std::setprecision(15) << meanDiscrepancy << '\n';

        std::cout << "Final min:                     "
                  << std::setprecision(15) << finalStats.minValue << '\n';

        std::cout << "Final mean:                    "
                  << std::setprecision(15) << finalStats.meanValue << '\n';

        std::cout << "Final max:                     "
                  << std::setprecision(15) << finalStats.maxValue << '\n';

        std::cout << "Final std.dev.:                "
                  << std::setprecision(15) << finalStats.stdDev << '\n';

        std::cout << "Final L2 norm:                 "
                  << std::setprecision(15) << finalStats.l2Norm << '\n';

        std::cout << "Final checksum:                "
                  << std::setprecision(15) << finalStats.checksum << '\n';

        std::cout << "Weight range:                  "
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
