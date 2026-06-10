#include <H5Cpp.h>

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
#include <system_error>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


// ============================================================
// PORTABILITY HELPERS
// ============================================================

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

using index_t = std::ptrdiff_t;


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

    // outputEvery is intentionally optional.
    // If hasOutputEvery == false, only the final state is written.
    bool hasOutputEvery{false};
    int outputEvery{0};

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
// UTILITIES
// ============================================================

inline std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) noexcept {
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
    int value = 0;

    const auto* begin = s.data();
    const auto* end = s.data() + s.size();

    auto [ptr, ec] = std::from_chars(begin, end, value);

    if (ec != std::errc{} || ptr != end) {
        throw std::runtime_error("Invalid " + what + ": '" + s + "'");
    }

    return value;
}

std::size_t parseStrictPositiveSize(const std::string& s, const std::string& what) {
    const int v = parseStrictInt(s, what);

    if (v <= 0) {
        throw std::runtime_error(what + " must be > 0");
    }

    return static_cast<std::size_t>(v);
}

bool startsWithDashDash(const std::string& s) {
    return s.rfind("--", 0) == 0;
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
            const int v = std::stoi(s, &used);

            if (used != s.size()) {
                throw std::runtime_error("not a pure integer");
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
            const double v = std::stod(s, &used);

            if (used != s.size()) {
                throw std::runtime_error("not a pure floating-point number");
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

    // Optional outputEvery.
    //
    // Old format:
    //   ... maxIters steps outputEvery
    //
    // New allowed format:
    //   ... maxIters steps
    //
    // If outputEvery is absent, the simulation writes only the final state.
    if (pos < tokens.size()) {
        cfg.outputEvery = nextInt();
        cfg.hasOutputEvery = true;

        if (cfg.outputEvery <= 0) {
            throw std::runtime_error("outputEvery must be > 0");
        }
    }

    if (pos != tokens.size()) {
        throw std::runtime_error("Malformed input: unexpected extra tokens at end of file");
    }

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
        sum += static_cast<long double>(
            m.v - analyticalField(m.x, m.y)
        );
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

    c.dgx = -2.0 * (1.0 + c.dd * c.hx / (c.hx * c.hx + c.dd));
    c.dgy = -2.0 * (1.0 + c.dd * c.hy / (c.hy * c.hy + c.dd));

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

        // /step dataset: [frame]
        {
            hsize_t dims[1] = {0};
            hsize_t maxdims[1] = {H5S_UNLIMITED};

            H5::DataSpace space(1, dims, maxdims);
            H5::DSetCreatPropList prop;

            hsize_t chunks[1] = {
                static_cast<hsize_t>(batch_)
            };

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

    void write(int stepNumber, const std::vector<double>& field) {
        if (closed_) {
            throw std::runtime_error("H5Writer: write() called after close()");
        }

        if (field.size() != safeGridSize(nx_, ny_)) {
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

            hsize_t count[3] = {
                1,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(3, count);

            field_.write(
                field.data(),
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

            hsize_t count[1] = {
                1
            };

            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(1, count);

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

        if (frame_ != capacity_) {
            extend(frame_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);

        field_.close();
        step_.close();
        file_.close();

        closed_ = true;
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

private:
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
// OPENMP COMPUTE ROUTINES
// ============================================================

void computeWeight(
    std::vector<int>& weight,
    const Config& cfg,
    const DomainMap& map
) {
    const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

    if (weight.size() != N) {
        throw std::runtime_error("computeWeight: weight size mismatch");
    }

    const index_t nx = static_cast<index_t>(cfg.nx);
    const index_t ny = static_cast<index_t>(cfg.ny);

    // Mandelbrot work is irregular, so dynamic scheduling improves load balance.
#pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (index_t j = 0; j < ny; ++j) {
        for (index_t i = 0; i < nx; ++i) {
            const std::size_t is = static_cast<std::size_t>(i);
            const std::size_t js = static_cast<std::size_t>(j);

            const std::size_t p = idx2D(is, js, cfg.nx);

            const double ca = map.x0 + map.dx * static_cast<double>(i);
            const double cb = map.y0 + map.dy * static_cast<double>(j);

            double za = 0.0;
            double zb = 0.0;

            int it = 0;

            for (; it < cfg.maxIters; ++it) {
                if (za * za + zb * zb > 4.0) {
                    break;
                }

                const double tmp = za * za - zb * zb + ca;
                zb = 2.0 * za * zb + cb;
                za = tmp;
            }

            weight[p] = it;
        }
    }
}

void initializeField(
    std::vector<double>& u,
    const std::vector<int>& weight,
    const Config& cfg,
    const DomainMap& map,
    double discrepancy
) {
    const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

    if (u.size() != N || weight.size() != N) {
        throw std::runtime_error("initializeField: size mismatch");
    }

    const auto [wminIt, wmaxIt] =
        std::minmax_element(weight.begin(), weight.end());

    const int wmin = *wminIt;
    const int wmax = *wmaxIt;

    const double denom =
        (wmax > wmin) ? static_cast<double>(wmax - wmin) : 1.0;

    const index_t nx = static_cast<index_t>(cfg.nx);
    const index_t ny = static_cast<index_t>(cfg.ny);

#pragma omp parallel for collapse(2) schedule(static)
    for (index_t j = 0; j < ny; ++j) {
        for (index_t i = 0; i < nx; ++i) {
            const std::size_t is = static_cast<std::size_t>(i);
            const std::size_t js = static_cast<std::size_t>(j);

            const std::size_t p = idx2D(is, js, cfg.nx);

            const double x = map.x0 + map.dx * static_cast<double>(i);
            const double y = map.y0 + map.dy * static_cast<double>(j);

            const double F = analyticalField(x, y);
            const double wnorm =
                static_cast<double>(weight[p] - wmin) / denom;

            u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm;
        }
    }
}

void updateInterior(
    const double* RESTRICT u1,
    double* RESTRICT u2,
    std::size_t nxSize,
    std::size_t nySize,
    const CoolingCoeffs& c
) {
    const index_t nx = static_cast<index_t>(nxSize);
    const index_t ny = static_cast<index_t>(nySize);

#pragma omp parallel for collapse(2) schedule(static)
    for (index_t j = 1; j < ny - 1; ++j) {
        for (index_t i = 1; i < nx - 1; ++i) {
            const std::size_t is = static_cast<std::size_t>(i);
            const std::size_t js = static_cast<std::size_t>(j);

            const std::size_t p = idx2D(is, js, nxSize);

            u2[p] =
                c.CX * (
                    u1[idx2D(is - 1, js, nxSize)]
                    + u1[idx2D(is + 1, js, nxSize)]
                    + (c.dgx + 0.5 / c.CX) * u1[p]
                )
                + c.CY * (
                    u1[idx2D(is, js - 1, nxSize)]
                    + u1[idx2D(is, js + 1, nxSize)]
                    + (c.dgy + 0.5 / c.CY) * u1[p]
                );
        }
    }
}

void applyBoundary(double* u, std::size_t nxSize, std::size_t nySize) {
    const index_t nx = static_cast<index_t>(nxSize);
    const index_t ny = static_cast<index_t>(nySize);

    // 1. Left/right boundaries first, excluding corners.
#pragma omp parallel for schedule(static)
    for (index_t j = 1; j < ny - 1; ++j) {
        const std::size_t js = static_cast<std::size_t>(j);

        u[idx2D(0, js, nxSize)] =
            u[idx2D(1, js, nxSize)];

        u[idx2D(nxSize - 1, js, nxSize)] =
            u[idx2D(nxSize - 2, js, nxSize)];
    }

    // 2. Top/bottom boundaries including corners.
    //
    // This intentionally runs after the left/right update to preserve
    // the same corner behavior as the CUDA-style two-kernel version.
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < nx; ++i) {
        const std::size_t is = static_cast<std::size_t>(i);

        u[idx2D(is, 0, nxSize)] =
            u[idx2D(is, 1, nxSize)];

        u[idx2D(is, nySize - 1, nxSize)] =
            u[idx2D(is, nySize - 2, nxSize)];
    }
}

void updateField(
    const double* RESTRICT u1,
    double* RESTRICT u2,
    std::size_t nx,
    std::size_t ny,
    const CoolingCoeffs& c
) {
    updateInterior(u1, u2, nx, ny, c);
    applyBoundary(u2, nx, ny);
}


// ============================================================
// STATISTICS
// ============================================================

Stats computeStatsAccurate(const std::vector<double>& u) {
    if (u.empty()) {
        throw std::runtime_error("computeStatsAccurate: empty field");
    }

    const auto [mnIt, mxIt] =
        std::minmax_element(u.begin(), u.end());

    long double sum = 0.0L;

    for (double v : u) {
        sum += static_cast<long double>(v);
    }

    const long double mean =
        sum / static_cast<long double>(u.size());

    long double ssd = 0.0L;

    for (double v : u) {
        const long double d = static_cast<long double>(v) - mean;
        ssd += d * d;
    }

    Stats s{};

    s.minv = *mnIt;
    s.maxv = *mxIt;
    s.mean = static_cast<double>(mean);
    s.stddev = static_cast<double>(
        std::sqrt(ssd / static_cast<long double>(u.size()))
    );

    return s;
}

Stats computeStatsFastOpenMP(const std::vector<double>& u) {
    if (u.empty()) {
        throw std::runtime_error("computeStatsFastOpenMP: empty field");
    }

    double minv = std::numeric_limits<double>::infinity();
    double maxv = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sum2 = 0.0;

    const index_t n = static_cast<index_t>(u.size());

#pragma omp parallel for schedule(static) reduction(min:minv) reduction(max:maxv) reduction(+:sum,sum2)
    for (index_t i = 0; i < n; ++i) {
        const double v = u[static_cast<std::size_t>(i)];

        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
        sum += v;
        sum2 += v * v;
    }

    const double count = static_cast<double>(u.size());
    const double mean = sum / count;
    const double var = std::max(0.0, sum2 / count - mean * mean);

    Stats s{};

    s.minv = minv;
    s.maxv = maxv;
    s.mean = mean;
    s.stddev = std::sqrt(var);

    return s;
}

Stats computeStats(const std::vector<double>& u, const std::string& mode) {
    if (mode == "accurate") {
        return computeStatsAccurate(u);
    }

    if (mode == "fast") {
        return computeStatsFastOpenMP(u);
    }

    throw std::runtime_error("Unknown stats mode: " + mode);
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
// SIMPLE CLI PARSER
// ============================================================

struct CliOptions {
    std::string inputFile{"input/Cooling.in"};
    std::string h5File{"output/Cooling.h5"};
    std::string csvFile{"output/Statistics.csv"};

    // Optional positional override:
    //   program input h5 csv outputEvery
    bool hasOutputEvery{false};
    int outputEvery{0};

    std::string statsMode{"accurate"};

    int threads{0}; // 0 means use environment/default.

    std::size_t h5TileY{256};
    std::size_t h5TileX{256};
};

CliOptions parseCommandLine(int argc, char** argv) {
    CliOptions opt{};
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--threads") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--threads requires a value");
            }

            opt.threads = parseStrictInt(argv[++i], "threads");

            if (opt.threads <= 0) {
                throw std::runtime_error("--threads must be > 0");
            }
        } else if (arg == "--stats") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--stats requires a value");
            }

            opt.statsMode = argv[++i];

            if (opt.statsMode != "accurate" && opt.statsMode != "fast") {
                throw std::runtime_error("--stats must be either 'accurate' or 'fast'");
            }
        } else if (arg == "--h5-tile-y") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--h5-tile-y requires a value");
            }

            opt.h5TileY = parseStrictPositiveSize(argv[++i], "h5-tile-y");
        } else if (arg == "--h5-tile-x") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--h5-tile-x requires a value");
            }

            opt.h5TileX = parseStrictPositiveSize(argv[++i], "h5-tile-x");
        } else if (startsWithDashDash(arg)) {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 4) {
        throw std::runtime_error(
            "Too many positional arguments. Expected: input h5 csv outputEvery"
        );
    }

    if (positional.size() >= 1) {
        opt.inputFile = positional[0];
    }

    if (positional.size() >= 2) {
        opt.h5File = positional[1];
    }

    if (positional.size() >= 3) {
        opt.csvFile = positional[2];
    }

    if (positional.size() >= 4) {
        opt.outputEvery = parseStrictInt(positional[3], "outputEvery");
        opt.hasOutputEvery = true;

        if (opt.outputEvery <= 0) {
            throw std::runtime_error("outputEvery must be > 0");
        }
    }

    return opt;
}


// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    H5::Exception::dontPrint();

    try {
        const CliOptions opt = parseCommandLine(argc, argv);

#ifdef _OPENMP
        if (opt.threads > 0) {
            omp_set_num_threads(opt.threads);
        }
#else
        if (opt.threads > 0) {
            std::cerr << "WARNING: --threads ignored because OpenMP is not enabled.\n";
        }
#endif

        Config cfg = readInput(opt.inputFile);

        // Command-line outputEvery overrides input-file outputEvery.
        // If neither specifies it, the program writes only the final state.
        if (opt.hasOutputEvery) {
            cfg.outputEvery = opt.outputEvery;
            cfg.hasOutputEvery = true;
        }

        if (cfg.hasOutputEvery && cfg.outputEvery <= 0) {
            throw std::invalid_argument("outputEvery must be > 0");
        }

        const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

        const DomainMap map = buildDomainMap(cfg);
        const CoolingCoeffs cooling = buildCoolingCoeffs(map.dx, map.dy, 100.0);
        const double discrepancy = computeDiscrepancy(cfg);

        std::vector<int> weight(N);
        std::vector<double> uCurr(N);
        std::vector<double> uNext(N);

        std::ofstream csv(opt.csvFile);

        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + opt.csvFile);
        }

        writeStatsHeader(csv);

        std::cout << "Input file:                   " << opt.inputFile << '\n';
        std::cout << "HDF5 output:                  " << opt.h5File << '\n';
        std::cout << "CSV output:                   " << opt.csvFile << '\n';
        std::cout << "Grid:                         " << cfg.nx << " x " << cfg.ny << '\n';
        std::cout << "Measured points:              " << cfg.measured.size() << '\n';
        std::cout << "Max iterations:               " << cfg.maxIters << '\n';
        std::cout << "Time steps:                   " << cfg.steps << '\n';

        if (cfg.hasOutputEvery) {
            std::cout << "Snapshot every:               "
                      << cfg.outputEvery << " step(s)\n";
        } else {
            std::cout << "Snapshot every:               final step only\n";
        }

        std::cout << "Stats mode:                   " << opt.statsMode << '\n';
        std::cout << "HDF5 chunk tile:              "
                  << std::min<std::size_t>(cfg.ny, opt.h5TileY)
                  << " x "
                  << std::min<std::size_t>(cfg.nx, opt.h5TileX)
                  << '\n';

#ifdef _OPENMP
        std::cout << "OpenMP enabled:               yes\n";
        std::cout << "OpenMP max threads:           " << omp_get_max_threads() << '\n';
#else
        std::cout << "OpenMP enabled:               no\n";
#endif

        std::cout << '\n';

        const auto totalT0 = std::chrono::steady_clock::now();

        // ----------------------------------------------------
        // Weight field
        // ----------------------------------------------------
        const auto weightT0 = std::chrono::steady_clock::now();

        computeWeight(weight, cfg, map);

        const auto weightT1 = std::chrono::steady_clock::now();

        // ----------------------------------------------------
        // Initialization
        // ----------------------------------------------------
        const auto initT0 = std::chrono::steady_clock::now();

        initializeField(uCurr, weight, cfg, map, discrepancy);

        const auto initT1 = std::chrono::steady_clock::now();

        // ----------------------------------------------------
        // Dynamics + output
        // ----------------------------------------------------
        double pureDynamicsTime = 0.0;
        double outputStatsIoTime = 0.0;
        int outputFrames = 0;

        const auto loopT0 = std::chrono::steady_clock::now();

        H5Writer writer(
            opt.h5File,
            cfg.nx,
            cfg.ny,
            32,
            opt.h5TileY,
            opt.h5TileX
        );

        bool hasLastWrittenStep = false;
        int lastWrittenStep = -1;

        auto writeOutputFrame = [&](int step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            const auto outT0 = std::chrono::steady_clock::now();

            writer.write(step, uCurr);
            writeStatsLine(csv, step, computeStats(uCurr, opt.statsMode));
            ++outputFrames;

            const auto outT1 = std::chrono::steady_clock::now();

            outputStatsIoTime +=
                std::chrono::duration<double>(outT1 - outT0).count();

            hasLastWrittenStep = true;
            lastWrittenStep = step;
        };

        // If outputEvery was explicitly specified, keep the traditional
        // behavior and write the initial condition.
        //
        // If outputEvery was not specified, do not write step 0 here.
        // In that mode, only the final state is written after the loop.
        if (cfg.hasOutputEvery) {
            writeOutputFrame(0);
        }

        for (int step = 1; step <= cfg.steps; ++step) {
            const auto dynT0 = std::chrono::steady_clock::now();

            updateField(
                uCurr.data(),
                uNext.data(),
                cfg.nx,
                cfg.ny,
                cooling
            );

            std::swap(uCurr, uNext);

            const auto dynT1 = std::chrono::steady_clock::now();

            pureDynamicsTime +=
                std::chrono::duration<double>(dynT1 - dynT0).count();

            // Periodic loop output only exists when outputEvery was specified.
            if (cfg.hasOutputEvery && (step % cfg.outputEvery) == 0) {
                writeOutputFrame(step);
            }
        }

        // Always write the final state.
        //
        // If periodic output already wrote this same final step,
        // writeOutputFrame() avoids duplication.
        writeOutputFrame(cfg.steps);

        writer.close();

        const auto loopT1 = std::chrono::steady_clock::now();
        const auto totalT1 = std::chrono::steady_clock::now();

        const double weightTime =
            std::chrono::duration<double>(weightT1 - weightT0).count();

        const double initTime =
            std::chrono::duration<double>(initT1 - initT0).count();

        const double loopWallTime =
            std::chrono::duration<double>(loopT1 - loopT0).count();

        const double totalWallTime =
            std::chrono::duration<double>(totalT1 - totalT0).count();

        const double updates =
            static_cast<double>(cfg.nx - 2)
            * static_cast<double>(cfg.ny - 2)
            * static_cast<double>(cfg.steps);

        std::cout << "Weight field time:            " << weightTime << " s\n";
        std::cout << "Init field time:              " << initTime << " s\n";
        std::cout << "Pure dynamics compute time:   " << pureDynamicsTime << " s\n";
        std::cout << "Stats + CSV + HDF5 time:      " << outputStatsIoTime << " s\n";
        std::cout << "Dynamics loop wall time:      " << loopWallTime << " s\n";
        std::cout << "Total measured wall time:     " << totalWallTime << " s\n";
        std::cout << "Output frames:                " << outputFrames << '\n';

        if (cfg.steps > 0 && pureDynamicsTime > 0.0) {
            std::cout << "Pure OpenMP dynamics perf:    "
                      << updates / pureDynamicsTime / 1e9
                      << " GLUP/s\n";
        }

        if (cfg.steps > 0 && loopWallTime > 0.0) {
            std::cout << "End-to-end loop performance:  "
                      << updates / loopWallTime / 1e9
                      << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:             "
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
