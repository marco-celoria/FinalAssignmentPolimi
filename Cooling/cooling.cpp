#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// UTIL
// ============================================================

inline std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) noexcept {
    return i + j * nx;
}

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// ============================================================
// DATA STRUCTURES
// ============================================================

struct MeasuredPoint {
    double x{};
    double y{};
    double v{};
};

struct Config {
    std::size_t nx{}, ny{};
    double Sreal{}, Simag{}, Dreal{}, Dimag{};
    int maxIters{};
    int steps{};
    int outputEvery{1}; // HDF5 snapshot cadence (CLI override)
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
// SAFE GRID SIZE
// ============================================================

std::size_t safeGridSize(std::size_t nx, std::size_t ny)
{
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument("Grid dimensions must be > 0");
    }
    if (nx > std::numeric_limits<std::size_t>::max() / ny) {
        throw std::overflow_error("Grid size overflow: nx * ny exceeds size_t range");
    }
    return nx * ny;
}

// ============================================================
// INPUT PARSER
//   Parses legacy Cooling.inp-like format by stripping comments
//   and reading numeric tokens in order.
//
//   Expected token order after comment stripping:
//
//   nx
//   ny
//   nMeasured
//   (x y v) repeated nMeasured times
//   Sreal
//   Simag
//   Dreal
//   Dimag
//   maxIters
//   steps
//   ppmFlag   <-- ignored here (legacy compatibility)
// ============================================================

Config readInput(const std::string& fname)
{
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
        throw std::runtime_error("Input file is empty or contains no numeric tokens: " + fname);
    }

    std::size_t pos = 0;

    auto nextInt = [&]() -> int {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing integer token");
        }
        try {
            return std::stoi(tokens.at(pos++));
        } catch (const std::exception&) {
            throw std::runtime_error("Malformed input: invalid integer token '" + tokens.at(pos - 1) + "'");
        }
    };

    auto nextDouble = [&]() -> double {
        if (pos >= tokens.size()) {
            throw std::runtime_error("Malformed input: missing floating-point token");
        }
        try {
            return std::stod(tokens.at(pos++));
        } catch (const std::exception&) {
            throw std::runtime_error("Malformed input: invalid floating-point token '" + tokens.at(pos - 1) + "'");
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
        cfg.measured[static_cast<std::size_t>(i)].x = nextDouble();
        cfg.measured[static_cast<std::size_t>(i)].y = nextDouble();
        cfg.measured[static_cast<std::size_t>(i)].v = nextDouble();
    }

    cfg.Sreal    = nextDouble();
    cfg.Simag    = nextDouble();
    cfg.Dreal    = nextDouble();
    cfg.Dimag    = nextDouble();
    cfg.maxIters = nextInt();
    cfg.steps    = nextInt();

    // Legacy input contains a final PPM flag. We parse and ignore it.
    const int legacyPPMFlag = nextInt();
    (void)legacyPPMFlag;

    if (cfg.maxIters <= 0) {
        throw std::runtime_error("maxIters must be > 0");
    }
    if (cfg.steps < 0) {
        throw std::runtime_error("steps must be >= 0");
    }

    // default HDF5 write cadence unless overridden on CLI
    cfg.outputEvery = 1;

    return cfg;
}

// ============================================================
// PHYSICS HELPERS
// ============================================================

DomainMap buildDomainMap(const Config& cfg)
{
    // Clean physics model:
    // treat (i,j) as nodes including both boundaries:
    // x(i) = x0 + i * (Dreal / (nx-1))
    // y(j) = y0 + j * (Dimag / (ny-1))
    //
    // Because nx, ny >= 3 is enforced, division by zero cannot happen.
    DomainMap map;
    map.x0 = cfg.Sreal;
    map.y0 = cfg.Simag;
    map.dx = cfg.Dreal / static_cast<double>(cfg.nx - 1);
    map.dy = cfg.Dimag / static_cast<double>(cfg.ny - 1);
    return map;
}

inline double xAt(std::size_t i, const DomainMap& map) noexcept {
    return map.x0 + map.dx * static_cast<double>(i);
}

inline double yAt(std::size_t j, const DomainMap& map) noexcept {
    return map.y0 + map.dy * static_cast<double>(j);
}

// Continuous theoretical field used for initialization
inline double analyticalField(double x, double y) noexcept
{
    return (x * x * x + y * y * y) / 6.0;
}

double computeDiscrepancy(const Config& cfg)
{
    if (cfg.measured.empty()) {
        return 0.0;
    }

    long double sum = 0.0L;
    for (const auto& m : cfg.measured) {
        sum += static_cast<long double>(m.v - analyticalField(m.x, m.y));
    }
    return static_cast<double>(sum / static_cast<long double>(cfg.measured.size()));
}

// ============================================================
// COOLING COEFFICIENTS
//   Precompute stencil coefficients once to keep update kernel
//   simple and GPU-portable.
// ============================================================

CoolingCoeffs buildCoolingCoeffs(std::size_t nx, std::size_t ny, double dd = 100.0)
{
    if (nx < 3 || ny < 3) {
        throw std::invalid_argument("buildCoolingCoeffs: nx and ny must be at least 3");
    }

    CoolingCoeffs c{};
    c.dd  = dd;
    c.hx  = 1.0 / static_cast<double>(nx - 1);
    c.hy  = 1.0 / static_cast<double>(ny - 1);
    c.dgx = -2.0 * (1.0 + c.dd * c.hx / (c.hx * c.hx + c.dd));
    c.dgy = -2.0 * (1.0 + c.dd * c.hy / (c.hy * c.hy + c.dd));
    c.CX  = (c.hx + c.dd * std::exp(c.hx)) / (15.0 * c.dd + c.hx);
    c.CY  = (c.hy + c.dd * std::exp(c.hy)) / (15.0 * c.dd + c.hy);
    return c;
}


// ============================================================
// HDF5 WRITER
//   Writes:
//     /field : [nframes, ny, nx] double
//     /step  : [nframes] int
// ============================================================

class H5Writer {
public:
    H5Writer(const std::string& fname,
             std::size_t nx,
             std::size_t ny,
             std::size_t batch = 32)
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

        // /field dataset
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

            // append full frames
            hsize_t chunks[3] = {
                1,
                static_cast<hsize_t>(ny_),
                static_cast<hsize_t>(nx_)
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

    ~H5Writer()
    {
        try {
            close();
        } catch (...) {
            // never throw from destructor
        }
    }

    void write(int stepNumber, const std::vector<double>& field)
    {
        if (closed_) {
            throw std::runtime_error("H5Writer: write() called after close()");
        }
        if (field.size() != nx_ * ny_) {
            throw std::runtime_error("H5Writer: field size mismatch");
        }

        if (frame_ >= capacity_) {
            capacity_ += batch_;
            extend(capacity_);
        }

        // write frame
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

            field_.write(field.data(),
                         H5::PredType::NATIVE_DOUBLE,
                         memspace,
                         filespace);
        }

        // write step number
        {
            H5::DataSpace filespace = step_.getSpace();

            hsize_t start[1] = {static_cast<hsize_t>(frame_)};
            hsize_t count[1] = {1};
            filespace.selectHyperslab(H5S_SELECT_SET, count, start);

            H5::DataSpace memspace(1, count);
            int value = stepNumber;

            step_.write(&value,
                        H5::PredType::NATIVE_INT,
                        memspace,
                        filespace);
        }

        ++frame_;
    }

    void close()
    {
        if (closed_) return;
        closed_ = true;

        // shrink datasets to actual number of written frames
        if (frame_ != capacity_) {
            extend(frame_);
        }

        file_.flush(H5F_SCOPE_GLOBAL);
        field_.close();
        step_.close();
        file_.close();
    }

private:
    void extend(std::size_t n)
    {
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

    std::size_t nx_, ny_;
    std::size_t batch_;
    std::size_t frame_;
    std::size_t capacity_;
    bool closed_;
};

// ============================================================
// KERNEL-LIKE COMPUTE ROUTINES
//   These are intentionally structured for later CUDA porting.
// ============================================================

// ------------------------------------------------------------
// 1) Sensitivity / weight field
//
// Standard Mandelbrot escape-time count:
// z_{n+1} = z_n^2 + c, z_0 = 0
//
// For each grid node we store the number of iterations completed
// before |z| > 2, capped by maxIters.
// ------------------------------------------------------------

void computeWeight(std::vector<int>& weight,
                   const Config& cfg,
                   const DomainMap& map)
{
    const std::size_t N = safeGridSize(cfg.nx, cfg.ny);
    if (weight.size() != N) {
        throw std::runtime_error("computeWeight: weight size mismatch");
    }

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t j = 0; j < cfg.ny; ++j) {
        for (std::size_t i = 0; i < cfg.nx; ++i) {
            const std::size_t p = idx2D(i, j, cfg.nx);

            const double ca = xAt(i, map);
            const double cb = yAt(j, map);

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

// ------------------------------------------------------------
// 2) Field initialization
//
// u0(x,y) = 293.16 + 80 * ( discrepancy + F(x,y) ) * w_norm
//
// where:
//   F(x,y) = analyticalField(x,y)
//   w_norm = (weight - wmin)/(wmax - wmin)
// ------------------------------------------------------------

void initializeField(std::vector<double>& u,
                     const std::vector<int>& weight,
                     const Config& cfg,
                     const DomainMap& map,
                     double discrepancy)
{
    const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

    if (u.size() != N || weight.size() != N) {
        throw std::runtime_error("initializeField: size mismatch");
    }

    const auto [wminIt, wmaxIt] = std::minmax_element(weight.begin(), weight.end());
    const int wmin = *wminIt;
    const int wmax = *wmaxIt;
    const double denom = (wmax > wmin) ? static_cast<double>(wmax - wmin) : 1.0;

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t j = 0; j < cfg.ny; ++j) {
        for (std::size_t i = 0; i < cfg.nx; ++i) {
            const std::size_t p = idx2D(i, j, cfg.nx);

            const double x = xAt(i, map);
            const double y = yAt(j, map);

            const double F = analyticalField(x, y);
            const double wnorm = static_cast<double>(weight[p] - wmin) / denom;

            u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm;
        }
    }
}

// ------------------------------------------------------------
// 3) Interior update
//
// This preserves the intended legacy cooling law, but coefficients
// are precomputed once and passed in cleanly.
//
// Boundary values are handled in a separate routine for better
// structure and easier GPU porting.
// ------------------------------------------------------------

void updateInterior(const double* RESTRICT u1,
                    double* RESTRICT u2,
                    std::size_t nx,
                    std::size_t ny,
                    const CoolingCoeffs& c)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t j = 1; j < ny - 1; ++j) {
        for (std::size_t i = 1; i < nx - 1; ++i) {
            const std::size_t p = idx2D(i, j, nx);

            u2[p] =
                c.CX * (
                    u1[idx2D(i - 1, j, nx)] +
                    u1[idx2D(i + 1, j, nx)] +
                    (c.dgx + 0.5 / c.CX) * u1[p]
                )
                +
                c.CY * (
                    u1[idx2D(i, j - 1, nx)] +
                    u1[idx2D(i, j + 1, nx)] +
                    (c.dgy + 0.5 / c.CY) * u1[p]
                );
        }
    }
}

// ------------------------------------------------------------
// 4) Boundary update
//
// Enforces zero-normal-gradient style copying:
//
// left  = neighbor at i=1
// right = neighbor at i=nx-2
// top   = neighbor at j=1
// bot   = neighbor at j=ny-2
//
// Split out separately for clarity and future GPU portability.
// ------------------------------------------------------------

void applyBoundary(double* u, std::size_t nx, std::size_t ny)
{
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t j = 1; j < ny - 1; ++j) {
            u[idx2D(0,     j, nx)] = u[idx2D(1,     j, nx)];
            u[idx2D(nx - 1, j, nx)] = u[idx2D(nx - 2, j, nx)];
        }

#pragma omp for schedule(static)
        for (std::size_t i = 0; i < nx; ++i) {
            u[idx2D(i, 0,      nx)] = u[idx2D(i, 1,      nx)];
            u[idx2D(i, ny - 1, nx)] = u[idx2D(i, ny - 2, nx)];
        }
    }
}

void updateField(const double* RESTRICT u1,
                 double* RESTRICT u2,
                 std::size_t nx,
                 std::size_t ny,
                 const CoolingCoeffs& c)
{
    updateInterior(u1, u2, nx, ny, c);
    applyBoundary(u2, nx, ny);
}

// ============================================================
// OPTIONAL STATISTICS (useful for regression / validation)
// ============================================================

Stats computeStats(const std::vector<double>& u)
{
    if (u.empty()) {
        throw std::runtime_error("computeStats: empty field");
    }

    const auto [mnIt, mxIt] = std::minmax_element(u.begin(), u.end());

    long double sum = 0.0L;
    for (double v : u) {
        sum += static_cast<long double>(v);
    }
    const long double mean = sum / static_cast<long double>(u.size());

    long double ssd = 0.0L;
    for (double v : u) {
        const long double d = static_cast<long double>(v) - mean;
        ssd += d * d;
    }

    Stats s{};
    s.minv   = *mnIt;
    s.maxv   = *mxIt;
    s.mean   = static_cast<double>(mean);
    s.stddev = static_cast<double>(std::sqrt(ssd / static_cast<long double>(u.size())));
    return s;
}

void writeStatsHeader(std::ostream& os)
{
    os << "Step;Min;Mean;Max;Std_dev\n";
}

void writeStatsLine(std::ostream& os, int step, const Stats& s)
{
    os << step << ';'
       << std::setprecision(15) << s.minv << ';'
       << std::setprecision(15) << s.mean << ';'
       << std::setprecision(15) << s.maxv << ';'
       << std::setprecision(15) << s.stddev << '\n';
}

// ============================================================
// MAIN
//
// CLI:
//   argv[1] = input file      (default: Cooling.inp)
//   argv[2] = output HDF5     (default: cooling.h5)
//   argv[3] = output CSV      (default: Statistics.csv)
//   argv[4] = outputEvery     (default: 1)
// ============================================================

int main(int argc, char** argv)
{
    H5::Exception::dontPrint();

    try {
        const std::string inputFile = (argc > 1) ? argv[1] : "Cooling.inp";
        const std::string h5File    = (argc > 2) ? argv[2] : "cooling.h5";
        const std::string csvFile   = (argc > 3) ? argv[3] : "Statistics.csv";

        Config cfg = readInput(inputFile);

        if (argc > 4) {
            cfg.outputEvery = std::stoi(argv[4]);
        }
        if (cfg.outputEvery <= 0) {
            throw std::invalid_argument("outputEvery must be > 0");
        }

        const std::size_t N = safeGridSize(cfg.nx, cfg.ny);

        const DomainMap map = buildDomainMap(cfg);
        const CoolingCoeffs cooling = buildCoolingCoeffs(cfg.nx, cfg.ny, 100.0);
        const double discrepancy = computeDiscrepancy(cfg);

        std::vector<int> weight(N);
        std::vector<double> uCurr(N);
        std::vector<double> uNext(N);

        std::ofstream csv(csvFile);
        if (!csv) {
            throw std::runtime_error("Cannot open CSV output file: " + csvFile);
        }
        writeStatsHeader(csv);

        std::cout << "Input file:      " << inputFile << '\n';
        std::cout << "HDF5 output:     " << h5File << '\n';
        std::cout << "CSV output:      " << csvFile << '\n';
        std::cout << "Grid:            " << cfg.nx << " x " << cfg.ny << '\n';
        std::cout << "Measured points: " << cfg.measured.size() << '\n';
        std::cout << "Max iterations:  " << cfg.maxIters << '\n';
        std::cout << "Time steps:      " << cfg.steps << '\n';
        std::cout << "Snapshot every:  " << cfg.outputEvery << " step(s)\n\n";

        auto t0 = std::chrono::steady_clock::now();

        computeWeight(weight, cfg, map);
        auto t1 = std::chrono::steady_clock::now();

        initializeField(uCurr, weight, cfg, map, discrepancy);
        auto t2 = std::chrono::steady_clock::now();

        H5Writer writer(h5File, cfg.nx, cfg.ny, 32);
        
        // step 0
        writer.write(0, uCurr);
        writeStatsLine(csv, 0, computeStats(uCurr));

       for (int step = 1; step <= cfg.steps; ++step) {
           updateField(uCurr.data(), uNext.data(), cfg.nx, cfg.ny, cooling);
           std::swap(uCurr, uNext);
           if ((step % cfg.outputEvery) == 0 || step == cfg.steps) {
               writer.write(step, uCurr);
               writeStatsLine(csv, step, computeStats(uCurr));
           }
       }

        writer.close();
        auto t3 = std::chrono::steady_clock::now();

        const double tWeight = std::chrono::duration<double>(t1 - t0).count();
        const double tInit   = std::chrono::duration<double>(t2 - t1).count();
        const double tDyn    = std::chrono::duration<double>(t3 - t2).count();

        std::cout << "Weight field time: " << tWeight << " s\n";
        std::cout << "Init field time:   " << tInit   << " s\n";
        std::cout << "Dynamics + I/O:    " << tDyn    << " s\n";

        if (cfg.steps > 0 && tDyn > 0.0) {
            const double updates =
                static_cast<double>(cfg.nx - 2) *
                static_cast<double>(cfg.ny - 2) *
                static_cast<double>(cfg.steps);

            std::cout << "Performance:       "
                      << updates / tDyn / 1e9
                      << " GLUP/s\n";
        }

        std::cout << "Mean discrepancy:  " << discrepancy << '\n';
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
