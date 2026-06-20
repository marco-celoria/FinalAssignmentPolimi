
/*
================================================================================
Particle System Solver - Serial C++17 Baseline
================================================================================

Course final project baseline for parallel programming with OpenMP, MPI, CUDA,
OpenACC, or hybrid approaches.

Primary objective
-----------------
Students should optimize and parallelize the particle dynamics phase.

Benchmark/no-output mode
---------------------------------
HDF5 output disabled. No screen generation. 
Final validation quantities are still printed.

Recommended benchmark run:

  ./path/to/particles_serial input/Particles.in none 0

Use the external validator on /step,/weight,/pos,/vel. 
The /screen dataset is intended mainly for visualization/debugging.

HDF5 support is optional at compile time.

Without HDF5:

  g++ -O3 -std=c++17 -Wall -Wextra -pedantic particles.cpp -o particles_serial

With HDF5:

  g++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 \
      particles.cpp -o particles_serial -lhdf5_cpp -lhdf5

or, depending on the system:

  h5c++ -O3 -std=c++17 -Wall -Wextra -pedantic -DUSE_HDF5 \
      particles.cpp -o particles_serial

Command line:

  ./particles_serial [inputFile] [h5File|none|--no-hdf5] [outputEvery]

Examples:

  ./particles_serial input/Particles.in none 0
  ./particles_serial input/Particles.in output/Particles_cpp.h5
  ./particles_serial input/Particles.in output/Particles_cpp.h5 1000

Input file format, after removing empty lines and comment-only lines beginning
with '#':

  generatingGridNx
  generatingGridNy
  generatingGridXs
  generatingGridXe
  generatingGridYs
  generatingGridYe
  screenGridNx
  screenGridNy
  screenGridXs
  screenGridXe
  screenGridYs
  screenGridYe
  maxFractalIterations
  timeSteps
  dt
  outputEvery

outputEvery:
  0  means final HDF5 frame only, if HDF5 output is enabled.
  >0 means step 0, every outputEvery steps, and the final step.

================================================================================
*/

#ifdef USE_HDF5
#include <H5Cpp.h>
#endif

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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>


// ============================================================
// CONSTANTS: part of the numerical model. Do not change.
// ============================================================

constexpr double kForce = 1.0e-3;
constexpr double eps    = 1.0e-2;
constexpr double eps2   = eps * eps;


// ============================================================
// UTILITIES
// ============================================================

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
    return text == "none" ||
           text == "NONE" ||
           text == "-" ||
           text == "--no-hdf5";
}

void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [inputFile] [h5File|none|--no-hdf5] [outputEvery]\n"
        << "Examples:\n"
        << "  " << prog << " input/Particles.in none 0\n"
        << "  " << prog << " input/Particles.in output/Particles_cpp.h5 1000\n";
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

    // Structure-of-arrays layout. This is deliberate: it is friendly to SIMD,
    // cache blocking, CUDA/OpenACC offload, and MPI packing.
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

    long long raw_g_nx  = 0;
    long long raw_g_ny  = 0;
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

    long long raw_max_iters    = 0;
    long long raw_max_steps    = 0;
    long long raw_output_every = 0;

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
// MANDELBROT FIELD GENERATION
// Secondary parallelization target: embarrassingly parallel over grid points.
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
// PARTICLE GENERATION
// ============================================================

Particles generateParticles(const Grid& g, const Grid& pg) {
    if (g.values.empty()) {
        throw std::runtime_error("generateParticles: empty generating field");
    }

    Particles P;

    const auto vmax = *std::max_element(g.values.begin(), g.values.end());
    auto vmin       = *std::min_element(g.values.begin(), g.values.end());

    // Select only grid points with iteration counts close to the maximum.
    // This keeps the particle count large enough for HPC benchmarking but
    // small enough for the O(N^2) dynamics to remain feasible.
    // This computes floor((29*vmax + vmin) / 30) without forming the
    // potentially overflowing product 29*vmax directly.
    {
        const unsigned long long qmax = vmax / 30ULL;
        const unsigned long long rmax = vmax % 30ULL;
        const unsigned long long qmin = vmin / 30ULL;
        const unsigned long long rmin = vmin % 30ULL;

        vmin = 29ULL * qmax + qmin + (29ULL * rmax + rmin) / 30ULL;
    }

    const std::size_t count = static_cast<std::size_t>(
        std::count_if(
            g.values.begin(),
            g.values.end(),
            [&](unsigned long long v) { return v >= vmin; }
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
                + (pg.xe - pg.xs) * static_cast<double>(i)
                / static_cast<double>(g.nx - 1);

            P.y[n] = pg.ys
                + (pg.ye - pg.ys) * static_cast<double>(j)
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
// PRIMARY HPC KERNEL: FORCE COMPUTATION
// ============================================================
// This O(N^2) all-pairs interaction loop is the main target for optimization
// and parallelization. Students may change the implementation strategy, data
// layout, blocking, MPI decomposition, GPU offload, or exploit pair symmetry,
// but the force law and the inclusion of all pair interactions must be
// preserved.
// ============================================================

void computeForces(const Particles& P, double* fx, double* fy) {
    const std::size_t N = P.n;

    if (!fx || !fy) {
        throw std::runtime_error("computeForces: null output pointer");
    }

    const double* const x = P.x.data();
    const double* const y = P.y.data();
    const double* const w = P.w.data();

    for (std::size_t i = 0; i < N; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double wi = w[i];

        double fxi = 0.0;
        double fyi = 0.0;

        for (std::size_t j = 0; j < N; ++j) {
            if (i != j) {
                const double dx = x[j] - xi;
                const double dy = y[j] - yi;

                const double r2 = dx * dx + dy * dy + eps2;
                const double invr  = 1.0 / std::sqrt(r2);
                const double invr2 = invr * invr;
                const double invr3 = invr2 * invr;

                const double coeff = kForce * wi * w[j] * invr3;

                fxi += coeff * dx;
                fyi += coeff * dy;
            }
        }

        fx[i] = fxi;
        fy[i] = fyi;
    }
}


// ============================================================
// VELOCITY-VERLET INTEGRATION
// ============================================================

void integrateVV(
    Particles& P,
    std::vector<double>& fx,
    std::vector<double>& fy,
    std::vector<double>& fx_new,
    std::vector<double>& fy_new,
    double dt
) {
    const std::size_t N = P.n;

    if (fx.size() != N || fy.size() != N || fx_new.size() != N || fy_new.size() != N) {
        throw std::runtime_error("integrateVV: force array size mismatch");
    }

    for (std::size_t i = 0; i < N; ++i) {
        assert(P.w[i] > 0.0);

        const double invm = 1.0 / P.w[i];

        P.vx[i] += 0.5 * fx[i] * invm * dt;
        P.vy[i] += 0.5 * fy[i] * invm * dt;

        P.x[i] += P.vx[i] * dt;
        P.y[i] += P.vy[i] * dt;
    }

    computeForces(P, fx_new.data(), fy_new.data());

    for (std::size_t i = 0; i < N; ++i) {
        assert(P.w[i] > 0.0);

        const double invm = 1.0 / P.w[i];

        P.vx[i] += 0.5 * fx_new[i] * invm * dt;
        P.vy[i] += 0.5 * fy_new[i] * invm * dt;
    }

    fx.swap(fx_new);
    fy.swap(fy_new);
}


// ============================================================
// SCREEN BUILDING
// Visualization/debugging only. Official benchmark runs should disable HDF5,
// so this function should not affect benchmark timing.
// ============================================================

void buildScreen(Grid& screen, const Particles& P, double wmin, double wr) {
    if (screen.values.empty()) {
        throw std::runtime_error("buildScreen: empty screen grid");
    }

    if (wr <= 0.0) {
        throw std::runtime_error("buildScreen: invalid weight range");
    }

    std::fill(screen.values.begin(), screen.values.end(), 0ULL);

    const double invdx = static_cast<double>(screen.nx - 1) / (screen.xe - screen.xs);
    const double invdy = static_cast<double>(screen.ny - 1) / (screen.ye - screen.ys);

    for (std::size_t n = 0; n < P.n; ++n) {
        int ix = static_cast<int>((P.x[n] - screen.xs) * invdx);
        int iy = static_cast<int>((P.y[n] - screen.ys) * invdy);

        if (ix < 0) {
            ix = 0;
        } else if (ix > static_cast<int>(screen.nx - 1)) {
            ix = static_cast<int>(screen.nx - 1);
        }

        if (iy < 0) {
            iy = 0;
        } else if (iy > static_cast<int>(screen.ny - 1)) {
            iy = static_cast<int>(screen.ny - 1);
        }

        int wp_i = static_cast<int>(10.0 * (P.w[n] - wmin) / wr);

        if (wp_i < 0) {
            wp_i = 0;
        } else if (wp_i > 1000) {
            wp_i = 1000;
        }

        const unsigned long long wp = static_cast<unsigned long long>(wp_i);

        for (int dj = -1; dj <= 1; ++dj) {
            const int jy = iy + dj;
            if (jy < 0 || jy >= static_cast<int>(screen.ny)) {
                continue;
            }

            for (int di = -1; di <= 1; ++di) {
                const int jx = ix + di;
                if (jx < 0 || jx >= static_cast<int>(screen.nx)) {
                    continue;
                }

                const std::size_t p =
                    static_cast<std::size_t>(jx)
                    + static_cast<std::size_t>(jy) * screen.nx;

                screen.values[p] += wp;
            }
        }
    }
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

    // O(N^2), computed only once at the end. Useful because it depends on the
    // final particle distribution and catches many classes of errors.
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
// Datasets:
//   /weight  shape = (particles)              fixed
//   /step    shape = (frames)                 unlimited
//   /pos     shape = (frames, particles, 2)   unlimited in frame dimension
//   /vel     shape = (frames, particles, 2)   unlimited in frame dimension
//   /screen  shape = (frames, ny, nx)         unlimited in frame dimension
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
        try {
            close();
        } catch (...) {
        }
    }

    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;

    void writeMetadata(
        const std::string& inputFile,
        const Config& cfg,
        const Grid& gen,
        const Grid& screenGrid
    ) {
        H5::Group root = file_.openGroup("/");

        writeStringAttribute(root, "application", "Particle System Solver");
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

    void writeFrame(std::size_t stepNumber, const Particles& P, const Grid& screen) {
        if (closed_) {
            throw std::runtime_error("H5StreamWriter: write after close");
        }
        if (P.n != np_) {
            throw std::runtime_error("H5StreamWriter: particle size mismatch");
        }
        if (screen.nx != nx_ || screen.ny != ny_) {
            throw std::runtime_error("H5StreamWriter: screen size mismatch");
        }
        if (screen.values.size() != safeGridSize(nx_, ny_)) {
            throw std::runtime_error("H5StreamWriter: screen storage size mismatch");
        }

        if (currentFrame_ >= capacity_) {
            capacity_ += chunkFrames_;
            extendDatasets(capacity_);
        }

        for (std::size_t i = 0; i < np_; ++i) {
            Pbuf_[2 * i]     = P.x[i];
            Pbuf_[2 * i + 1] = P.y[i];

            Vbuf_[2 * i]     = P.vx[i];
            Vbuf_[2 * i + 1] = P.vy[i];
        }

        writeDoubleFrame(pos_, Pbuf_.data(), currentFrame_, np_, 2);
        writeDoubleFrame(vel_, Vbuf_.data(), currentFrame_, np_, 2);
        writeScreenFrame(screen_, screen.values.data(), currentFrame_, ny_, nx_);
        writeStep(step_, static_cast<long long>(stepNumber), currentFrame_);

        ++currentFrame_;
    }

    hsize_t framesWritten() const noexcept {
        return currentFrame_;
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
        const hsize_t screenChunkY =
            static_cast<hsize_t>(std::min<std::size_t>(ny_, screenTileY));
        const hsize_t screenChunkX =
            static_cast<hsize_t>(std::min<std::size_t>(nx_, screenTileX));

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
        {
            std::array<hsize_t, 3> size = {newSize, static_cast<hsize_t>(np_), 2};
            pos_.extend(size.data());
            vel_.extend(size.data());
        }
        {
            std::array<hsize_t, 3> size = {newSize, static_cast<hsize_t>(ny_), static_cast<hsize_t>(nx_)};
            screen_.extend(size.data());
        }
        {
            std::array<hsize_t, 1> size = {newSize};
            step_.extend(size.data());
        }
    }

    void shrinkToFit() {
        extendDatasets(currentFrame_);
    }

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
    H5StreamWriter(
        const std::string&,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t = 64,
        std::size_t = 256,
        std::size_t = 256
    ) {
        throw std::runtime_error(
            "This executable was built without HDF5 support. "
            "Use 'none' as the HDF5 output argument, or rebuild with -DUSE_HDF5."
        );
    }

    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;

    void writeMetadata(const std::string&, const Config&, const Grid&, const Grid&) {}
    void writeWeights(const Particles&) {}
    void writeFrame(std::size_t, const Particles&, const Grid&) {}

    std::size_t framesWritten() const noexcept { return 0; }
    void close() {}
};

#endif


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

        const std::string inputFile = (argc > 1) ? argv[1] : "input/Particles.in";
        const std::string outputFile = (argc > 2) ? argv[2] : "none";
        const bool writeHdf5 = !isNoHdf5Token(outputFile);

#ifndef USE_HDF5
        if (writeHdf5) {
            throw std::runtime_error(
                "HDF5 output requested, but this executable was built without HDF5 support. "
                "Use 'none' as the output argument, or rebuild with -DUSE_HDF5."
            );
        }
#endif

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

        // ----------------------------------------------------
        // 1. Generate Mandelbrot field
        // ----------------------------------------------------
        const auto mandelStart = std::chrono::steady_clock::now();
        computeGeneratingField(gen, cfg.maxIters);
        const auto mandelStop = std::chrono::steady_clock::now();

        const double mandelSeconds =
            std::chrono::duration<double>(mandelStop - mandelStart).count();

        // ----------------------------------------------------
        // 2. Generate particles
        // ----------------------------------------------------
        const auto particleStart = std::chrono::steady_clock::now();
        Particles P = generateParticles(gen, screen);
        const auto particleStop = std::chrono::steady_clock::now();

        const double particleGenerationSeconds =
            std::chrono::duration<double>(particleStop - particleStart).count();

        const std::size_t N = P.n;
        std::cout << "Particles:                  " << N << "\n";

        // ----------------------------------------------------
        // 3. Allocate force arrays
        // ----------------------------------------------------
        std::vector<double> fx(N, 0.0);
        std::vector<double> fy(N, 0.0);
        std::vector<double> fx_new(N, 0.0);
        std::vector<double> fy_new(N, 0.0);

        // ----------------------------------------------------
        // 4. Initial force computation
        // ----------------------------------------------------
        const auto initForceStart = std::chrono::steady_clock::now();
        computeForces(P, fx.data(), fy.data());
        const auto initForceStop = std::chrono::steady_clock::now();

        const double initForceSeconds =
            std::chrono::duration<double>(initForceStop - initForceStart).count();

        // ----------------------------------------------------
        // 5. Weight range for screen output
        // ----------------------------------------------------
        const auto [wminIt, wmaxIt] = std::minmax_element(P.w.begin(), P.w.end());
        const double wmin = *wminIt;
        const double wmax = *wmaxIt;
        const double wr   = std::max(wmax - wmin, 1.0);

        // ----------------------------------------------------
        // 6. Optional HDF5 writer
        // ----------------------------------------------------
        std::unique_ptr<H5StreamWriter> h5;

        if (writeHdf5) {
            h5 = std::make_unique<H5StreamWriter>(outputFile, N, screen.nx, screen.ny);
            h5->writeMetadata(inputFile, cfg, gen, screen);
            h5->writeWeights(P);
        }

        std::size_t outputFrames = 0;
        bool hasLastWrittenStep = false;
        std::size_t lastWrittenStep = 0;

        double screenBuildSeconds = 0.0;
        double hdf5WriteSeconds = 0.0;

        auto writeOutputFrame = [&](std::size_t step) {
            if (!h5) {
                return;
            }

            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            const auto screenStart = std::chrono::steady_clock::now();
            buildScreen(screen, P, wmin, wr);
            const auto screenStop = std::chrono::steady_clock::now();

            screenBuildSeconds +=
                std::chrono::duration<double>(screenStop - screenStart).count();

            const auto h5Start = std::chrono::steady_clock::now();
            h5->writeFrame(step, P, screen);
            const auto h5Stop = std::chrono::steady_clock::now();

            hdf5WriteSeconds +=
                std::chrono::duration<double>(h5Stop - h5Start).count();

            hasLastWrittenStep = true;
            lastWrittenStep = step;
            ++outputFrames;
        };

        // ----------------------------------------------------
        // 7. Simulation loop
        // ----------------------------------------------------
        const auto loopStart = std::chrono::steady_clock::now();
        double pureDynamicsSeconds = 0.0;

        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            if (h5 && shouldWriteStep(step, cfg.maxSteps, cfg.outputEvery)) {
                writeOutputFrame(step);
            }

            const auto dynStart = std::chrono::steady_clock::now();
            integrateVV(P, fx, fy, fx_new, fy_new, cfg.dt);
            const auto dynStop = std::chrono::steady_clock::now();

            pureDynamicsSeconds +=
                std::chrono::duration<double>(dynStop - dynStart).count();
        }

        if (h5) {
            writeOutputFrame(cfg.maxSteps);
            h5->close();
        }

        const auto loopStop = std::chrono::steady_clock::now();
        const double loopWallSeconds =
            std::chrono::duration<double>(loopStop - loopStart).count();

        // ----------------------------------------------------
        // 8. Validation
        // ----------------------------------------------------
        const auto validationStart = std::chrono::steady_clock::now();
        const ValidationQuantities validation = computeValidationQuantities(P);
        const auto validationStop = std::chrono::steady_clock::now();

        const double validationSeconds =
            std::chrono::duration<double>(validationStop - validationStart).count();

        // ----------------------------------------------------
        // 9. Reporting
        // ----------------------------------------------------
        // This performance metric intentionally excludes the initial force
        // computation and reports only the repeated dynamics loop.
        const double interactions =
            static_cast<double>(N)
            * static_cast<double>(N - 1)
            * static_cast<double>(cfg.maxSteps);

        const double gigaInteractions = interactions / 1.0e9;

        std::cout << "Simulation completed successfully.\n";
        std::cout << "Output frames:                 " << outputFrames << "\n";
        std::cout << "Mandelbrot wall time:          " << mandelSeconds << " s\n";
        std::cout << "Particle generation wall time: " << particleGenerationSeconds << " s\n";
        std::cout << "Initial force wall time:       " << initForceSeconds << " s\n";
        std::cout << "Pure dynamics time:            " << pureDynamicsSeconds << " s\n";
        std::cout << "Screen build time:             " << screenBuildSeconds << " s\n";
        std::cout << "HDF5 write time:               " << hdf5WriteSeconds << " s\n";
        std::cout << "Validation time:               " << validationSeconds << " s\n";
        std::cout << "Loop wall time:                " << loopWallSeconds << " s\n";

        if (cfg.maxSteps > 0 && pureDynamicsSeconds > 0.0) {
            std::cout << "Pure dynamics performance:  "
                      << gigaInteractions / pureDynamicsSeconds
                      << " GInteractions/s\n";
        }

        if (cfg.maxSteps > 0 && loopWallSeconds > 0.0) {
            std::cout << "Loop end-to-end performance: "
                      << gigaInteractions / loopWallSeconds
                      << " GInteractions/s\n";
        }

        printValidationQuantities(validation);

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

