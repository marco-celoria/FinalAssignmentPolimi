#include <H5Cpp.h>

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
#include <system_error>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


// ============================================================
// CONSTANTS
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

    long long raw_max_iters   = 0;
    long long raw_max_steps   = 0;
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

    g.nx  = static_cast<std::size_t>(raw_g_nx);
    g.ny  = static_cast<std::size_t>(raw_g_ny);
    pg.nx = static_cast<std::size_t>(raw_pg_nx);
    pg.ny = static_cast<std::size_t>(raw_pg_ny);

    cfg.maxIters   = static_cast<std::size_t>(raw_max_iters);
    cfg.maxSteps   = static_cast<std::size_t>(raw_max_steps);
    cfg.outputEvery = static_cast<std::size_t>(raw_output_every);

    g.allocate();
    pg.allocate();

    return cfg;
}


// ============================================================
// MANDELBROT FIELD GENERATION
// ============================================================

void computeGeneratingField(Grid& g, std::size_t maxIter) {
    if (g.values.empty()) {
        throw std::runtime_error("computeGeneratingField: empty grid");
    }

    const double dx = (g.xe - g.xs) / static_cast<double>(g.nx - 1);
    const double dy = (g.ye - g.ys) / static_cast<double>(g.ny - 1);

#pragma omp parallel for collapse(2) schedule(dynamic)
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

            g.values[idx2D(i, j, g.nx)] =
                static_cast<unsigned long long>(iter);
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
// FORCE COMPUTATION
// ============================================================

void computeForces(const Particles& P, double* fx, double* fy) {
    const std::size_t N = P.n;

    if (!fx || !fy) {
        throw std::runtime_error("computeForces: null output pointer");
    }

    const double* const x = P.x.data();
    const double* const y = P.y.data();
    const double* const w = P.w.data();

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double wi = w[i];

        double fxi = 0.0;
        double fyi = 0.0;

#pragma omp simd reduction(+:fxi,fyi)
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

    if (fx.size() != N || fy.size() != N ||
        fx_new.size() != N || fy_new.size() != N) {
        throw std::runtime_error("integrateVV: force array size mismatch");
    }

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        assert(P.w[i] > 0.0);

        const double invm = 1.0 / P.w[i];

        P.vx[i] += 0.5 * fx[i] * invm * dt;
        P.vy[i] += 0.5 * fy[i] * invm * dt;

        P.x[i] += P.vx[i] * dt;
        P.y[i] += P.vy[i] * dt;
    }

    computeForces(P, fx_new.data(), fy_new.data());

#pragma omp parallel for schedule(static)
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
// ============================================================

void buildScreen(Grid& screen, const Particles& P, double wmin, double wr) {
    if (screen.values.empty()) {
        throw std::runtime_error("buildScreen: empty screen grid");
    }

    if (wr <= 0.0) {
        throw std::runtime_error("buildScreen: invalid weight range");
    }

    std::fill(screen.values.begin(), screen.values.end(), 0ULL);

    if (P.n == 0) {
        return;
    }

    const double invdx = (screen.xe != screen.xs)
        ? static_cast<double>(screen.nx - 1) / (screen.xe - screen.xs)
        : 0.0;

    const double invdy = (screen.ye != screen.ys)
        ? static_cast<double>(screen.ny - 1) / (screen.ye - screen.ys)
        : 0.0;

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

        const unsigned long long wp =
            static_cast<unsigned long long>(wp_i);

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
// HDF5 STREAM WRITER
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
        }
    }

    H5StreamWriter(const H5StreamWriter&) = delete;
    H5StreamWriter& operator=(const H5StreamWriter&) = delete;

    void writeFrame(
        std::size_t stepNumber,
        const Particles& P,
        const Grid& screen
    ) {
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

#pragma omp parallel for schedule(static)
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

private:
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

        std::cout << "Input file:                 " << inputFile << "\n";
        std::cout << "HDF5 output:                " << outputFile << "\n";
        std::cout << "Generating grid:            " << gen.nx << " x " << gen.ny << "\n";
        std::cout << "Screen grid:                " << screen.nx << " x " << screen.ny << "\n";
        std::cout << "Max iterations:             " << cfg.maxIters << "\n";
        std::cout << "Steps:                      " << cfg.maxSteps << "\n";
        std::cout << "Output every:               " << cfg.outputEvery << "\n";

#ifdef _OPENMP
        std::cout << "OpenMP threads:             " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP:                     disabled at compile time\n";
#endif

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
        const auto [wminIt, wmaxIt] =
            std::minmax_element(P.w.begin(), P.w.end());

        const double wmin = *wminIt;
        const double wmax = *wmaxIt;
        const double wr   = std::max(wmax - wmin, 1.0);

        // ----------------------------------------------------
        // 6. HDF5 writer
        // ----------------------------------------------------
        H5StreamWriter h5(
            outputFile,
            N,
            screen.nx,
            screen.ny
        );

        std::size_t outputFrames = 0;

        bool hasLastWrittenStep = false;
        std::size_t lastWrittenStep = 0;

        auto writeOutputFrame = [&](std::size_t step) {
            if (hasLastWrittenStep && step == lastWrittenStep) {
                return;
            }

            buildScreen(screen, P, wmin, wr);

            h5.writeFrame(
                step,
                P,
                screen
            );

            hasLastWrittenStep = true;
            lastWrittenStep = step;

            ++outputFrames;
        };

        // ----------------------------------------------------
        // 7. Simulation loop
        // ----------------------------------------------------
        const auto loopStart = std::chrono::steady_clock::now();

        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            if ((step % cfg.outputEvery) == 0) {
                writeOutputFrame(step);
            }

            integrateVV(
                P,
                fx,
                fy,
                fx_new,
                fy_new,
                cfg.dt
            );
        }

        // Always save the final state after cfg.maxSteps integrations.
        // The writeOutputFrame guard prevents accidental duplication.
        writeOutputFrame(cfg.maxSteps);

        h5.close();

        const auto loopStop = std::chrono::steady_clock::now();

        const double loopWallSeconds =
            std::chrono::duration<double>(loopStop - loopStart).count();

        // ----------------------------------------------------
        // 8. Reporting
        // ----------------------------------------------------
        const double interactions =
            static_cast<double>(N)
            * static_cast<double>(N - 1)
            * static_cast<double>(cfg.maxSteps);

        const double gigaInteractions =
            interactions / 1.0e9;

        std::cout << "Simulation completed successfully.\n";
        std::cout << "Output frames:              " << outputFrames << "\n";
        std::cout << "Mandelbrot CPU time:        " << mandelSeconds << " s\n";
        std::cout << "Particle generation wall:   " << particleGenerationSeconds << " s\n";
        std::cout << "Initial force CPU time:     " << initForceSeconds << " s\n";
        std::cout << "Loop wall time incl. HDF5:  " << loopWallSeconds << " s\n";

        if (cfg.maxSteps > 0 && loopWallSeconds > 0.0) {
            std::cout << "End-to-end loop performance:"
                      << " " << gigaInteractions / loopWallSeconds
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
