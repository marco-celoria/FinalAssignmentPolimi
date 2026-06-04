#include <H5Cpp.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstddef>
#include <array>
#include <cassert>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// CONSTANTS
// ============================================================

constexpr double kForce = 1e-3;
constexpr double eps    = 1e-2;
constexpr double eps2   = eps * eps;

// ============================================================
// UTIL
// ============================================================

inline std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) noexcept {
    return i + j * nx;
}

// ============================================================
// DATA STRUCTURES
// ============================================================

struct Grid {
    std::size_t nx{}, ny{};
    double xs{}, xe{}, ys{}, ye{};
    std::vector<unsigned long long> values;

    void allocate() { values.assign(nx * ny, 0ULL); }
};

struct Particles {
    std::size_t n{};
    std::vector<double> w, x, y, vx, vy;

    void resize(std::size_t N) {
        n = N;
        w.resize(N);
        x.resize(N);
        y.resize(N);
        vx.resize(N);
        vy.resize(N);
    }
};

struct Config {
    std::size_t maxIters{};
    std::size_t maxSteps{};
    std::size_t outputEvery{10};
    double dt{};
};

// ============================================================
// VALIDATION
// ============================================================

void validate(const Grid& g, const Grid& pg, const Config& cfg) {
    if (g.nx < 2 || g.ny < 2 || pg.nx < 2 || pg.ny < 2)
        throw std::runtime_error("Grids must have at least 2 points");

    if (g.xe <= g.xs || g.ye <= g.ys)
        throw std::runtime_error("Invalid generating domain");

    if (pg.xe <= pg.xs || pg.ye <= pg.ys)
        throw std::runtime_error("Invalid particle domain");

    if (cfg.dt <= 0.0)
        throw std::runtime_error("dt must be > 0");

    if (cfg.maxSteps == 0 || cfg.maxIters == 0)
        throw std::runtime_error("Invalid iteration counts");

    if (cfg.outputEvery == 0)
        throw std::runtime_error("outputEvery must be > 0");
}

// ============================================================
// PARSER (STRICT + ROBUST)
// ============================================================

template<typename T>
bool parseLine(std::istream& in, T& value)
{
    std::string line;

    while (std::getline(in, line))
    {
        const auto first = line.find_first_not_of(" \t\r\n");

        if (first == std::string::npos || line[first] == '#')
            continue;

        std::istringstream iss(line);

        if (!(iss >> value))
            throw std::runtime_error("Parse error: " + line);

        iss >> std::ws;

        if (!iss.eof()) {
            if (iss.peek() == '#') return true;
            throw std::runtime_error("Trailing junk: " + line);
        }

        return true;
    }

    return false;
}

// ============================================================
// PHYSICS
// ============================================================

void computeGeneratingField(Grid& g, std::size_t maxIter) {
    const double dx = (g.xe - g.xs) / (g.nx - 1);
    const double dy = (g.ye - g.ys) / (g.ny - 1);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {

            const double ca = g.xs + i * dx;
            const double cb = g.ys + j * dy;

            double za = 0.0, zb = 0.0;

            std::size_t iter = 0;
            for (; iter < maxIter; ++iter) {
                const double a = za * za - zb * zb + ca;
                const double b = 2.0 * za * zb + cb;
                za = a;
                zb = b;
                if (za * za + zb * zb > 4.0) break;
            }

            g.values[idx2D(i, j, g.nx)] = static_cast<unsigned long long>(iter);
        }
    }
}

// ============================================================

Particles generateParticles(const Grid& g, const Grid& pg) {
    Particles P;

    auto vmax = *std::max_element(g.values.begin(), g.values.end());
    auto vmin = *std::min_element(g.values.begin(), g.values.end());
    vmin = (29 * vmax + vmin) / 30;

    const std::size_t count = std::count_if(
        g.values.begin(), g.values.end(),
        [&](unsigned long long v){ return v >= vmin; });

    if (count == 0)
        throw std::runtime_error("No particles generated");

    P.resize(count);

    std::size_t n = 0;

    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {

            const auto v = g.values[idx2D(i, j, g.nx)];
            if (v < vmin) continue;

            P.w[n] = std::max(1.0, 10.0 * static_cast<double>(v));

            const double px = (pg.xe - pg.xs) * static_cast<double>(i) / (g.nx - 1);
            const double py = (pg.ye - pg.ys) * static_cast<double>(j) / (g.ny - 1);

            P.x[n] = pg.xs + px;
            P.y[n] = pg.ys + py;
            P.vx[n] = 0.0;
            P.vy[n] = 0.0;

            ++n;
        }
    }

    assert(n == count);
    return P;
}

// ============================================================

void computeForces(const Particles& P, double* fx, double* fy) {
    const std::size_t N = P.n;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        const double xi = P.x[i];
        const double yi = P.y[i];
        const double wi = P.w[i];

        double fxi = 0.0;
        double fyi = 0.0;

        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) continue;

            const double dx = P.x[j] - xi;
            const double dy = P.y[j] - yi;
            const double r2 = dx * dx + dy * dy + eps2;

            const double invr  = 1.0 / std::sqrt(r2);
            const double invr2 = invr * invr;
            const double invr3 = invr2 * invr;
            const double coeff = kForce * wi * P.w[j] * invr3;

            fxi += coeff * dx;
            fyi += coeff * dy;
        }

        fx[i] = fxi;
        fy[i] = fyi;
    }
}

// ============================================================

void integrateVV(Particles& P,
                 std::vector<double>& fx,
                 std::vector<double>& fy,
                 std::vector<double>& fx_new,
                 std::vector<double>& fy_new,
                 double dt) {
    // Match CUDA order exactly:
    // 1) v += 0.5 * a_old * dt
    // 2) x += v * dt
#pragma omp parallel for
    for (std::size_t i = 0; i < P.n; ++i) {
        assert(P.w[i] > 0.0);
        const double invm = 1.0 / P.w[i];

        P.vx[i] += 0.5 * fx[i] * invm * dt;
        P.vy[i] += 0.5 * fy[i] * invm * dt;

        P.x[i] += P.vx[i] * dt;
        P.y[i] += P.vy[i] * dt;
    }

    // Recompute forces at new positions
    computeForces(P, fx_new.data(), fy_new.data());

    // 3) v += 0.5 * a_new * dt
#pragma omp parallel for
    for (std::size_t i = 0; i < P.n; ++i) {
        assert(P.w[i] > 0.0);
        const double invm = 1.0 / P.w[i];

        P.vx[i] += 0.5 * fx_new[i] * invm * dt;
        P.vy[i] += 0.5 * fy_new[i] * invm * dt;
    }

    fx.swap(fx_new);
    fy.swap(fy_new);
}

// ============================================================
// SCREEN (aligned to final CUDA version)
// ============================================================

void buildScreen(Grid& g, const Particles& P, double wmin, double wr) {
    std::fill(g.values.begin(), g.values.end(), 0ULL);

    if (P.n == 0)
        return;

    const double invdx = (g.xe != g.xs) ? (g.nx - 1) / (g.xe - g.xs) : 0.0;
    const double invdy = (g.ye != g.ys) ? (g.ny - 1) / (g.ye - g.ys) : 0.0;

    for (std::size_t n = 0; n < P.n; ++n) {
        const int ix = std::clamp(static_cast<int>((P.x[n] - g.xs) * invdx),
                                  0, static_cast<int>(g.nx - 1));
        const int iy = std::clamp(static_cast<int>((P.y[n] - g.ys) * invdy),
                                  0, static_cast<int>(g.ny - 1));

        const unsigned long long wp =
            std::clamp(static_cast<unsigned long long>(10.0 * (P.w[n] - wmin) / wr),
                       0ULL, 1000ULL);

        for (int dj = -1; dj <= 1; ++dj) {
            const int jy = iy + dj;
            if (jy < 0 || jy >= static_cast<int>(g.ny)) continue;

            for (int di = -1; di <= 1; ++di) {
                const int jx = ix + di;
                if (jx < 0 || jx >= static_cast<int>(g.nx)) continue;

                g.values[idx2D(static_cast<std::size_t>(jx),
                               static_cast<std::size_t>(jy),
                               g.nx)] += wp;
            }
        }
    }
}

// ============================================================
// HDF5 WRITER (aligned to CUDA version)
// ============================================================

class H5StreamWriter {
public:
    H5StreamWriter(const std::string& name,
                   std::size_t np,
                   std::size_t nx,
                   std::size_t ny,
                   std::size_t chunk_frames = 64)
        : file_(name, H5F_ACC_TRUNC),
          np_(np), nx_(nx), ny_(ny),
          chunk_frames_(chunk_frames),
          capacity_(chunk_frames),
          Pbuf_(np * 2), Vbuf_(np * 2), closed_(false)
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
            static_cast<hsize_t>(chunk_frames_),
            static_cast<hsize_t>(np_),
            2
        };
        prop.setChunk(3, chunk);

        pos_ = file_.createDataSet("/pos", H5::PredType::NATIVE_DOUBLE, space, prop);
        vel_ = file_.createDataSet("/vel", H5::PredType::NATIVE_DOUBLE, space, prop);

        hsize_t gdims[3] = {
            0,
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(nx_)
        };
        hsize_t gmaxdims[3] = {
            H5S_UNLIMITED,
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(nx_)
        };

        H5::DataSpace gspace(3, gdims, gmaxdims);

        H5::DSetCreatPropList gprop;
        hsize_t gchunk[3] = {
            static_cast<hsize_t>(chunk_frames_),
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(nx_)
        };
        gprop.setChunk(3, gchunk);

        // Match CUDA exactly
        grid_ = file_.createDataSet("/screen", H5::PredType::NATIVE_ULLONG, gspace, gprop);

        extendDatasets(capacity_);
    }

    ~H5StreamWriter() noexcept {
        try {
            close();
        } catch (...) {
        }
    }

    void writeFrame(const Particles& P, const Grid& G) {
        if (P.n != np_)
            throw std::runtime_error("Particle size mismatch");

        if (G.nx != nx_ || G.ny != ny_)
            throw std::runtime_error("Grid size mismatch");

        if (currentFrame_ >= capacity_) {
            capacity_ += chunk_frames_;
            extendDatasets(capacity_);
        }

#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < np_; ++i) {
            Pbuf_[2 * i]     = P.x[i];
            Pbuf_[2 * i + 1] = P.y[i];
            Vbuf_[2 * i]     = P.vx[i];
            Vbuf_[2 * i + 1] = P.vy[i];
        }

        write_double(pos_, Pbuf_.data(), currentFrame_, np_, 2);
        write_double(vel_, Vbuf_.data(), currentFrame_, np_, 2);
        write_ull(grid_, G.values.data(), currentFrame_, ny_, nx_);

        ++currentFrame_;
    }

    void close() {
        if (closed_)
            return;

        shrinkToFit();
        file_.flush(H5F_SCOPE_GLOBAL);
        closed_ = true;
    }

private:
    void shrinkToFit() {
        std::array<hsize_t, 3> pos_size = {
            currentFrame_,
            static_cast<hsize_t>(np_),
            2
        };
        std::array<hsize_t, 3> grid_size = {
            currentFrame_,
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(nx_)
        };

        pos_.extend(pos_size.data());
        vel_.extend(pos_size.data());
        grid_.extend(grid_size.data());
    }

    void extendDatasets(hsize_t new_size) {
        std::array<hsize_t, 3> pos_size = {
            new_size,
            static_cast<hsize_t>(np_),
            2
        };
        std::array<hsize_t, 3> grid_size = {
            new_size,
            static_cast<hsize_t>(ny_),
            static_cast<hsize_t>(nx_)
        };

        pos_.extend(pos_size.data());
        vel_.extend(pos_size.data());
        grid_.extend(grid_size.data());
    }

    void write_double(H5::DataSet& ds,
                      const double* data,
                      hsize_t frame,
                      hsize_t dim1,
                      hsize_t dim2)
    {
        H5::DataSpace filespace = ds.getSpace();

        hsize_t start[3] = {frame, 0, 0};
        hsize_t count[3] = {1, dim1, dim2};

        filespace.selectHyperslab(H5S_SELECT_SET, count, start);

        hsize_t memdims[3] = {1, dim1, dim2};
        H5::DataSpace memspace(3, memdims);

        ds.write(data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
    }

    void write_ull(H5::DataSet& ds,
                   const unsigned long long* data,
                   hsize_t frame,
                   hsize_t dim1,
                   hsize_t dim2)
    {
        H5::DataSpace filespace = ds.getSpace();

        hsize_t start[3] = {frame, 0, 0};
        hsize_t count[3] = {1, dim1, dim2};

        filespace.selectHyperslab(H5S_SELECT_SET, count, start);

        hsize_t memdims[3] = {1, dim1, dim2};
        H5::DataSpace memspace(3, memdims);

        ds.write(data, H5::PredType::NATIVE_ULLONG, memspace, filespace);
    }

private:
    H5::H5File file_;
    H5::DataSet pos_, vel_, grid_;

    std::size_t np_, nx_, ny_;
    std::size_t chunk_frames_;
    hsize_t currentFrame_{0};
    hsize_t capacity_;

    std::vector<double> Pbuf_, Vbuf_;
    bool closed_;
};

// ============================================================
// INPUT
// ============================================================

Config readInput(const std::string& file, Grid& g, Grid& pg)
{
    std::ifstream in(file);
    if (!in)
        throw std::runtime_error("Cannot open input");

    Config cfg;

    auto req = [&](auto& x) {
        if (!parseLine(in, x))
            throw std::runtime_error("Unexpected EOF");
    };

    req(g.nx); req(g.ny);
    req(g.xs); req(g.xe);
    req(g.ys); req(g.ye);

    req(pg.nx); req(pg.ny);
    req(pg.xs); req(pg.xe);
    req(pg.ys); req(pg.ye);

    req(cfg.maxIters);
    req(cfg.maxSteps);
    req(cfg.dt);
    req(cfg.outputEvery);

    validate(g, pg, cfg);

    g.allocate();
    pg.allocate();

    return cfg;
}

// ============================================================
// MAIN
// ============================================================

int main(int argc, char** argv) {
    try {
        const std::string input = (argc > 1) ? argv[1] : "Particles.inp";

        Grid gen, screen;
        const Config cfg = readInput(input, gen, screen);

        computeGeneratingField(gen, cfg.maxIters);
        Particles P = generateParticles(gen, screen);

        H5StreamWriter h5("particles.h5", P.n, screen.nx, screen.ny);

        std::vector<double> fx(P.n), fy(P.n), fx_new(P.n), fy_new(P.n);

        // Match CUDA: compute once before timing / simulation loop
        computeForces(P, fx.data(), fy.data());

        // Match CUDA: precompute once (weights do not change)
        const double wmin = *std::min_element(P.w.begin(), P.w.end());
        const double wmax = *std::max_element(P.w.begin(), P.w.end());
        const double wr   = std::max(wmax - wmin, 1.0);

        const auto t0 = std::chrono::high_resolution_clock::now();

        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            if (step % cfg.outputEvery == 0) {
                buildScreen(screen, P, wmin, wr);
                h5.writeFrame(P, screen);
            }

            integrateVV(P, fx, fy, fx_new, fy_new, cfg.dt);
        }

        const auto t1 = std::chrono::high_resolution_clock::now();

        // Match CUDA final catch-up logic
        if ((cfg.maxSteps - 1) % cfg.outputEvery != 0) {
            buildScreen(screen, P, wmin, wr);
            h5.writeFrame(P, screen);
        }

        h5.close();

        std::cout << "Simulation completed successfully.\n";

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double interactions = double(P.n) * double(P.n - 1) * cfg.maxSteps;
        const double giga_interactions = interactions / 1e9;

        std::cout << "Total time: " << elapsed_s << " s\n";
        std::cout << "Time per step: " << elapsed_s / cfg.maxSteps << " s\n";
        std::cout << "Particles: " << P.n << "\n";
        std::cout << "Performance: " << giga_interactions / elapsed_s << " GInteractions/s\n";

        return EXIT_SUCCESS;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
