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
#include <cuda_runtime.h>

// ============================================================
// MACROS & CONSTANTS
// ============================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr double kForce = 1e-3;
constexpr double eps    = 1e-2;
constexpr double eps2   = eps * eps;
constexpr int BLOCK_SIZE = 256;

// ============================================================
// RAII DEVICE BUFFER
// ============================================================

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(size_t n) : size_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }

    ~DeviceBuffer() noexcept {
        if (ptr_) cudaFree(ptr_);
    }

    // Disable copying to prevent double-free scenarios
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Enable moving to support safe stack swapping
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_{};
    size_t size_{};
};

// ============================================================
// UTIL
// ============================================================

inline __host__ __device__ std::size_t idx2D(std::size_t i, std::size_t j, std::size_t nx) {
    return i + j * nx;
}

template <typename T>
inline __device__ T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}

// ============================================================
// DATA STRUCTURES
// ============================================================

struct Grid {
    std::size_t nx{}, ny{};
    double xs{}, xe{}, ys{}, ye{};
    std::vector<unsigned long long> values; 

    void allocate() { values.assign(nx * ny, 0); }
};

struct Particles {
    std::size_t n{};
    std::vector<double> w, x, y, vx, vy;

    void resize(std::size_t N) {
        n = N;
        w.resize(N); x.resize(N); y.resize(N);
        vx.resize(N); vy.resize(N);
    }
};

struct Config {
    std::size_t maxIters{};
    std::size_t maxSteps{};
    std::size_t outputEvery{10};
    double dt{};
};

// ============================================================
// VALIDATION & PARSER
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

template<typename T>
bool parseLine(std::istream& in, T& value) {
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream iss(line);
        if (!(iss >> value)) throw std::runtime_error("Parse error: " + line);
        iss >> std::ws;
        if (!iss.eof()) {
            if (iss.peek() == '#') return true;
            throw std::runtime_error("Trailing junk: " + line);
        }
        return true;
    }
    return false;
}

Config readInput(const std::string& file, Grid& g, Grid& pg) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("Cannot open input");

    Config cfg;
    auto req = [&](auto& x){
        if (!parseLine(in,x)) throw std::runtime_error("Unexpected EOF");
    };

    req(g.nx); req(g.ny);
    req(g.xs); req(g.xe);
    req(g.ys); req(g.ye);
    req(pg.nx); req(pg.ny);
    req(pg.xs); req(pg.xe);
    req(pg.ys); req(pg.ye);
    req(cfg.maxIters); req(cfg.maxSteps); req(cfg.dt); req(cfg.outputEvery);

    validate(g, pg, cfg);
    g.allocate(); pg.allocate();
    return cfg;
}

// ============================================================
// CUDA KERNELS
// ============================================================

__global__ void mandelbrotKernel(unsigned long long* d_values, std::size_t nx, std::size_t ny,
                                 double xs, double xe, double ys, double ye, std::size_t maxIter) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        double dx = (xe - xs) / (nx - 1);
        double dy = (ye - ys) / (ny - 1);
        double ca = xs + i * dx;
        double cb = ys + j * dy;

        double za = 0.0, zb = 0.0;
        std::size_t iter = 0;
        for (; iter < maxIter; ++iter) {
            double a = za * za - zb * zb + ca;
            double b = 2.0 * za * zb + cb;
            za = a; zb = b;
            if (za * za + zb * zb > 4.0) break;
        }
        d_values[idx2D(i, j, nx)] = static_cast<unsigned long long>(iter);
    }
}

__global__ void computeForcesKernel(const double* __restrict__ x, const double* __restrict__ y, 
                                    const double* __restrict__ w, double* __restrict__ fx, 
                                    double* __restrict__ fy, std::size_t N) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double xi = x[i], yi = y[i], wi = w[i];
    double fxi = 0.0, fyi = 0.0;

    for (std::size_t j = 0; j < N; ++j) {
        if (i == j) continue;
        double dx = x[j] - xi;
        double dy = y[j] - yi;
        double r2 = dx*dx + dy*dy + eps2;

        //double invr = rsqrt(r2); 
        double invr = 1.0 / sqrt(r2);
        double invr2 = invr * invr;
        double invr3 = invr2 * invr;
        double coeff = kForce * wi * w[j] * invr3;

        fxi += coeff * dx;
        fyi += coeff * dy;
    }

    fx[i] = fxi;
    fy[i] = fyi;
}



template<int BLOCK_SIZE>
__global__ void computeForcesKernelTiled(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ w,
    double* __restrict__ fx,
    double* __restrict__ fy,
    std::size_t N)
{
    assert(blockDim.x == BLOCK_SIZE);
    __shared__ double sh_x[BLOCK_SIZE];
    __shared__ double sh_y[BLOCK_SIZE];
    __shared__ double sh_w[BLOCK_SIZE];

    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (i < N);

    double xi = active ? x[i] : 0.0;
    double yi = active ? y[i] : 0.0;
    double wi = active ? w[i] : 0.0;

    double fxi = 0.0;
    double fyi = 0.0;

    std::size_t tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        std::size_t j = tile * BLOCK_SIZE + threadIdx.x;

        if (j < N) {
            sh_x[threadIdx.x] = x[j];
            sh_y[threadIdx.x] = y[j];
            sh_w[threadIdx.x] = w[j];
        } else {
            sh_x[threadIdx.x] = 0.0;
            sh_y[threadIdx.x] = 0.0;
            sh_w[threadIdx.x] = 0.0;
        }

        __syncthreads();

        if (active) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                std::size_t global_j = tile * BLOCK_SIZE + k;
                if (global_j >= N || global_j == i) continue;

                double dx = sh_x[k] - xi;
                double dy = sh_y[k] - yi;
                double r2 = dx * dx + dy * dy + eps2;

                double invr = 1.0 / sqrt(r2);
                double invr2 = invr * invr;
                double invr3 = invr2 * invr;

                double coeff = kForce * wi * sh_w[k] * invr3;
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


__global__ void updatePositionKernel(double* x, double* y, double* vx, double* vy, 
                                     const double* fx, const double* fy, const double* w, 
                                     double dt, std::size_t N) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double invm = 1.0 / w[i];
    
    // v += 0.5 * a_old * dt
    vx[i] += 0.5 * fx[i] * invm * dt;
    vy[i] += 0.5 * fy[i] * invm * dt;
    
    // x += v * dt
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
}

__global__ void updateVelocityKernel(double* vx, double* vy, const double* fx_new, 
                                     const double* fy_new, const double* w, 
                                     double dt, std::size_t N) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double invm = 1.0 / w[i];
    
    // v += 0.5 * a_new * dt
    vx[i] += 0.5 * fx_new[i] * invm * dt;
    vy[i] += 0.5 * fy_new[i] * invm * dt;
}

__global__ void buildScreenKernel(unsigned long long* d_screen, const double* x, const double* y, 
                                  const double* w, std::size_t nx, std::size_t ny, 
                                  double xs, double xe, double ys, double ye, 
                                  double wmin, double wr, std::size_t N) {
    std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double invdx = (xe != xs) ? (nx - 1) / (xe - xs) : 0.0;
    double invdy = (ye != ys) ? (ny - 1) / (ye - ys) : 0.0;

    int ix = clamp(static_cast<int>((x[n] - xs) * invdx), 0, static_cast<int>(nx - 1));
    int iy = clamp(static_cast<int>((y[n] - ys) * invdy), 0, static_cast<int>(ny - 1));
    unsigned long long wp = clamp(static_cast<unsigned long long>(10.0 * (w[n] - wmin) / wr), 0ULL, 1000ULL);
    for (int dj = -1; dj <= 1; ++dj) {
        int jy = iy + dj;
        if (jy < 0 || jy >= static_cast<int>(ny)) continue;
        for (int di = -1; di <= 1; ++di) {
            int jx = ix + di;
            if (jx < 0 || jx >= static_cast<int>(nx)) continue;
            atomicAdd(&d_screen[jx + jy * nx], wp);
        }
    }
}

// ============================================================
// CPU LOGIC
// ============================================================

Particles generateParticles(const Grid& g, const Grid& pg) {
    Particles P;
    auto vmax = *std::max_element(g.values.begin(), g.values.end());
    auto vmin = *std::min_element(g.values.begin(), g.values.end());
    vmin = (29 * vmax + vmin) / 30;

    std::size_t count = std::count_if(g.values.begin(), g.values.end(),
                                      [&](auto v){ return v >= vmin; });
    if (count == 0) throw std::runtime_error("No particles generated");

    P.resize(count);
    std::size_t n = 0;
    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            auto v = g.values[idx2D(i,j,g.nx)];
            if (v < vmin) continue;

            P.w[n] = std::max(1.0, 10.0 * static_cast<double>(v));
            double px = (pg.xe - pg.xs) * static_cast<double>(i) / (g.nx - 1);
            double py = (pg.ye - pg.ys) * static_cast<double>(j) / (g.ny - 1);

            P.x[n] = pg.xs + px; P.y[n] = pg.ys + py;
            P.vx[n] = 0.0; P.vy[n] = 0.0;
            ++n;
        }
    }
    assert(n == count);
    return P;
}

// ============================================================
// HDF5 WRITER
// ============================================================

class H5StreamWriter {
public:
    H5StreamWriter(const std::string& name, std::size_t np, std::size_t nx, std::size_t ny, std::size_t chunk_frames = 64)
        : file_(name, H5F_ACC_TRUNC), np_(np), nx_(nx), ny_(ny), chunk_frames_(chunk_frames),
          capacity_(chunk_frames), Pbuf_(np * 2), Vbuf_(np * 2), closed_(false)
    {
        hsize_t dims[3]    = {0, np_, 2};
        hsize_t maxdims[3] = {H5S_UNLIMITED, np_, 2};
        H5::DataSpace space(3, dims, maxdims);
        H5::DSetCreatPropList prop;
        hsize_t chunk[3] = {chunk_frames_, np_, 2};
        prop.setChunk(3, chunk);

        pos_ = file_.createDataSet("/pos", H5::PredType::NATIVE_DOUBLE, space, prop);
        vel_ = file_.createDataSet("/vel", H5::PredType::NATIVE_DOUBLE, space, prop);

        hsize_t gdims[3]    = {0, ny_, nx_};
        hsize_t gmaxdims[3] = {H5S_UNLIMITED, ny_, nx_};
        H5::DataSpace gspace(3, gdims, gmaxdims);
        H5::DSetCreatPropList gprop;
        hsize_t gchunk[3] = {chunk_frames_, ny_, nx_};
        gprop.setChunk(3, gchunk);

        grid_ = file_.createDataSet("/screen", H5::PredType::NATIVE_ULLONG, gspace, gprop);

        extendDatasets(capacity_);
    }

    ~H5StreamWriter() noexcept { try { close(); } catch (...) {} }

    void writeFrame(const Particles& P, const Grid& G) {
        if (P.n != np_) throw std::runtime_error("Particle size mismatch");
        if (G.nx != nx_ || G.ny != ny_) throw std::runtime_error("Grid size mismatch");

        if (currentFrame_ >= capacity_) {
            capacity_ += chunk_frames_;
            extendDatasets(capacity_);
        }

        for (std::size_t i = 0; i < np_; ++i) {
            Pbuf_[2*i]   = P.x[i];  Pbuf_[2*i+1] = P.y[i];
            Vbuf_[2*i]   = P.vx[i]; Vbuf_[2*i+1] = P.vy[i];
        }

        write_double(pos_, Pbuf_.data(), currentFrame_, np_, 2);
        write_double(vel_, Vbuf_.data(), currentFrame_, np_, 2);
        write_hsize(grid_, G.values.data(), currentFrame_, ny_, nx_);
        ++currentFrame_;
    }

    void close() {
        if (closed_) return;
        shrinkToFit();
        file_.flush(H5F_SCOPE_GLOBAL);
        closed_ = true;
    }

private:
    void shrinkToFit() {
        std::array<hsize_t,3> pos_size  = {currentFrame_, np_, 2};
        std::array<hsize_t,3> grid_size = {currentFrame_, ny_, nx_};
        pos_.extend(pos_size.data());
        vel_.extend(pos_size.data());
        grid_.extend(grid_size.data());
    }

    void extendDatasets(hsize_t new_size) {
        std::array<hsize_t,3> pos_size  = {new_size, np_, 2};
        std::array<hsize_t,3> grid_size = {new_size, ny_, nx_};
        pos_.extend(pos_size.data());
        vel_.extend(pos_size.data());
        grid_.extend(grid_size.data());
    }

    void write_double(H5::DataSet& ds, const double* data, hsize_t frame, hsize_t dim1, hsize_t dim2) {
        H5::DataSpace filespace = ds.getSpace();
        hsize_t start[3] = {frame, 0, 0}; hsize_t count[3] = {1, dim1, dim2};
        filespace.selectHyperslab(H5S_SELECT_SET, count, start);
        hsize_t memdims[3] = {1, dim1, dim2};
        H5::DataSpace memspace(3, memdims);
        ds.write(data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
    }

    void write_hsize(H5::DataSet& ds, const unsigned long long* data, hsize_t frame, hsize_t dim1, hsize_t dim2) {
        H5::DataSpace filespace = ds.getSpace();
        hsize_t start[3] = {frame, 0, 0}; hsize_t count[3] = {1, dim1, dim2};
        filespace.selectHyperslab(H5S_SELECT_SET, count, start);
        hsize_t memdims[3] = {1, dim1, dim2};
        H5::DataSpace memspace(3, memdims);
        ds.write(data, H5::PredType::NATIVE_ULLONG, memspace, filespace);
    }

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
// MAIN
// ============================================================

int main(int argc, char** argv) {
    try {
        std::string input = (argc > 1) ? argv[1] : "Particles.inp";
        Grid gen, screen;
        Config cfg = readInput(input, gen, screen);

        // --- 1. Compute Generating Field on Device ---
        DeviceBuffer<unsigned long long> d_gen_values(gen.nx * gen.ny);
        std::size_t gen_bytes = gen.nx * gen.ny * sizeof(unsigned long long);

        dim3 blockGen(16, 16);
        dim3 gridGen((gen.nx + blockGen.x - 1) / blockGen.x, (gen.ny + blockGen.y - 1) / blockGen.y);
        
        mandelbrotKernel<<<gridGen, blockGen>>>(d_gen_values.get(), gen.nx, gen.ny, gen.xs, gen.xe, gen.ys, gen.ye, cfg.maxIters);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gen.values.data(), d_gen_values.get(), gen_bytes, cudaMemcpyDeviceToHost));

        Particles P = generateParticles(gen, screen);
        std::size_t N = P.n;

        H5StreamWriter h5("particles.h5", N, screen.nx, screen.ny);

        double wmin = *std::min_element(P.w.begin(), P.w.end());
        double wmax = *std::max_element(P.w.begin(), P.w.end());
        double wr = std::max(wmax - wmin, 1.0);

        // --- 2. Allocate RAII Device Memory Buffer Wrappers ---
        DeviceBuffer<double> d_x(N);             DeviceBuffer<double> d_y(N);
        DeviceBuffer<double> d_vx(N);            DeviceBuffer<double> d_vy(N);
        DeviceBuffer<double> d_w(N);
        DeviceBuffer<double> d_fx(N);            DeviceBuffer<double> d_fy(N);
        DeviceBuffer<double> d_fx_new(N);        DeviceBuffer<double> d_fy_new(N);
        DeviceBuffer<unsigned long long> d_screen(screen.nx * screen.ny);

        std::size_t p_bytes = N * sizeof(double);
        std::size_t screen_bytes = screen.nx * screen.ny * sizeof(unsigned long long);

        CUDA_CHECK(cudaMemcpy(d_x.get(), P.x.data(), p_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y.get(), P.y.data(), p_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx.get(), P.vx.data(), p_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy.get(), P.vy.data(), p_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w.get(), P.w.data(), p_bytes, cudaMemcpyHostToDevice));

        // --- 3. Execution Configurations ---
        constexpr unsigned int threadsPerBlock = BLOCK_SIZE;
        unsigned int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Calculate Initial Forces
        //computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x.get(), d_y.get(), d_w.get(), d_fx.get(), d_fy.get(), N);
        computeForcesKernelTiled<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(d_x.get(), d_y.get(), d_w.get(), d_fx.get(), d_fy.get(), N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // --- 4. Simulation Loop ---
        for (std::size_t step = 0; step < cfg.maxSteps; ++step) {
            
            if (step % cfg.outputEvery == 0) {
                CUDA_CHECK(cudaMemset(d_screen.get(), 0, screen_bytes));
                
                buildScreenKernel<<<blocksPerGrid, threadsPerBlock>>>(
                    d_screen.get(), d_x.get(), d_y.get(), d_w.get(), screen.nx, screen.ny, 
                    screen.xs, screen.xe, screen.ys, screen.ye, wmin, wr, N
                );
                CUDA_CHECK(cudaGetLastError());
                
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(screen.values.data(), d_screen.get(), screen_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(P.x.data(), d_x.get(), p_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(P.y.data(), d_y.get(), p_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(P.vx.data(), d_vx.get(), p_bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(P.vy.data(), d_vy.get(), p_bytes, cudaMemcpyDeviceToHost));
                
                h5.writeFrame(P, screen);
            }

            // Integrate Position
            updatePositionKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_x.get(), d_y.get(), d_vx.get(), d_vy.get(), d_fx.get(), d_fy.get(), d_w.get(), cfg.dt, N);
            CUDA_CHECK(cudaGetLastError());

            // Compute New Forces
            //computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x.get(), d_y.get(), d_w.get(), d_fx_new.get(), d_fy_new.get(), N);
            computeForcesKernelTiled<BLOCK_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(d_x.get(), d_y.get(), d_w.get(), d_fx_new.get(), d_fy_new.get(), N);

            CUDA_CHECK(cudaGetLastError());

            // Integrate Velocity
            updateVelocityKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_vx.get(), d_vy.get(), d_fx_new.get(), d_fy_new.get(), d_w.get(), cfg.dt, N);
            CUDA_CHECK(cudaGetLastError());
            // CUDA_CHECK(cudaDeviceSynchronize());
            // Safe RAII Stack Swapping via Move Semantics
            std::swap(d_fx, d_fx_new);
            std::swap(d_fy, d_fy_new);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Final catch-up window frame mapping
        if ((cfg.maxSteps - 1) % cfg.outputEvery != 0) {
            CUDA_CHECK(cudaMemset(d_screen.get(), 0, screen_bytes));
            
            buildScreenKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_screen.get(), d_x.get(), d_y.get(), d_w.get(), screen.nx, screen.ny, 
                screen.xs, screen.xe, screen.ys, screen.ye, wmin, wr, N
            );
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(screen.values.data(), d_screen.get(), screen_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(P.x.data(), d_x.get(), p_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(P.y.data(), d_y.get(), p_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(P.vx.data(), d_vx.get(), p_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(P.vy.data(), d_vy.get(), p_bytes, cudaMemcpyDeviceToHost));
            
            h5.writeFrame(P, screen);
        }

        h5.close();
        std::cout << "Simulation completed successfully.\n";
        double interactions = double(P.n) * double(P.n - 1) * cfg.maxSteps;
        double giga_interactions = interactions / 1e9;
        float elapsed_s = elapsed_ms / 1000.0;
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
