# 1. Create and enter a build directory
mkdir build && cd build

# 2. Configure the project (Release mode ensures -O3 optimization is active)
cmake -DCMAKE_BUILD_TYPE=Release ..

# 3. Compile the executable
cmake --build . --config Release
