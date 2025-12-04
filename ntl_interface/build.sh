# prerequisits: cmake, gcc or clang, python3-dev, googletest, pybind11, libntl

cmake -B build
cmake --build build --config Release
ctest --test-dir build

# note that Python requires a full path on any linked files.
# soft-link all the so files from the build folder to the parent folder:
ln -sf $(pwd)/build/*.so ..
