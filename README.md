# Visual Odometry Library
Library for performing Visual Odometry

## C++ build
On project's root dir run these commands
```bash
mkdir build && cd build
cmake ../
cmake --build . {-DUSE_GPU=ON}
```
`NOTE: (`{-DXX=ON}` are optional flags used for building tests and enabling GPU support)`