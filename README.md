# ottobot
Autonomous mobile robot project

## Initial set up

After installing ROS and librealsense go to the [initial set up](https://github.com/pfontana96/ottobot/wiki/Initial-setup) wiki for configuring the workspace.

## C++ build
On project's root dir run these commands
```bash
mkdir build && cd build
cmake ../
cmake --build . {-DUSE_GPU=ON} {-DBUILD_TESTS=ON}
```
`NOTE: (`{-DXX=ON}` are optional flags used for building tests and enabling GPU support)`