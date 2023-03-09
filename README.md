# Visual Odometry Library
Library for performing RGB-D Visual Odometry

Currently supported approaches: 
- Robust Dense Visual Odometry [1]

This project contain some external modules as git submodules, so after clonning the repository one must run:
```bash
git submodule update --init --recursive
```

## C++ build
To buld the library `vo`, on project's root dir run these commands
```bash
mkdir build && cd build
cmake ../
cmake --build . {-DUSE_GPU=ON, -DBUILD_PYTHON=OFF, -DUSE_OMP=OFF}
```
`NOTE: (`{-DXX=ON}` are optional flags)`

## Pyhton Bindings
This repository contains Python bindings (tested with `Python3.7`) thanks to [pybind11](https://github.com/pybind/pybind11). One can install this library's Python wrappers `pyvo`
by running (on project's root):
```bash
pip install setup.py
```

## Example
There is a Python CLI [script](examples/test_dvo.py) to test the approach, supported benchmark are:
- [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) RGB-D benchmarks
- Custom benchmark format (TODO: Explain)

This scripts requirements are specified in [here](examples/requirements-examples.txt).

To run an evaluation (from project's root):
```bash
python3 examples/test_dvo.py {tum-fr1, test} -d /path/to/benchmark/dir -c /path/to/dvo/config.yaml -i /path/to/camera/intrinsics.yaml {-v} {-s 100}
```

An example of a DVO config file located [here](examples/dvo_config.yaml) and for the camera intrinsics file [here](examples/dvo_config.yaml). The optional arguments:
- `-v`: Boolean flag to display the estimated trajectory vs groundtruth
- `-s X`: Optional option to use the first `X` samples of the benchmark. If not set, the complete benchmark is computed


## References

[1] C. Kerl, J. Sturm, and D. Cremers, “Robust odometry estimation for
rgb-d cameras,” in International Conference on Robotics and Automation
(ICRA), May 2013.
