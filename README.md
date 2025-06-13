# schnell

Implementation of the paper *Low-Constraint Ray Optimization for Photogrammetric Shallow Water Bathymetry* to be published in *3D Underwater Mapping from Above and Below*, Vienna, 2025.

# compilation

you will need `eigen3`, `opencv4.x`, [`apriltag`](https://github.com/AprilRobotics/apriltag), `nlohmann_json`, [`fmt`](https://github.com/fmtlib/fmt), and [`cxxopts`](https://github.com/jarro2783/cxxopts) to build the C++ optimizer. CCTag and ceres-solver come bundled as subprojects. For the python wrapper `viswrap.py`, you need [`vispy`](https://github.com/vispy/vispy) and `numpy`.

```sh
meson setup builddir --buildtype release
meson compile -C builddir
```

# running

```sh
python3 viswrap.py path/to/data --lone=0.04 --solve
```

## data input

where `path/to/data` contains each camera's images, saved as `0.png` for camera 0, `1.png` for camera 1 etc. with optional masks as `0_mask.png`. Additionally, for each pair of cameras, (0,1), (0,2), (1,2), etc. there needs to be a file `0-to-1.json`, `0-to-2.json`, etc. that contains calibration matrices `K{1,2}`, distortion coefficients `d{1,2}` and the relative orientation `R,T` of camera 2. `R,T` are defined according to OpenCV's `stereoCalibrate()` (4.11.0).

## optimizer parameters

optionally, `--abcd` or `--point` and `--perpvec` can be used to manually supply an initial guess for the water plane, with `abcd` as the Hesse normal form or `point` as a 3d point on the plane, and `perpvec` as its perpendicular vector. `--solve` will make the optimizer try to solve the problem. without it, only the initial guess is displayed. for robustness, `--lone` supplies a floating point value for SoftL1-loss robustification on the backrefractive disparity.

## features

for feature detection, three methods are available: `--cctag`, `--sift` or apriltag (default). CCTag and apriltag will require fiducials to be present in the underwater scene.
