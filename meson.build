project('schnell', 'cpp', version: '0.1')

cmake = import('cmake')
cctag_opts = cmake.subproject_options()
cctag_opts.add_cmake_defines({
    'CCTAG_WITH_CUDA': false,
    'BUILD_SHARED_LIBS': false,
    'CCTAG_BUILD_APPS': false,
    'CCTAG_BUILD_TESTS': false,
    'CCTAG_ENABLE_SIMD_AVX2': true,
    'CMAKE_BUILD_TYPE': 'Release'
})
cctag = cmake.subproject('CCTag', options: cctag_opts)

# opencv...
add_project_arguments(
    '-Wno-deprecated-enum-enum-conversion',
    '-Wno-old-style-cast',
    '-Wno-abi-tag',
    '-Wno-effc++',
    '-Wno-useless-cast',
    '-Wno-noexcept',
    '-Wno-zero-as-null-pointer-constant',
    language: 'cpp'
)

executable('schnell', 
    'schnell.cpp',
    dependencies: [
        dependency('Ceres'),
        dependency('eigen3'),
        dependency('opencv4'),
        dependency('apriltag'),
        cctag.dependency('CCTag')
    ],
    override_options: ['cpp_std=c++23', 'warning_level=3']
)
