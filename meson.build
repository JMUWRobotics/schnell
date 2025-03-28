project('schnell', 'cpp', version: '0.1')

cmake = import('cmake')

cctag_opts = cmake.subproject_options()
cctag_opts.add_cmake_defines(
    {
        'CCTAG_WITH_CUDA': false,
        'BUILD_SHARED_LIBS': false,
        'CCTAG_BUILD_APPS': false,
        'CCTAG_BUILD_TESTS': false,
        'CCTAG_ENABLE_SIMD_AVX2': true,
        'CMAKE_BUILD_TYPE': 'Release',
    },
)
cctag_opts.set_install(false)
cctag = cmake.subproject('CCTag', options: cctag_opts)

ceres_dep = dependency('Ceres', required: false, not_found_message: 'falling back to subproject')

if not ceres_dep.found()
    ceres_opts = cmake.subproject_options()
    ceres_opts.add_cmake_defines(
        {
            'BUILD_BENCHMARKS': false,
            'BUILD_EXAMPLES': false,
            'BUILD_TESTING': false,
            'MINIGLOG': true,
            'USE_CUDA': false,
            'GFLAGS': false,
        },
    )
    ceres_opts.set_install(false)
    ceres = cmake.subproject('ceres-solver', options: ceres_opts)
    ceres_dep = ceres.dependency('ceres')
endif

# opencv...
add_project_arguments(
    '-Wno-deprecated-enum-enum-conversion',
    '-Wno-old-style-cast',
    '-Wno-abi-tag',
    '-Wno-effc++',
    '-Wno-useless-cast',
    '-Wno-noexcept',
    '-Wno-zero-as-null-pointer-constant',
    language: 'cpp',
)

executable(
    'schnell',
    'schnell.cpp',
    dependencies: [
        dependency('eigen3'),
        dependency('opencv4'),
        dependency('apriltag'),
        dependency('nlohmann_json'),
        dependency('fmt', version: '>=11.1.1'),
        dependency('cxxopts'),
        ceres_dep,
        cctag.dependency('CCTag'),
    ],
    override_options: ['cpp_std=c++20', 'warning_level=3'],
)