#!/bin/sh

CXXFLAGS="-Wno-error=cpp"
CXXFLAGS="${CXXFLAGS} -march=haswell -mtune=skylake-avx512"
export CXXFLAGS

nohup ./build.sh				\
	--no_telemetry				\
	--config RelWithDebInfo			\
	--use_cuda				\
	--cuda_home=/usr/local/cuda-13.3	\
	--cudnn_home=/usr			\
	--nvcc_threads 8			\
	--build_shared_lib			\
	--parallel 16				\
	--cmake_extra_defines			\
		CMAKE_INSTALL_PREFIX=/opt/vespa-deps			\
		CMAKE_CXX_STANDARD=20					\
		CMAKE_CXX_COMPILER_LAUNCHER=ccache			\
		CMAKE_CUDA_COMPILER_LAUNCHER=+ccache			\
		CMAKE_CUDA_ARCHITECTURES='89'				\
		CMAKE_CUDA_FLAGS='--std c++20 --diag-suppress 68 --jobserver'	\
		onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON		\
	> b.log 2>&1 </dev/null &
