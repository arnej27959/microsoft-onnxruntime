# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# CCCL (libcu++/CUB/Thrust) header handling shared by the in-tree CUDA provider
# (onnxruntime_providers_cuda.cmake) and the CUDA plugin EP (onnxruntime_providers_cuda_plugin.cmake).
# Both compile host C++ translation units that include CUTLASS headers, which pull in <cuda/std/...>,
# so both need the same include path handling.

include_guard(GLOBAL)

# Work around a CUDA 13.3 cudafe++ (EDG front-end) regression that mis-parses CCCL's
# global-qualified partial specializations, e.g. in <cub/device/device_transform.cuh>:
#   template <typename T>
#   struct ::cuda::proclaims_copyable_arguments<...> : ::cuda::std::true_type {};
# nvcc fails with "global qualification of class name is invalid before ':' token".
# The fix is to write the specialization with the namespace reopened instead of using a
# global-qualified name. We cannot edit the (often read-only) toolkit headers, so generate
# corrected copies of the affected headers into the build tree and place that directory
# ahead of the toolkit cccl include path. This is a no-op on toolkits whose headers do not
# contain the offending pattern (e.g. once NVIDIA fixes it), so it is safe to keep enabled.
function(ort_cuda133_patch_cccl_header src dst)
  if (NOT EXISTS "${src}")
    return()
  endif()
  file(READ "${src}" _content)
  set(_orig "${_content}")
  # <cub/device/device_transform.cuh>
  string(REPLACE
    "template <typename T>\nstruct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};"
    "_CCCL_BEGIN_NAMESPACE_CUDA\ntemplate <typename T>\nstruct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};\n_CCCL_END_NAMESPACE_CUDA"
    _content "${_content}")
  # <cub/device/dispatch/tuning/tuning_transform.cuh>
  string(REPLACE
    "template <>\nstruct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::transform::always_true_predicate>\n    : ::cuda::std::true_type\n{};"
    "_CCCL_BEGIN_NAMESPACE_CUDA\ntemplate <>\nstruct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::transform::always_true_predicate>\n    : ::cuda::std::true_type\n{};\n_CCCL_END_NAMESPACE_CUDA"
    _content "${_content}")
  if (NOT _content STREQUAL _orig)
    get_filename_component(_dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(WRITE "${dst}" "${_content}")
  elseif (EXISTS "${dst}")
    # The toolkit header no longer matches the offending pattern (e.g. after a CUDA
    # upgrade in an existing build tree). Remove any previously generated copy so a
    # stale patched header does not keep shadowing the toolkit header.
    file(REMOVE "${dst}")
  endif()
endfunction()

# Handle the CUDA 13.0 CCCL header directory move: libcu++, CUB and Thrust moved from
# <toolkit>/include to <toolkit>/include/cccl, so <cuda/std/utility> - reached from the CUTLASS
# headers that host .cc files include - is no longer on the default include path of the host
# compiler. nvcc adds it by itself, so this only matters for targets that compile host C++.
function(ort_add_cuda_cccl_include_dirs target)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    return()
  endif()

  foreach(inc_dir ${CUDAToolkit_INCLUDE_DIRS})
    if (EXISTS "${inc_dir}/cccl")
      if (UNIX AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.4)
        # Generate cudafe++-parseable copies of the CCCL headers that contain global-qualified
        # partial specializations (see ort_cuda133_patch_cccl_header above) and put the fixed
        # directory ahead of the toolkit cccl include so the corrected headers win.
        set(_ort_cccl_fix_dir "${CMAKE_CURRENT_BINARY_DIR}/cccl_cuda13_fix")
        ort_cuda133_patch_cccl_header(
          "${inc_dir}/cccl/cub/device/device_transform.cuh"
          "${_ort_cccl_fix_dir}/cub/device/device_transform.cuh")
        ort_cuda133_patch_cccl_header(
          "${inc_dir}/cccl/cub/device/dispatch/tuning/tuning_transform.cuh"
          "${_ort_cccl_fix_dir}/cub/device/dispatch/tuning/tuning_transform.cuh")
        if (EXISTS "${_ort_cccl_fix_dir}/cub/device/device_transform.cuh" OR
            EXISTS "${_ort_cccl_fix_dir}/cub/device/dispatch/tuning/tuning_transform.cuh")
          target_include_directories(${target} BEFORE PRIVATE "${_ort_cccl_fix_dir}")
        endif()
      endif()

      # Add the cccl subdirectory to the include path so <cuda/std/utility> can be found
      target_include_directories(${target} PRIVATE "${inc_dir}/cccl")
    endif()
  endforeach()
endfunction()
