// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression test for the two per-step tensors that beam search's CUDA TopK device helper allocates.
//
// GenerationCudaDeviceHelper::TopK is called once per decode step and hands its two outputs to
// TopKImpl, which fills them with work queued on `stream`. They are re-created on the next step, so
// the previous pair goes straight back to the arena while that work may still be in flight - the
// same hazard the Einsum intermediates had (see einsum_stream_allocation_test.cc). The helper has
// to allocate through IAllocator::AllocOnStream so the arena can keep the chunks away from another
// stream until the work completes.
//
// This is a provider-world translation unit: it reaches into CUDA EP internals through the shared
// provider bridge, so it must not include the core framework headers.

#include "gtest/gtest.h"

#ifndef DISABLE_CONTRIB_OPS

#include <memory>
#include <vector>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/stream_handles.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "contrib_ops/cuda/transformers/generation_device_helper.h"
#include "test/providers/cuda/test_cases/stream_recording_allocator.h"

namespace onnxruntime {
namespace test {

class BeamSearchTopKStreamAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }

    CUDA_CALL_THROW(cudaSetDevice(0));
    CUDA_CALL_THROW(cudaStreamCreate(&cuda_stream_));

    device_allocator_ = std::make_shared<CUDAAllocator>(0, CUDA);
    recording_allocator_ = std::make_shared<StreamRecordingAllocator>(device_allocator_);

    device_ = device_allocator_->Info().device;  // Stream holds a reference to this
    stream_ = std::make_unique<Stream>(cuda_stream_, device_);
  }

  void TearDown() override {
    stream_.reset();
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
  }

  std::unique_ptr<Tensor> DeviceTensor(const TensorShape& shape, const std::vector<float>& values) {
    auto tensor = Tensor::Create(DataTypeImpl::GetType<float>(), shape, device_allocator_);
    CUDA_CALL_THROW(cudaMemcpy(tensor->MutableDataRaw(), values.data(), values.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
    return tensor;
  }

  template <typename T>
  std::vector<T> ReadBack(const Tensor& tensor) {
    std::vector<T> values(static_cast<size_t>(tensor.Shape().Size()));
    CUDA_CALL_THROW(cudaStreamSynchronize(cuda_stream_));
    CUDA_CALL_THROW(cudaMemcpy(values.data(), tensor.DataRaw(), values.size() * sizeof(T),
                               cudaMemcpyDeviceToHost));
    return values;
  }

  cudaStream_t cuda_stream_ = nullptr;
  OrtDevice device_;
  AllocatorPtr device_allocator_;
  std::shared_ptr<StreamRecordingAllocator> recording_allocator_;
  std::unique_ptr<Stream> stream_;
};

// Both the values and the indices tensor have to be tagged: TopKImpl writes both on `stream`.
TEST_F(BeamSearchTopKStreamAllocationTest, TopKAllocatesBothOutputsOnTheRunStream) {
  auto input = DeviceTensor(TensorShape({2, 4}), {1.f, 3.f, 2.f, 4.f,
                                                  8.f, 5.f, 7.f, 6.f});

  auto output_values = Tensor::CreateDefault();
  auto output_indices = Tensor::CreateDefault();

  // k is well under the 256 beam limit, so TopK takes the branch that passes a null kernel and needs
  // no allocator of its own beyond the one handed in here.
  auto status = contrib::GenerationCudaDeviceHelper::TopK(input.get(), /*axis*/ 1, /*k*/ 2,
                                                          /*largest*/ true, /*sorted*/ true,
                                                          recording_allocator_, stream_.get(),
                                                          /*threadpool*/ nullptr,
                                                          *output_values, *output_indices);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  EXPECT_EQ(recording_allocator_->stream_allocs, 2);
  EXPECT_EQ(recording_allocator_->untagged_allocs, 0);
  EXPECT_EQ(recording_allocator_->last_stream, stream_.get());

  EXPECT_EQ(output_values->Shape(), TensorShape({2, 2}));
  EXPECT_EQ(output_indices->Shape(), TensorShape({2, 2}));
  EXPECT_EQ(ReadBack<float>(*output_values), std::vector<float>({4.f, 3.f, 8.f, 7.f}));
  EXPECT_EQ(ReadBack<int64_t>(*output_indices), std::vector<int64_t>({3, 1, 0, 2}));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // DISABLE_CONTRIB_OPS
