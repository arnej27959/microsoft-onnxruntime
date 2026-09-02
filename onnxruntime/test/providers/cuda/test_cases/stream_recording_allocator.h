// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shared helper for the CUDA EP "intermediates are allocated on their stream" regression tests.
//
// A stream aware arena can only keep a buffer away from a second stream if the allocation was
// tagged with the stream that uses it, and the only way it learns that is IAllocator::AllocOnStream.
// A per-run intermediate that goes back to the arena through the plain Alloc() may be handed to
// another stream while the work writing it is still queued. The tests wrap the real device
// allocator in this recorder and assert that each intermediate arrived through AllocOnStream with
// the run's stream, so a call site that reverts to an untagged allocation is caught.
//
// This is a provider-world header: it is only usable from a translation unit that includes the
// shared provider bridge headers rather than the core framework ones.

#pragma once

#include <utility>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/stream_handles.h"

namespace onnxruntime {
namespace test {

// Forwards to a real device allocator and records which entry point each allocation came in on.
class StreamRecordingAllocator : public IAllocator {
 public:
  explicit StreamRecordingAllocator(AllocatorPtr inner)
      : IAllocator(inner->Info()), inner_(std::move(inner)) {}

  bool IsStreamAware() const override { return true; }

  void* Alloc(size_t size) override {
    ++untagged_allocs;
    return inner_->Alloc(size);
  }

  void* AllocOnStream(size_t size, Stream* stream) override {
    ++stream_allocs;
    last_stream = stream;
    return inner_->AllocOnStream(size, stream);
  }

  void Free(void* p) override { inner_->Free(p); }

  int untagged_allocs = 0;
  int stream_allocs = 0;
  Stream* last_stream = nullptr;

 private:
  AllocatorPtr inner_;
};

}  // namespace test
}  // namespace onnxruntime
