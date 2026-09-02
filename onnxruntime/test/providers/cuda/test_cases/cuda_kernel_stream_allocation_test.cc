// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression tests for the per-run intermediates that CUDA EP kernels allocate inside Compute().
//
// A kernel that builds an intermediate with Tensor::Create(type, shape, allocator) reaches
// IAllocator::Alloc() with no stream, so a stream aware arena cannot associate the chunk with the
// stream that writes it and is free to hand it to another stream the moment the tensor is released
// - which happens at the end of Compute(), while the copy or kernel filling it may still be queued.
// The fix is to pass the run's stream so the allocation goes through IAllocator::AllocOnStream.
//
// einsum_stream_allocation_test.cc covers the Einsum device helpers by calling them directly,
// because they take the allocator and the stream as arguments. The kernels here get theirs from
// OpKernelContext, so they can only be driven through a real Run(). These tests register a
// recording allocator as the environment's shared CUDA allocator, point a session at it with
// session.use_env_allocators, and assert that no device allocation made during Run() arrived
// through the untagged Alloc().
//
// This translation unit runs a real InferenceSession, so it includes the core framework headers and
// must not include the shared-provider-bridge ones.

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/span_utils.h"
#include "core/framework/allocator.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/session/allocator_adapters.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

namespace {

// A device allocator over raw cudaMalloc/cudaFree that records, for the allocations made while it
// is recording, which entry point each one came in on. There is no arena underneath: every request
// is a distinct call, so the counts below are exactly the number of tensors the run allocated.
class StreamTaggingRecorder : public IAllocator {
 public:
  StreamTaggingRecorder()
      : IAllocator(OrtMemoryInfo(onnxruntime::CUDA, OrtAllocatorType::OrtDeviceAllocator,
                                 OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                           OrtDevice::VendorIds::NVIDIA, 0),
                                 OrtMemTypeDefault)) {}

  bool IsStreamAware() const override { return true; }

  void* Alloc(size_t size) override {
    if (recording_) {
      ++untagged_allocs_;
      untagged_sizes_.push_back(size);
    }
    return RawAlloc(size);
  }

  void* AllocOnStream(size_t size, Stream* /*stream*/) override {
    if (recording_) {
      ++stream_allocs_;
    }
    return RawAlloc(size);
  }

  // Reserve() is the deliberate arena-bypassing entry point (GetTransientScratchBuffer), not an
  // untagged Alloc(). Overridden so the IAllocator default of forwarding to Alloc() does not report
  // one of those as a missing stream tag.
  void* Reserve(size_t size) override { return RawAlloc(size); }

  void Free(void* p) override {
    if (p != nullptr) {
      cudaSetDevice(0);
      cudaFree(p);
    }
  }

  // Session initialization allocates through this allocator too; only Run() is of interest here.
  void StartRecording() {
    untagged_allocs_ = 0;
    stream_allocs_ = 0;
    untagged_sizes_.clear();
    recording_ = true;
  }
  void StopRecording() { recording_ = false; }

  int untagged_allocs() const { return untagged_allocs_; }
  int stream_allocs() const { return stream_allocs_; }

  // Renders the untagged allocation sizes so a failure names what was missed rather than just
  // reporting a count.
  std::string UntaggedSummary() const {
    std::string out = "untagged device allocations during Run(), sizes in bytes:";
    for (size_t size : untagged_sizes_) {
      out += " " + std::to_string(size);
    }
    return out;
  }

 private:
  // CUDAAllocator selects the device on every call for the same reason: ORT calls in on several
  // threads and the current device is per thread.
  static void* RawAlloc(size_t size) {
    if (size == 0) {
      return nullptr;
    }
    void* p = nullptr;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&p, size) != cudaSuccess) {
      return nullptr;
    }
    return p;
  }

  bool recording_ = false;
  int untagged_allocs_ = 0;
  int stream_allocs_ = 0;
  std::vector<size_t> untagged_sizes_;
};

void AddFloatValueInfo(ONNX_NAMESPACE::ValueInfoProto* vi, const char* name,
                       std::initializer_list<int64_t> dims) {
  vi->set_name(name);
  auto* tensor_type = vi->mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* shape = tensor_type->mutable_shape();
  for (int64_t dim : dims) {
    shape->add_dim()->set_dim_value(dim);
  }
}

// The element shape is left unset: it is whatever the sequence ops infer, and declaring one here
// would only give graph resolution a chance to disagree with that.
void AddFloatSequenceValueInfo(ONNX_NAMESPACE::ValueInfoProto* vi, const char* name) {
  vi->set_name(name);
  vi->mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
}

// Single Softmax node over X[2,3,4]. With opset >= 13 the kernel transposes `axis` to the innermost
// dim first, and that transpose is what allocates the two intermediates; when `axis` is already the
// innermost dim the kernel works in place on the input and output and allocates nothing.
std::string BuildSoftmaxModelBytes(int64_t axis) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("softmax_stream_allocation");
  AddFloatValueInfo(graph->add_input(), "X", {2, 3, 4});
  AddFloatValueInfo(graph->add_output(), "Y", {2, 3, 4});

  auto* node = graph->add_node();
  node->set_name("softmax");
  node->set_op_type("Softmax");
  node->add_input("X");
  node->add_output("Y");
  auto* attr = node->add_attribute();
  attr->set_name("axis");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr->set_i(axis);

  return model.SerializeAsString();
}

// A chain that copies tensors into and out of a TensorSeq: SequenceConstruct copies A and B,
// SequenceInsert copies C plus the two elements it carries over, SequenceErase copies the two it
// keeps, and SequenceAt returns the last one. Every one of those copies is a Tensor::Create in the
// CUDA EP followed by a cudaMemcpyAsync on the run's stream.
//
// `with_identity` appends an Identity over the sequence and makes *that* the sequence shaped graph
// output, instead of the SequenceErase result. IdentityOp's copy loop needs the node's output to be
// a distinct OrtValue from its input, and the Identity kernel def aliases output 0 to input 0 - so
// the loop only runs when something stops the planner applying that alias. Being a graph output is
// what does it: AllocationPlanner tests for a graph output (allocation_planner.cc, "node_output is
// graph's output, so we can't reuse intermediate buffer") before it consults the alias map, so the
// output is planned kAllocateOutput and ExecutionFrame hands the kernel a fresh TensorSeq. Both
// variants are run so the difference between them isolates the Identity copies.
std::string BuildSequenceModelBytes(bool with_identity) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(17);

  auto* graph = model.mutable_graph();
  graph->set_name("sequence_stream_allocation");
  AddFloatValueInfo(graph->add_input(), "A", {2, 3});
  AddFloatValueInfo(graph->add_input(), "B", {2, 3});
  AddFloatValueInfo(graph->add_input(), "C", {2, 3});
  AddFloatValueInfo(graph->add_output(), "Y", {2, 3});
  AddFloatSequenceValueInfo(graph->add_output(), with_identity ? "S3" : "S2");

  // Scalar int64 positions. Both are CPU inputs in the CUDA kernel defs, so they never reach the
  // device allocator being recorded.
  auto add_position = [graph](const char* name, int64_t value) {
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    init->add_int64_data(value);
  };
  add_position("erase_pos", 0);
  add_position("at_pos", 1);

  auto add_node = [graph](const char* name, const char* op_type,
                          std::initializer_list<const char*> inputs, const char* output) {
    auto* node = graph->add_node();
    node->set_name(name);
    node->set_op_type(op_type);
    for (const char* input : inputs) {
      node->add_input(input);
    }
    node->add_output(output);
  };

  add_node("construct", "SequenceConstruct", {"A", "B"}, "S0");   // [A, B]
  add_node("insert", "SequenceInsert", {"S0", "C"}, "S1");        // [A, B, C]
  add_node("erase", "SequenceErase", {"S1", "erase_pos"}, "S2");  // [B, C]
  add_node("at", "SequenceAt", {"S2", "at_pos"}, "Y");            // C
  if (with_identity) {
    add_node("identity", "Identity", {"S2"}, "S3");  // [B, C]
  }

  return model.SerializeAsString();
}

}  // namespace

class CudaKernelStreamAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }

    // A private environment: registering a shared allocator is process wide, and the recorder must
    // not be handed to any other test's session.
    auto logging_manager = std::make_unique<logging::LoggingManager>(
        std::unique_ptr<logging::ISink>(new logging::CLogSink()), logging::Severity::kWARNING, false,
        logging::LoggingManager::InstanceType::Temporal);
    ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), env_));

    recorder_ = std::make_shared<StreamTaggingRecorder>();
    ort_allocator_ = std::make_unique<OrtAllocatorImplWrappingIAllocator>(AllocatorPtr(recorder_));
    ASSERT_STATUS_OK(env_->RegisterAllocator(ort_allocator_.get()));
  }

  // Loads `model_bytes` onto the CUDA EP with the recorder as the device allocator, runs it once
  // with recording on, and returns the fetches.
  void RunRecorded(const std::string& model_bytes, const NameMLValMap& feeds,
                   const std::vector<std::string>& output_names, std::vector<OrtValue>& fetches) {
    SessionOptions so;
    so.session_logid = "CudaKernelStreamAllocation";
    // Keep the graph as written, and keep every tensor an individual allocation: the memory pattern
    // planner would otherwise satisfy them out of one pre-planned block and hide the counts.
    so.graph_optimization_level = TransformerLevel::Default;
    so.enable_mem_pattern = false;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1"));

    InferenceSessionWrapper session(so, *env_);
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_shared<CUDAExecutionProvider>(CUDAExecutionProviderInfo{})));
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    // A node that fell back to the CPU EP would allocate nothing on the device and make the
    // assertions below vacuously true.
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == "MemcpyFromHost" || node.OpType() == "MemcpyToHost") {
        continue;
      }
      ASSERT_EQ(node.GetExecutionProviderType(), onnxruntime::kCudaExecutionProvider)
          << "Node '" << node.Name() << "' (" << node.OpType() << ") was not assigned to the CUDA EP.";
    }

    recorder_->StartRecording();
    Status run_status = session.Run(feeds, output_names, &fetches);
    recorder_->StopRecording();
    ASSERT_STATUS_OK(run_status);
  }

  static OrtValue CpuInput(std::vector<float>& data, std::initializer_list<int64_t> dims) {
    OrtValue value;
    CreateMLValue<float>(AsSpan(dims), data.data(), OrtMemoryInfo(), &value);
    return value;
  }

  static std::vector<float> ReadTensor(const Tensor& tensor) {
    auto span = tensor.DataAsSpan<float>();
    return std::vector<float>(span.begin(), span.end());
  }

  static std::vector<float> ReadFetch(const OrtValue& value) { return ReadTensor(value.Get<Tensor>()); }

  static std::vector<float> ReadSeqElement(const OrtValue& value, size_t index) {
    return ReadTensor(value.Get<TensorSeq>().Get(index));
  }

  std::shared_ptr<StreamTaggingRecorder> recorder_;
  std::unique_ptr<OrtAllocatorImplWrappingIAllocator> ort_allocator_;
  // Declared last so it is destroyed first, before the allocator it holds a reference to.
  std::unique_ptr<Environment> env_;
};

// Softmax's transpose scratch and intermediate output. Running the same model with the axis already
// innermost gives the baseline the transposed run is measured against: the difference has to be
// exactly the two intermediates, and all of them have to be tagged.
TEST_F(CudaKernelStreamAllocationTest, SoftmaxTransposeIntermediatesAreOnTheRunStream) {
  std::vector<float> x_data(24);
  for (size_t i = 0; i < x_data.size(); ++i) {
    x_data[i] = static_cast<float>(i);
  }

  NameMLValMap feeds;
  feeds.emplace("X", CpuInput(x_data, {2, 3, 4}));
  const std::vector<std::string> output_names{"Y"};

  // axis == rank - 1: no transpose, so the kernel allocates no intermediates at all.
  std::vector<OrtValue> baseline_fetches;
  RunRecorded(BuildSoftmaxModelBytes(/*axis*/ 2), feeds, output_names, baseline_fetches);
  ASSERT_EQ(recorder_->untagged_allocs(), 0) << recorder_->UntaggedSummary();
  const int baseline_tagged = recorder_->stream_allocs();

  std::vector<OrtValue> fetches;
  RunRecorded(BuildSoftmaxModelBytes(/*axis*/ 1), feeds, output_names, fetches);
  EXPECT_EQ(recorder_->untagged_allocs(), 0) << recorder_->UntaggedSummary();
  EXPECT_EQ(recorder_->stream_allocs(), baseline_tagged + 2)
      << "The transposed Softmax path should add exactly two stream tagged allocations - the "
         "transpose scratch and the intermediate output.";

  // X[b,i,j] == b*12 + i*4 + j, so along axis 1 every one of the 8 columns holds {0,4,8} plus a
  // constant and softmaxes to the same three increasing values that sum to 1.
  const std::vector<float> y = ReadFetch(fetches[0]);
  ASSERT_EQ(y.size(), static_cast<size_t>(24));
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(y[b * 12 + i * 4 + j], y[i * 4], 1e-6f) << "at [" << b << "," << i << "," << j << "]";
      }
    }
  }
  EXPECT_NEAR(y[0] + y[4] + y[8], 1.0f, 1e-5f);
  EXPECT_LT(y[0], y[4]);
  EXPECT_LT(y[4], y[8]);
}

// The element copies in SequenceConstruct, SequenceInsert, SequenceErase and IdentityOp. Unlike the
// Softmax scratch these tensors are moved into the output TensorSeq rather than released inside
// Compute(), so the race Einsum hit is not open here - but a sequence output can outlive the run
// and be released at an arbitrary point, so the tagging still has to be right.
TEST_F(CudaKernelStreamAllocationTest, SequenceElementCopiesAreOnTheRunStream) {
  std::vector<float> a_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> b_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  std::vector<float> c_data{100.f, 200.f, 300.f, 400.f, 500.f, 600.f};

  NameMLValMap feeds;
  feeds.emplace("A", CpuInput(a_data, {2, 3}));
  feeds.emplace("B", CpuInput(b_data, {2, 3}));
  feeds.emplace("C", CpuInput(c_data, {2, 3}));

  // Without Identity: 2 (construct) + 3 (insert) + 2 (erase) element copies, plus the graph's own
  // tensors. The exact total depends on how the inputs reach the device, so the baseline is
  // measured rather than predicted, and only the floor the element copies establish is asserted.
  const std::vector<std::string> baseline_names{"Y", "S2"};
  std::vector<OrtValue> baseline_fetches;
  RunRecorded(BuildSequenceModelBytes(/*with_identity*/ false), feeds, baseline_names, baseline_fetches);
  ASSERT_EQ(recorder_->untagged_allocs(), 0) << recorder_->UntaggedSummary();
  ASSERT_GE(recorder_->stream_allocs(), 7);
  const int baseline_tagged = recorder_->stream_allocs();

  const std::vector<std::string> output_names{"Y", "S3"};
  std::vector<OrtValue> fetches;
  RunRecorded(BuildSequenceModelBytes(/*with_identity*/ true), feeds, output_names, fetches);
  EXPECT_EQ(recorder_->untagged_allocs(), 0) << recorder_->UntaggedSummary();
  // Two elements in the sequence, so Identity copies exactly two tensors. A zero delta would mean
  // the planner aliased its output onto its input after all and the copy loop never ran, which
  // would make this case cover nothing.
  EXPECT_EQ(recorder_->stream_allocs(), baseline_tagged + 2)
      << "The Identity node should add exactly one stream tagged allocation per sequence element.";

  // SequenceAt(1) of [B, C] is C, and the sequence output is [B, C] either way.
  EXPECT_EQ(ReadFetch(fetches[0]), c_data);
  ASSERT_EQ(fetches[1].Get<TensorSeq>().Size(), static_cast<size_t>(2));
  EXPECT_EQ(ReadSeqElement(fetches[1], 0), b_data);
  EXPECT_EQ(ReadSeqElement(fetches[1], 1), c_data);
}

}  // namespace test
}  // namespace onnxruntime
