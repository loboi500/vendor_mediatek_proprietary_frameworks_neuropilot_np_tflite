/*
 * Copyright (C) 2021 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/experimental/delegates/neuron/APUWareUtilsApi.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_builder.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_utils.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_validation.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include <android/hardware_buffer.h>
#include <BufferAllocator/BufferAllocatorWrapper.h>
#include "ion/ion_internal.h"
#include "ion/ion_drv.h"
#include "ion/mtk_ion.h"
#include <ion/ion.h>

#define NEURON_REUSABLE_EXECUTION

namespace tflite {
namespace neuron {

// The kernel that represents the node sub set of TFLite being run on Neuron
// API.
struct NeuronOpMappingArgs {
  TfLiteContext* context;
  NeuronOpBuilder* builder;
  TfLiteNode* node;
  std::vector<int>* model_state_outputs;
  std::vector<int>* model_state_tfl_inputs;
  std::vector<std::tuple<int, int>>* feedback_loops;
  int* neuron_errno;
  /// M: NeuroPilot {@
  TfLiteRegistration* registration;
  /// M: NeuroPilot @}
};

// Manage Neuron shared memory handle
class NNMemory {
 public:
  NNMemory(const NeuronApi* neuronapi, const char* name, size_t size, bool use_ahwb, bool use_ion);

  ~NNMemory();

  uint8_t* get_data_ptr() { return data_ptr_; }
  NeuronMemory* get_handle() { return nn_memory_handle_; }
  size_t get_byte_size() { return byte_size_; }

  int ion_memory_lock() {
      if (buffer_) {
          return AHardwareBuffer_lock(buffer_,
                  AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                  -1, NULL, reinterpret_cast<void**>(&data_ptr_));
      }
      return 0;
  }
  int ion_memory_unlock() {
      if (buffer_) {
          return AHardwareBuffer_unlock(buffer_, nullptr);
      }
      return 0;
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
  int fd_ = 0;
  size_t byte_size_ = 0;
  uint8_t* data_ptr_ = nullptr;
  bool use_ahwb_ = true;
  NeuronMemory* nn_memory_handle_ = nullptr;
  AHardwareBuffer* buffer_ = nullptr;
  int ion_handle_ = 0;
  ion_user_handle_t buf_handle_ = 0;
  BufferAllocator* bufferAllocator_ = nullptr;
};

// RAII Neuron Model Destructor for use with std::unique_ptr
class NNFreeModel {
 public:
  explicit NNFreeModel(const NeuronApi* neuronapi) : neuronapi_(neuronapi) {}
  void operator()(NeuronModel* model) { neuronapi_->NeuronModel_free(model); }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Compilation Destructor for use with std::unique_ptr
class NNFreeCompilation {
 public:
  explicit NNFreeCompilation(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronCompilation* compilation) {
    neuronapi_->NeuronCompilation_free(compilation);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// RAII Neuron Execution Destructor for use with std::unique_ptr
class NNFreeExecution {
 public:
  explicit NNFreeExecution(const NeuronApi* neuronapi)
      : neuronapi_(neuronapi) {}
  void operator()(NeuronExecution* execution) {
    neuronapi_->NeuronExecution_free(execution);
  }

 private:
  // NeuronApi instance to use. Not owned by this object.
  const NeuronApi* neuronapi_;
};

// Neuron delegate kernel.
class NeuronDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  TfLiteStatus Map(TfLiteContext* context, int builtin_code, int version,
                          const char* custom_name, int neuron_platform_version,
                          const NeuronOpMappingArgs& mapping_args,
                          NeuronOperationType* nn_op_type);

  explicit NeuronDelegateKernel(const NeuronApi* neuronapi,
                                const NeuronDelegateOptions& options)
      : neuronapi_(neuronapi),
        options_(options),
        nn_model_(nullptr, NNFreeModel(neuronapi_)),
#ifdef NEURON_REUSABLE_EXECUTION
        nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)),
        nn_execution_(nullptr, NNFreeExecution(neuronapi_)) {}
#else
        nn_compilation_(nullptr, NNFreeCompilation(neuronapi_)) {}
#endif


  ~NeuronDelegateKernel() { releasePerformanceLock(perf_handle_); }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

 private:
  const NeuronApi* neuronapi_;
  NeuronDelegateOptions options_;
  // True if initialization has been completed successfully
  bool initialised_;
  std::unique_ptr<NeuronModel, NNFreeModel> nn_model_;
  std::unique_ptr<NeuronCompilation, NNFreeCompilation> nn_compilation_;
#ifdef NEURON_REUSABLE_EXECUTION
  std::unique_ptr<NeuronExecution, NNFreeExecution> nn_execution_;
#endif
  // Node indices that this delegate is responsible for. Indices here
  // indexes into the nodes array in the TfLiteContext.
  std::vector<int> nodes_;
  // Track indices we use
  OperandMapping operand_mapping_;

  std::vector<NeuronMemory*> tensor_memory_map_kernel_;

  std::vector<int> model_state_outputs_;
  std::vector<int> model_state_tfl_inputs_;
  // This is the equivalent of the pair model_state_outputs_,
  // model_state_tfl_inputs_ for all tensors where we have to keep the output
  // data available for TFLite model users
  std::vector<std::tuple<int, int>> feedback_loops_;

  std::unique_ptr<NNMemory> nn_input_memory_;
  std::unique_ptr<NNMemory> nn_output_memory_;

  std::vector<uint8_t> nn_compilation_cache_token_;

  /// M: NeuroPilot {@
  // List of the MTK OPs to be fused with other OPs
  std::vector<TfLiteNode> fusion_mtk_op_list_;
  /// M: NeuroPilot @}

  void AddDequantizeOperatorsWhereNeeded(const TfLiteContext* context,
                                         int builtin_code,
                                         const TfLiteNode* node,
                                         NeuronOpBuilder* builder);

  TfLiteStatus AddOpsAndTensors(TfLiteContext* context);

  TfLiteStatus BuildGraph(TfLiteContext* context,
                          const TfLiteIntArray* input_tensors,
                          const TfLiteIntArray* output_tensors);
  int perf_handle_ = 0;
  // Prevent set memory multi-times when we use ahwb
  bool need_set_memory = true;
};

}  // namespace neuron
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_DELEGATES_NEURON_NEURON_KERNEL_H_
