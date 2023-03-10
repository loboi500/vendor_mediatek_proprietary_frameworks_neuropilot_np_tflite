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

#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_kernel.h"

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_builder.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_utils.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_validation.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_types.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/core/subgraph.h"
#include "utils/hash/farmhash.h"

/// M: NeuroPilot {@
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/mtk/mtk_options.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>
#include "flatbuffers/flexbuffers.h"
#include <BufferAllocator/BufferAllocatorWrapper.h>
#include <sys/mman.h>
#include <unistd.h>
#include <array>
/// M: NeuroPilot @}

namespace tflite {
namespace neuron {

namespace {

std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    if (buffer.data() != nullptr) {
      result += buffer.data();
    }
  }
  return result;
}

bool is_number(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

bool IsScalarInputSupported(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinSum:
    case kTfLiteBuiltinCustom:
      return true;
    default:
      return false;
  }
}

// Check if the operation requires explict conversion from int8 to uint8 values.
bool NeedInt8Conversion(const TfLiteContext* context, int builtin_code,
                        const TfLiteNode* node) {
  const int input_id = node->inputs->data[0];
  const TfLiteType input_type = context->tensors[input_id].type;
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinFullyConnected: {
      if (input_type == kTfLiteInt8) {
        const int weights_id = node->inputs->data[1];
        const auto& weights_tensor = context->tensors[weights_id];
        if ((weights_tensor.type == kTfLiteInt8 ||
             weights_tensor.type == kTfLiteUInt8) &&
            weights_tensor.quantization.type == kTfLiteAffineQuantization) {
          return true;
        }
      }
      return false;
    }
    case kTfLiteBuiltinTransposeConv: {
      // Transpose convolution has a different order of inputs:
      // 0: output_shape, 1: filter, 2: input, 3: bias.
      const int input_id = 2;
      const TfLiteType input_type = context->tensors[input_id].type;
      if (input_type == kTfLiteInt8) {
        return true;
      }
      return false;
    }
    case kTfLiteBuiltinSelect: {
      const auto value_type = context->tensors[node->inputs->data[1]].type;
      return value_type == kTfLiteInt8;
    }
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinBatchToSpaceNd:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinEqual:
    case kTfLiteBuiltinExpandDims:
    case kTfLiteBuiltinGather:
    case kTfLiteBuiltinGreater:
    case kTfLiteBuiltinGreaterEqual:
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLeakyRelu:
    case kTfLiteBuiltinLess:
    case kTfLiteBuiltinLessEqual:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinNotEqual:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinReduceMax:
    case kTfLiteBuiltinReduceMin:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReluN1To1:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinSpaceToBatchNd:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTile:
    case kTfLiteBuiltinTopkV2:
    case kTfLiteBuiltinTranspose: {
      return input_type == kTfLiteInt8;
    }
    default:
      return false;
  }
}  // namespace

constexpr int kLstmFullKernelInputSize = 24;
// The 20 input version is deprecated and kept only to
// support old model. The latest version of the LSTM Full Kernel
// is the one with 24 inputs
constexpr int kLstmFullKernelNoOptionalParamsInputSize = 20;
constexpr int kLstmBasicKernelInputSize = 5;

inline bool isLstmBasicKernel(const TfLiteNode* node) {
  return node->inputs->size == kLstmBasicKernelInputSize;
}

inline bool isLstmFullKernel(const TfLiteNode* node) {
  return node->inputs->size == kLstmFullKernelInputSize ||
         node->inputs->size == kLstmFullKernelNoOptionalParamsInputSize;
}

bool IsHybridOperator(const TfLiteContext* context, int builtin_code,
                      const TfLiteNode* node) {
  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      const int input_id = node->inputs->data[0];
      const int filter_id = node->inputs->data[1];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType filter_type = context->tensors[filter_id].type;
      return IsFloat(input_type) && IsQuantized(filter_type);
    }
    case kTfLiteBuiltinLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return isLstmFullKernel(node) && IsFloat(input_type) &&
             IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinBidirectionalSequenceLstm: {
      const int input_id = node->inputs->data[0];
      // Input #1 is optional so use #2 to determine if hybrid.
      const int weights_id = node->inputs->data[2];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    case kTfLiteBuiltinUnidirectionalSequenceRnn: {
      const int input_id = node->inputs->data[0];
      const int weights_id = node->inputs->data[1];
      const TfLiteType input_type = context->tensors[input_id].type;
      const TfLiteType weights_type = context->tensors[weights_id].type;
      return IsFloat(input_type) && IsQuantized(weights_type);
    }
    default:
      return false;
  }
}

constexpr size_t kDefaultByteAlignmentForNeuron = 128;

static size_t getNumPaddingBytes(size_t byte_size) {
  size_t num_padding_bytes = 0;
  if (byte_size % kDefaultByteAlignmentForNeuron) {
    num_padding_bytes = kDefaultByteAlignmentForNeuron -
                        (byte_size % kDefaultByteAlignmentForNeuron);
  }
  return num_padding_bytes;
}

// Compute the hash of a TfLiteIntArray.
uint64_t GetHash(const TfLiteIntArray* int_array, uint64_t combine_with = 0) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = combine_with;
  for (auto i : TfLiteIntArrayView(int_array)) {
    result = result ^ (i + kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

uint64_t GetInOutPutHash(TfLiteContext* context, const TfLiteIntArray* int_array) {
  constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
  uint64_t result = 0;
  for (auto i : TfLiteIntArrayView(int_array)) {
    result = result ^ (context->tensors[i].bytes + kHashConst + (result << 10) + (result >> 4));
  }
  return result;
}

TfLiteStatus GetAcceleratorDevice(const NeuronApi* neuron_api,
                                  TfLiteContext* context,
                                  const char* device_name,
                                  NeuronDevice** result) {
  if (!device_name) {
    return kTfLiteError;
  }
  uint32_t num_devices = 0;
  *result = nullptr;

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuron_api->Neuron_getDeviceCount(&num_devices),
      "getting device count");

  for (uint32_t i = 0; i < num_devices; ++i) {
    const char* name = nullptr;
    NeuronDevice* device = nullptr;
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuron_api->Neuron_getDevice(i, &device), "getting device");
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context, neuron_api->NeuronDevice_getName(device, &name),
        "getting device name");
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Got device name: %s", name);
    if (std::string(device_name) == std::string(name)) {
      *result = device;
      return kTfLiteOk;
    }
  }

  context->ReportError(context,
                       "Could not find the specified Neuron accelerator: %s.",
                       device_name);
  return kTfLiteError;
}

}  // namespace

NNMemory::NNMemory(const NeuronApi* neuronapi, const char* name, size_t size,
                   bool use_ahwb, bool use_ion) {
  if (name && size > 0) {
    neuronapi_ = neuronapi;
    byte_size_ = size;
    if (neuronapi_->android_sdk_version <= kMinSdkVersionForNeuron13) {
      if (use_ion) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "use Ion Buffer");
        ion_handle_ = mt_ion_open(__FILE__);
        if (ion_handle_) {
          if (!ion_alloc_mm(ion_handle_, size, 0, 0, &buf_handle_)) {
            if (ion_share(ion_handle_, buf_handle_, (int*)&fd_)) {
              ion_free(ion_handle_, buf_handle_);
            } else {
              data_ptr_ = reinterpret_cast<uint8_t*>(
                  ion_mmap(ion_handle_, NULL, size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, 0));
              if (!data_ptr_) {
                ion_share_close(ion_handle_, fd_);
                ion_free(ion_handle_, buf_handle_);
              } else {
                neuronapi_->NeuronMemory_createFromFd(
                    size, PROT_READ | PROT_WRITE, fd_, 0, &nn_memory_handle_);
              }
            }
          }
        }
      }
    } else {
#ifdef YOCTO_BUILD
      bufferAllocator_ = CreateDmabufHeapBufferAllocator();
      if (bufferAllocator_) {
        fd_ = DmabufHeapAllocSystem(bufferAllocator_, false /* cpu_access */,
                                    byte_size_, 0, 0);
        if (fd_ >= 0) {
          data_ptr_ = reinterpret_cast<uint8_t*>(mmap(
              NULL, byte_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
          if (data_ptr_ != MAP_FAILED) {
            neuronapi_->NeuronMemory_createFromFd(size, PROT_READ | PROT_WRITE,
                                                  fd_, 0, &nn_memory_handle_);
          }
        }
      }
#else
      if (use_ahwb) {
        uint64_t usage;
        std::string version = exec("uname -r");
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "kernel version %s",
                        version.c_str());
        std::string big_version;
        std::size_t found = version.find(".");
        if (found != std::string::npos) {
          big_version = version.substr(0, found);
        }
        if (is_number(big_version) && std::stoi(big_version) >= 5) {
          TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                          "use cacheable AhardwareBuffer");
          usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        } else {
          TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                          "use non cacheable AhardwareBuffer");
          usage = AHARDWAREBUFFER_USAGE_CPU_READ_RARELY |
                  AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;
        }
        AHardwareBuffer_Desc desc{
            .width = static_cast<uint32_t>(size),
            .height = 1,
            .layers = 1,
            .format = AHARDWAREBUFFER_FORMAT_BLOB,
            .usage = usage,
            .stride = static_cast<uint32_t>(size),
        };
        if (AHardwareBuffer_allocate(&desc, &buffer_) == 0) {
          neuronapi_->NeuronMemory_createFromAHardwareBuffer(
              buffer_, &nn_memory_handle_);
              ion_memory_lock();
        }
      }
#endif
    }
    if (!nn_memory_handle_) {
      data_ptr_ = reinterpret_cast<uint8_t*>(malloc(size));
      if (!data_ptr_) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                        "fail to malloc for NN memory handle");
      }
    }
  } else {
    neuronapi_ = nullptr;
  }
}

NNMemory::~NNMemory() {
  if (data_ptr_) {
    if (ion_handle_) {
        ion_munmap(ion_handle_, data_ptr_, byte_size_);
    } else if (buffer_) {
        munmap(data_ptr_, byte_size_);
    } else {
        free(data_ptr_);
    }
  }
  if (nn_memory_handle_) {
    ion_memory_unlock();
    neuronapi_->NeuronMemory_free(nn_memory_handle_);
  }
  if (fd_ > 0) {
      if (ion_handle_) {
          ion_share_close(ion_handle_, fd_);
      } else {
          close(fd_);
      }
  }
  if (ion_handle_) {
      if (buf_handle_) {
          ion_free(ion_handle_, buf_handle_);
      }
      ion_close(ion_handle_);
  }
  if (bufferAllocator_ != nullptr) {
      FreeDmabufHeapBufferAllocator(bufferAllocator_);
  }
  if (buffer_) {
      AHardwareBuffer_release(buffer_);
  }
}

TfLiteStatus NeuronDelegateKernel::Map(TfLiteContext* context, int builtin_code,
                                       int version, const char* custom_name, int android_sdk_version,
                                       const NeuronOpMappingArgs& mapping_args,
                                       NeuronOperationType* nn_op_type) {
  auto add_zero_bias = [mapping_args](int input_id, int filter_id,
                                      int num_elements) -> void {
    // Neuron requires a bias tensor, so we allocate a new tensor to fill
    // it with zeroes. It is deleted with other tensors in the context
    // during subgraph destructor call.
    int bias_index = -1;
    mapping_args.context->AddTensors(mapping_args.context, 1, &bias_index);
    TfLiteTensor* bias_tensor = &mapping_args.context->tensors[bias_index];
    const auto input_type = mapping_args.context->tensors[input_id].type;
    if (input_type == kTfLiteFloat32) {
      bias_tensor->type = kTfLiteFloat32;
    } else {
      bias_tensor->type = kTfLiteInt32;
    }
    // Create an array with a required bias shape and resize the bias
    // tensor.
    TfLiteIntArray* bias_shape = TfLiteIntArrayCreate(1);
    bias_shape->data[0] = num_elements;
    bias_tensor->allocation_type = kTfLiteDynamic;
    mapping_args.context->ResizeTensor(mapping_args.context, bias_tensor,
                                       bias_shape);
    // Set tensor's values to zeroes and add it using AddVector*, so
    // that the values are copied to Neuron. We don't use the AddTensor
    // function because it doesn't copy values and the tensor we just
    // created is not in the node->inputs.
    if (input_type == kTfLiteFloat32) {
      memset(bias_tensor->data.f, 0, num_elements * sizeof(float));
      mapping_args.builder->AddVectorFloat32Operand(bias_tensor->data.f,
                                                    num_elements);
    } else {
      memset(bias_tensor->data.i32, 0, num_elements * sizeof(int));
      const TfLiteTensor& input_tensor =
          mapping_args.context->tensors[input_id];
      const TfLiteTensor& filter_tensor =
          mapping_args.context->tensors[filter_id];
      // Neuron requires bias scale to be a product of an input scale and
      // a filter scale.
      bias_tensor->params.scale =
          input_tensor.params.scale * filter_tensor.params.scale;
      mapping_args.builder->AddVectorInt32Operand(
          bias_tensor->data.i32, num_elements, bias_tensor->params.scale,
          /*zero_point=*/0);
    }
  };
  switch (builtin_code) {
    case kTfLiteBuiltinAdd: {
      auto builtin =
          reinterpret_cast<TfLiteAddParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_ADD;
    } break;
    case kTfLiteBuiltinArgMax: {
      *nn_op_type = NEURON_ARGMAX;
    } break;
    case kTfLiteBuiltinArgMin: {
      *nn_op_type = NEURON_ARGMIN;
    } break;
    case kTfLiteBuiltinMul: {
      auto builtin =
          reinterpret_cast<TfLiteMulParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_MUL;
    } break;
    case kTfLiteBuiltinAveragePool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = NEURON_AVERAGE_POOL_2D;
    } break;
    case kTfLiteBuiltinMaxPool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = NEURON_MAX_POOL_2D;
    } break;
    case kTfLiteBuiltinL2Pool2d: {
      mapping_args.builder->AddPoolingParams(mapping_args.node->builtin_data);
      *nn_op_type = NEURON_L2_POOL_2D;
    } break;
    case kTfLiteBuiltinConv2d: {
      auto builtin =
          reinterpret_cast<TfLiteConvParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      mapping_args.builder->AddScalarInt32Operand(
          builtin->dilation_width_factor);
      mapping_args.builder->AddScalarInt32Operand(
          builtin->dilation_height_factor);
      *nn_op_type = NEURON_CONV_2D;
    } break;
    case kTfLiteBuiltinDepthwiseConv2d: {
      auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(builtin->depth_multiplier);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      if (builtin->dilation_width_factor != 1 ||
          builtin->dilation_height_factor != 1) {
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_width_factor);
        mapping_args.builder->AddScalarInt32Operand(
            builtin->dilation_height_factor);
      }
      *nn_op_type = NEURON_DEPTHWISE_CONV_2D;
    } break;
    case kTfLiteBuiltinFullyConnected: {
      const bool is_bias_present =
          mapping_args.node->inputs->size == 3 &&
          mapping_args.node->inputs->data[2] != kTfLiteOptionalTensor;
      if (!is_bias_present) {
        const int input_tensor_id =
            mapping_args.node->inputs->data[/*kInputTensor*/ 0];
        const int filter_tensor_id =
            mapping_args.node->inputs->data[/*kWeightsTensor*/ 1];
        const int num_units =
            mapping_args.context->tensors[filter_tensor_id].dims->data[0];
        add_zero_bias(input_tensor_id, filter_tensor_id, num_units);
      }
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_FULLY_CONNECTED;
    } break;
    case kTfLiteBuiltinHardSwish: {
      *nn_op_type = NEURON_HARD_SWISH;
    } break;
    case kTfLiteBuiltinSoftmax: {
      auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      // Optional scalar specifying the dimension the activation would be
      // performed on is not added. Default to -1.
      *nn_op_type = NEURON_SOFTMAX;
    } break;
    case kTfLiteBuiltinReshape: {
      if (mapping_args.node->inputs->size == 1) {
        // if no new_shape tensor, construct the new shape from params.
        auto* params = reinterpret_cast<TfLiteReshapeParams*>(
            mapping_args.node->builtin_data);
        int num_dimensions = params->num_dimensions;
        std::vector<int32_t> output_shape(num_dimensions);
        for (int i = 0; i < num_dimensions; ++i) {
          output_shape[i] = params->shape[i];
        }
        mapping_args.builder->AddVectorInt32Operand(
            output_shape.data(), static_cast<uint32_t>(num_dimensions));
      }
      *nn_op_type = NEURON_RESHAPE;
    } break;
    case kTfLiteBuiltinResizeBilinear: {
      const int output_id = mapping_args.node->outputs->data[0];
      auto& output = mapping_args.context->tensors[output_id];
      const int output_height = output.dims->data[1];
      const int output_width = output.dims->data[2];
      mapping_args.builder->AddScalarInt32Operand(output_width);
      mapping_args.builder->AddScalarInt32Operand(output_height);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      auto builtin = reinterpret_cast<TfLiteResizeBilinearParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
      mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      *nn_op_type = NEURON_RESIZE_BILINEAR;
    } break;
    case kTfLiteBuiltinResizeNearestNeighbor: {
      const TfLiteTensor& new_shape =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      // Neuron uses scalar inputs for height and width.
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[1]);
      mapping_args.builder->AddScalarInt32Operand(new_shape.data.i32[0]);
      mapping_args.builder->AddScalarBoolOperand(false);  // Use NHWC format
      auto builtin = reinterpret_cast<TfLiteResizeNearestNeighborParams*>(
          mapping_args.node->builtin_data);
      if (builtin->align_corners == true ||
          builtin->half_pixel_centers == true) {
        mapping_args.builder->AddScalarBoolOperand(builtin->align_corners);
        mapping_args.builder->AddScalarBoolOperand(builtin->half_pixel_centers);
      }
      *nn_op_type = NEURON_RESIZE_NEAREST_NEIGHBOR;
    } break;
    case kTfLiteBuiltinSqueeze: {
      auto builtin = reinterpret_cast<TfLiteSqueezeParams*>(
          mapping_args.node->builtin_data);
      // Note that we add the squeeze dimensions even if the dimensions
      // were unspecified (empty), as Neuron requires the operand.
      mapping_args.builder->AddVectorInt32Operand(
          builtin->num_squeeze_dims ? builtin->squeeze_dims : nullptr,
          static_cast<uint32_t>(builtin->num_squeeze_dims));
      *nn_op_type = NEURON_SQUEEZE;
    } break;
    case kTfLiteBuiltinL2Normalization: {
      *nn_op_type = NEURON_L2_NORMALIZATION;
    } break;
    case kTfLiteBuiltinLocalResponseNormalization: {
      auto builtin = reinterpret_cast<TfLiteLocalResponseNormParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->radius);
      mapping_args.builder->AddScalarFloat32Operand(builtin->bias);
      mapping_args.builder->AddScalarFloat32Operand(builtin->alpha);
      mapping_args.builder->AddScalarFloat32Operand(builtin->beta);
      *nn_op_type = NEURON_LOCAL_RESPONSE_NORMALIZATION;
    } break;
    case kTfLiteBuiltinConcatenation: {
      auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(
          mapping_args.node->builtin_data);
      int axis = builtin->axis < 0
                     ? mapping_args.context
                               ->tensors[mapping_args.node->inputs->data[0]]
                               .dims->size +
                           builtin->axis
                     : builtin->axis;
      mapping_args.builder->AddScalarInt32Operand(axis);
      *nn_op_type = NEURON_CONCATENATION;
    } break;
    case kTfLiteBuiltinDequantize: {
      *nn_op_type = NEURON_DEQUANTIZE;
    } break;
    case kTfLiteBuiltinFloor: {
      *nn_op_type = NEURON_FLOOR;
    } break;
    case kTfLiteBuiltinRelu: {
      *nn_op_type = NEURON_RELU;
    } break;
    case kTfLiteBuiltinReluN1To1: {
      *nn_op_type = NEURON_RELU1;
    } break;
    case kTfLiteBuiltinRelu6: {
      *nn_op_type = NEURON_RELU6;
    } break;
    case kTfLiteBuiltinLogistic: {
      *nn_op_type = NEURON_LOGISTIC;
    } break;
    case kTfLiteBuiltinTanh: {
      *nn_op_type = NEURON_TANH;
    } break;
    case kTfLiteBuiltinSub: {
      auto builtin =
          reinterpret_cast<TfLiteSubParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_SUB;
    } break;
    case kTfLiteBuiltinDiv: {
      auto builtin =
          reinterpret_cast<TfLiteDivParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->activation);
      *nn_op_type = NEURON_DIV;
    } break;
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinPadv2: {
      // We want to map to PAD as much as possible since it is more widely
      // supported. We map to PadV2 only when there is the need to specify
      // the padding value
      if (mapping_args.node->inputs->size == 2) {
        *nn_op_type = NEURON_PAD;
      } else {
        const int constant_value_id = mapping_args.node->inputs->data[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          *nn_op_type = NEURON_PAD;
        } else {
          *nn_op_type = NEURON_PAD_V2;
        }
      }
    } break;
    case kTfLiteBuiltinSpaceToBatchNd: {
      *nn_op_type = NEURON_SPACE_TO_BATCH_ND;
    } break;
    case kTfLiteBuiltinBatchToSpaceNd: {
      *nn_op_type = NEURON_BATCH_TO_SPACE_ND;
    } break;
    case kTfLiteBuiltinStridedSlice: {
      auto builtin = reinterpret_cast<TfLiteStridedSliceParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->begin_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->end_mask);
      mapping_args.builder->AddScalarInt32Operand(builtin->shrink_axis_mask);
      *nn_op_type = NEURON_STRIDED_SLICE;
    } break;
    case kTfLiteBuiltinTranspose: {
      *nn_op_type = NEURON_TRANSPOSE;
    } break;
    case kTfLiteBuiltinAbs: {
      *nn_op_type = NEURON_ABS;
    } break;
    case kTfLiteBuiltinExp: {
      *nn_op_type = NEURON_EXP;
    } break;
    case kTfLiteBuiltinLog: {
      *nn_op_type = NEURON_LOG;
    } break;
    case kTfLiteBuiltinPow: {
      *nn_op_type = NEURON_POW;
    } break;
    case kTfLiteBuiltinSlice: {
      *nn_op_type = NEURON_SLICE;
    } break;
    case kTfLiteBuiltinTransposeConv: {
      int input_tensor_flags = 0;
      const int input_tensor_id =
          mapping_args.node->inputs->data[/*kDataInputTensor*/ 2];
      const int weight_tensor_id =
          mapping_args.node->inputs->data[/*kWeightsTensor*/ 1];

      // Transpose convolution doesn't have hybrid variation.
      const bool hybrid_op = false;

      mapping_args.builder->AddTensorInput(
          input_tensor_id, hybrid_op,
          input_tensor_flags | NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED);

      // Transpose convlution uses per-channel quantization with int8 inputs
      // even if the number of channels in quantization parameters is equal to 1
      // (as opposed to conv2d, which uses per-tensor quantization in this
      // case).
      mapping_args.builder->AddTensorInput(
          weight_tensor_id, hybrid_op,
          input_tensor_flags | NN_TENSOR_FLAG_FORCE_PER_CHANNEL);

      const bool is_bias_present =
          mapping_args.node->inputs->size == 4 &&
          mapping_args.node->inputs->data[/*kBiasTensor*/ 3] !=
              kTfLiteOptionalTensor;

      if (is_bias_present) {
        mapping_args.builder->AddTensorInput(
            mapping_args.node->inputs->data[/*kBiasTensor*/ 3], hybrid_op);
      } else {
        const TfLiteTensor& output_shape =
            mapping_args.context->tensors[mapping_args.node->inputs
                                              ->data[/*kOutputShapeTensor*/ 0]];
        const int output_depth = output_shape.data.i32[3];
        add_zero_bias(input_tensor_id, weight_tensor_id, output_depth);
      }
      mapping_args.builder->AddTensorInput(
          mapping_args.node->inputs->data[/*kOutputShapeTensor*/ 0], hybrid_op);

      auto builtin = reinterpret_cast<TfLiteTransposeConvParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->padding);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_width);
      mapping_args.builder->AddScalarInt32Operand(builtin->stride_height);
      mapping_args.builder->AddScalarInt32Operand(
          /*NEURON_FUSED_NONE*/ 0);
      // Use NHWC layout for input and output.
      mapping_args.builder->AddScalarBoolOperand(false);
      *nn_op_type = NEURON_TRANSPOSE_CONV_2D;
    } break;
    case kTfLiteBuiltinSqrt: {
      *nn_op_type = NEURON_SQRT;
    } break;
    case kTfLiteBuiltinSpaceToDepth: {
      auto builtin = reinterpret_cast<TfLiteSpaceToDepthParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = NEURON_SPACE_TO_DEPTH;
    } break;
    case kTfLiteBuiltinMean: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      int32_t keep_dims = 0;
      if (builtin->keep_dims) keep_dims = 1;
      mapping_args.builder->AddScalarInt32Operand(keep_dims);
      *nn_op_type = NEURON_MEAN;
    } break;
    case kTfLiteBuiltinEmbeddingLookup: {
      *nn_op_type = NEURON_EMBEDDING_LOOKUP;
    } break;
    case kTfLiteBuiltinHashtableLookup: {
      *nn_op_type = NEURON_HASHTABLE_LOOKUP;
    } break;
    case kTfLiteBuiltinMaximum: {
      *nn_op_type = NEURON_MAXIMUM;
    } break;
    case kTfLiteBuiltinMinimum: {
      *nn_op_type = NEURON_MINIMUM;
    } break;
    case kTfLiteBuiltinCast: {
      *nn_op_type = NEURON_CAST;
    } break;
    case kTfLiteBuiltinLeakyRelu: {
      const auto input_type =
          mapping_args.context->tensors[mapping_args.node->inputs->data[0]]
              .type;
      auto builtin = reinterpret_cast<TfLiteLeakyReluParams*>(
          mapping_args.node->builtin_data);

      TfLiteTensor alpha_tensor;
      alpha_tensor.type = input_type;
      alpha_tensor.allocation_type = kTfLiteDynamic;
      alpha_tensor.dims = TfLiteIntArrayCreate(1);
      alpha_tensor.dims->data[0] = 1;
      alpha_tensor.params.zero_point = 0;

      int new_tensor_index = -1;
      if (input_type == kTfLiteFloat32) {
        alpha_tensor.params.scale = 0;
        std::vector<float> alpha_value = {builtin->alpha};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_FLOAT32, kTfLiteFloat32, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      } else if (input_type == kTfLiteInt8) {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<int8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_QUANT8_ASYMM_SIGNED, kTfLiteInt8, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      } else {
        alpha_tensor.params.scale = builtin->alpha;
        std::vector<uint8_t> alpha_value = {1};
        mapping_args.builder->AddNewInputConstantTensor(
            NEURON_TENSOR_QUANT8_ASYMM, kTfLiteUInt8, alpha_tensor.dims,
            alpha_value, alpha_tensor.params, &new_tensor_index);
      }

      *nn_op_type = NEURON_PRELU;
    } break;
    case kTfLiteBuiltinPrelu: {
      *nn_op_type = NEURON_PRELU;
    } break;
    case kTfLiteBuiltinLogicalOr: {
      *nn_op_type = NEURON_LOGICAL_OR;
    } break;
    case kTfLiteBuiltinLogicalAnd: {
      *nn_op_type = NEURON_LOGICAL_AND;
    } break;
    case kTfLiteBuiltinLogicalNot: {
      *nn_op_type = NEURON_LOGICAL_NOT;
    } break;
    case kTfLiteBuiltinLess: {
      *nn_op_type = NEURON_LESS;
    } break;
    case kTfLiteBuiltinLessEqual: {
      *nn_op_type = NEURON_LESS_EQUAL;
    } break;
    case kTfLiteBuiltinGreater: {
      *nn_op_type = NEURON_GREATER;
    } break;
    case kTfLiteBuiltinGreaterEqual: {
      *nn_op_type = NEURON_GREATER_EQUAL;
    } break;
    case kTfLiteBuiltinEqual: {
      *nn_op_type = NEURON_EQUAL;
    } break;
    case kTfLiteBuiltinNotEqual: {
      *nn_op_type = NEURON_NOT_EQUAL;
    } break;
    case kTfLiteBuiltinNeg: {
      *nn_op_type = NEURON_NEG;
    } break;
    case kTfLiteBuiltinTopkV2: {
      const TfLiteTensor& k_param =
          mapping_args.context->tensors[mapping_args.node->inputs->data[1]];
      mapping_args.builder->AddScalarInt32Operand(*k_param.data.i32);
      *nn_op_type = NEURON_TOPK_V2;
    } break;
    case kTfLiteBuiltinSelect: {
      *nn_op_type = NEURON_SELECT;
    } break;
    case kTfLiteBuiltinGather: {
      auto builtin = reinterpret_cast<TfLiteGatherParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddTensorInput(mapping_args.node->inputs->data[0],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      mapping_args.builder->AddScalarInt32Operand(builtin->axis);

      mapping_args.builder->AddTensorInput(mapping_args.node->inputs->data[1],
                                           /* hybrid_op */ false,
                                           /* scalar_as_tensor */ false);

      *nn_op_type = NEURON_GATHER;
    } break;
    case kTfLiteBuiltinSplit: {
      const TfLiteTensor& axis =
          mapping_args.context->tensors[mapping_args.node->inputs->data[0]];
      auto builtin =
          reinterpret_cast<TfLiteSplitParams*>(mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(*axis.data.i32);
      mapping_args.builder->AddScalarInt32Operand(builtin->num_splits);
      *nn_op_type = NEURON_SPLIT;
    } break;

    case kTfLiteBuiltinQuantize: {
      /*
            // TODO(Code): Add AddDequantize here to avoid Neuron adapter error
            // temporally Check if we can remove the AddDequantize latter, since
            // Neuron adapter should support requantize directly
      */
      auto input_index = mapping_args.node->inputs->data[0];
      auto output_index = mapping_args.node->outputs->data[0];
      if (IsQuantized(mapping_args.context->tensors[input_index].type) &&
          IsQuantized(mapping_args.context->tensors[output_index].type) &&
          mapping_args.context->tensors[input_index].type !=
              mapping_args.context->tensors[output_index].type) {
        // Use requantize instead of quantize
        const char* custom_name = "requantizemtk";
        size_t oem_scalar_size = 0;
        uint8_t* oem_scalar = nullptr;
        oem_scalar_size = ::tflite::mtk::PackOemScalarString(
                                custom_name,
                                &oem_scalar);
        if (oem_scalar != nullptr) {
          mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                    oem_scalar_size);
          free(oem_scalar);
        }
        *nn_op_type = NEURON_OEM_OPERATION;
        if (android_sdk_version > kMinSdkVersionForNeuron13) {
             neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                     kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                            nn_op_type);
        }
      } else {
        *nn_op_type = NEURON_QUANTIZE;
      }
    } break;
    case kTfLiteBuiltinReduceAny: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_ANY;
    } break;
    case kTfLiteBuiltinReduceMin: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_MIN;
    } break;
    case kTfLiteBuiltinReduceMax: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_MAX;
    } break;
    case kTfLiteBuiltinDepthToSpace: {
      auto builtin = reinterpret_cast<TfLiteDepthToSpaceParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->block_size);
      *nn_op_type = NEURON_DEPTH_TO_SPACE;
    } break;
    case kTfLiteBuiltinSum: {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarBoolOperand(builtin->keep_dims);
      *nn_op_type = NEURON_REDUCE_SUM;
    } break;
    case kTfLiteBuiltinSquaredDifference: {
      const char* custom_name = "MTKEXT_SQUARED_DIFFERENCE";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                  oem_scalar_size);
        free(oem_scalar);
      }
      *nn_op_type = NEURON_OEM_OPERATION;
      if (android_sdk_version > kMinSdkVersionForNeuron13) {
           neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                   kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                          nn_op_type);
      }
    } break;
    case kTfLiteBuiltinRsqrt: {
      const char* custom_name = "MTKEXT_RSQRT";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                  oem_scalar_size);
        free(oem_scalar);
      }
      *nn_op_type = NEURON_OEM_OPERATION;
      if (android_sdk_version > kMinSdkVersionForNeuron13) {
           neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                   kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                          nn_op_type);
      }
    } break;
    case kTfLiteBuiltinUnpack: {
      const char* custom_name = "unpackmtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      auto builtin = reinterpret_cast<TfLiteUnpackParams*>(
          mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->axis);
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                  oem_scalar_size);
        free(oem_scalar);
      }
      *nn_op_type = NEURON_OEM_OPERATION;
      if (android_sdk_version > kMinSdkVersionForNeuron13) {
           neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                   kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                          nn_op_type);
      }
    } break;
    case kTfLiteBuiltinReverseV2: {
      const char* custom_name = "reversemtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      oem_scalar_size =
          ::tflite::mtk::PackOemScalarString(custom_name, &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                  oem_scalar_size);
        free(oem_scalar);
      }
      *nn_op_type = NEURON_OEM_OPERATION;
      if (android_sdk_version > kMinSdkVersionForNeuron13) {
           neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                   kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                          nn_op_type);
      }
    } break;
    case kTfLiteBuiltinMirrorPad: {
      const char* custom_name = "mirrorpadmtk";
      size_t oem_scalar_size = 0;
      uint8_t* oem_scalar = nullptr;
      auto builtin = reinterpret_cast<TfLiteMirrorPaddingParams*>(
        mapping_args.node->builtin_data);
      mapping_args.builder->AddScalarInt32Operand(builtin->mode);
      oem_scalar_size = ::tflite::mtk::PackOemScalarString(
                              custom_name,
                              &oem_scalar);
      if (oem_scalar != nullptr) {
        mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                  oem_scalar_size);
        free(oem_scalar);
      }
      *nn_op_type = NEURON_OEM_OPERATION;
      if (android_sdk_version > kMinSdkVersionForNeuron13) {
           neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                   kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                          nn_op_type);
      }
    } break;
    case kTfLiteBuiltinCustom: {
        if (strcmp(custom_name, "MTK_OPT") == 0) {
          *nn_op_type = NEURON_OEM_OPERATION;
          if (android_sdk_version > kMinSdkVersionForNeuron13) {
              neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                      kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                             nn_op_type);
          }
        } else if (strcmp(custom_name, "MTK_DEPTH_TO_SPACE") == 0 ||
                   strcmp(custom_name, "MTKEXT_DEPTH_TO_SPACE") == 0) {
              uint32_t next_id =
                mapping_args.builder->GetNextNeuronOperandIdx();
              ::tflite::mtk::CustomOpHelper::MtkExtOpParameterFuncNeuron mtk_op_param_func =
                ::tflite::mtk::CustomOpHelper::GetInstance().GetMtkExtOpParamFuncNeuron(
                  custom_name);
              if (mtk_op_param_func != nullptr) {
                  mtk_op_param_func(mapping_args.builder->GetNnModel(),
                                    mapping_args.builder->GetAugmentedInputs(),
                                    next_id,
                                    mapping_args.node->user_data);
              }
              *nn_op_type = NEURON_DEPTH_TO_SPACE;
        } else if (strcmp(custom_name, "MTK_REQUANTIZE") == 0) {
            *nn_op_type = NEURON_QUANTIZE;
            return kTfLiteOk;
        } else if (strncmp(custom_name, "MTK", 3) == 0) {
            size_t oem_scalar_size = 0;
            uint8_t* oem_scalar = nullptr;
            uint32_t next_id = mapping_args.builder->GetNextNeuronOperandIdx();
            ::tflite::mtk::CustomOpHelper::MtkExtOpParameterFuncNeuron mtk_op_param_func =
              ::tflite::mtk::CustomOpHelper::GetInstance().GetMtkExtOpParamFuncNeuron(
                custom_name);
            if (mtk_op_param_func != nullptr) {
                mtk_op_param_func(mapping_args.builder->GetNnModel(),
                                  mapping_args.builder->GetAugmentedInputs(),
                                  next_id,
                                  mapping_args.node->user_data);
            }
            mapping_args.builder->UpdateNeuronOperandIdx(next_id);
            oem_scalar_size = ::tflite::mtk::PackOemScalarString(
                                      custom_name,
                                      &oem_scalar);
            if (oem_scalar != nullptr) {
              mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                        oem_scalar_size);
              free(oem_scalar);
            }
            *nn_op_type = NEURON_OEM_OPERATION;
             if (android_sdk_version > kMinSdkVersionForNeuron13) {
                 neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                         kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                                nn_op_type);
             }
        } else if (strncmp(custom_name, "mcv_", 4) == 0 || strncmp(custom_name, "cust_", 4) == 0 ||
                strncmp(custom_name, "halidegen_", 4) == 0) {

            // Add mvpu custom tensor
            std::string jsonstr = "";
            if ((mapping_args.node->custom_initial_data != nullptr) &&
                                   (mapping_args.node->custom_initial_data_size > 0)) {
                // Convert custom option to a JSON-like string
                const flexbuffers::Reference& flexbuf = flexbuffers::GetRoot(
                        reinterpret_cast<const uint8_t*>(mapping_args.node->custom_initial_data),
                                             mapping_args.node->custom_initial_data_size);
                flexbuf.ToString(/* strings_quoted */ true, /* keys_quoted */ true, jsonstr);
            } else {
                jsonstr = "{}";
            }

            mapping_args.builder->AddScalarOemOperand(reinterpret_cast<const uint8_t*>(&jsonstr[0]), jsonstr.length()+1);

            // Add the OP name + target as the last input operand
            size_t oem_scalar_size = 0;
            uint8_t* oem_scalar = nullptr;
            std::string s(custom_name);
            s += ":common:mediatek";
            const char* target = s.c_str();
            oem_scalar_size = tflite::mtk::PackOemScalarString(target, &oem_scalar);
            if (oem_scalar != nullptr) {
              mapping_args.builder->AddScalarOemOperand(oem_scalar, oem_scalar_size);
              free(oem_scalar);
            }

            *nn_op_type = NEURON_OEM_OPERATION;
            if (android_sdk_version > kMinSdkVersionForNeuron13) {
                neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                        kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                               nn_op_type);
            }
        } else {
            // Process 3rd party custom OP for NNAPI 1.1 HAL
            uint32_t next_id =
              mapping_args.builder->GetNextNeuronOperandIdx();
            tflite::mtk::CustomOpHelper::ParameterFuncNeuron custom_op_param_func =
              ::tflite::mtk::CustomOpHelper::GetInstance().GetParamFuncNeuron(custom_name);
            custom_op_param_func(mapping_args.node->user_data,
                            mapping_args.builder->GetNnModel(),
                            mapping_args.builder->GetAugmentedInputs(),
                            next_id);
            mapping_args.builder->UpdateNeuronOperandIdx(next_id);
            // Add the OP name + target as the last input operand
            size_t oem_scalar_size = 0;
            uint8_t* oem_scalar = nullptr;
            const char* target =
                tflite::mtk::CustomOpHelper::GetInstance().FindTarget(custom_name);
            if (target != nullptr) {
              oem_scalar_size = tflite::mtk::PackOemScalarString(target,
                                                        &oem_scalar);
              if (oem_scalar != nullptr) {
                mapping_args.builder->AddScalarOemOperand(oem_scalar,
                                                          oem_scalar_size);
                free(oem_scalar);
              }
            }

            *nn_op_type = NEURON_OEM_OPERATION;
            if (android_sdk_version > kMinSdkVersionForNeuron13) {
                neuronapi_->NeuronModel_getExtensionOperationType(mapping_args.builder->GetNnModel(),
                        kExtensionGeneralOpration, ADAPTER_EXTENSION_GENERAL_OPERATION_TYPE,
                                                               nn_op_type);
            }
        }
    } break;
    default:
      // All other operators are not mapped.
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Unmapped OP: %d", builtin_code);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NeuronDelegateKernel::Init(TfLiteContext* context,
                                        const TfLiteDelegateParams* params) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Init");
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    nodes_.push_back(node_index);
  }
  if (!nn_model_) {
    if(!options_.neuron_model) {
        NeuronModel* model = nullptr;
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context,
                                            neuronapi_->NeuronModel_create(&model),
                                            "creating Neuron model");
        nn_model_.reset(model);
    } else {
      nn_model_.reset(*options_.neuron_model);
    }
    TF_LITE_ENSURE_STATUS(
        BuildGraph(context, params->input_tensors, params->output_tensors));
  }

  if (options_.cache_dir != nullptr) {
      nn_compilation_cache_token_.clear();
      int bytes_count = 0;
      for (int i=0; i<context->tensors_size; i++) {
          bytes_count += context->tensors[i].bytes;
      }
      std::string bytes_count_str = std::to_string(bytes_count);
      const char* model_token = bytes_count_str.c_str();
      // Compilation caching could be enabled, try construct the uint8
      // token.
      // TODO(b/133342794): use a generic token generator class.
      uint64_t token_parts[4];
      // Create bits from model_token.
      // Using farmhash fingerprint instead of std::hash, as the latter is not
      // guaranteed to be stable across program invocations.
      token_parts[0] =
          farmhash::Fingerprint64(model_token, std::strlen(model_token));
      // Create bits from params->nodes_to_replace.
      token_parts[1] = GetHash(params->nodes_to_replace);
      // Create bits from params->input_tensors. These include the input tensor
      // sizes, as the cached compilations are size-dependent.
      token_parts[2] = GetInOutPutHash(context, params->input_tensors);
      for (int i : TfLiteIntArrayView(params->input_tensors)) {
        if (i != kTfLiteOptionalTensor) {
          TfLiteTensor* t = &context->tensors[i];
          TF_LITE_ENSURE(context, t->dims);
          token_parts[2] = GetHash(t->dims, token_parts[2]);
        }
      }
      // bits from params->output_tensors.
      token_parts[3] = GetInOutPutHash(context, params->output_tensors);
      // NNAPI requires the token to be 256bit long.
      // TODO(b/172238515): get token size from header instead of
      // hardcoding.
      std::vector<uint8_t> nnapi_cache_token(32, 0);
      // Copy the token bits.
      uint8_t* p = reinterpret_cast<uint8_t*>(token_parts);
      for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
        nnapi_cache_token[i] = p[i];
      }
      nn_compilation_cache_token_ = nnapi_cache_token;
  }

  need_set_memory = true;
  initialised_ = true;
  return kTfLiteOk;
}

TfLiteStatus NeuronDelegateKernel::Prepare(TfLiteContext* context,
                                           TfLiteNode* node) {
  if (!initialised_) {
    return kTfLiteError;
  }

  if (nn_compilation_) {
    return kTfLiteOk;
  }

  NeuronCompilation* compilation = nullptr;
  uint32_t num_devices = 0;

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->Neuron_getDeviceCount(&num_devices), "getting device count");

  NeuronDevice* device[num_devices];

  int device_count = 0;
  if (options_.accelerator_name != nullptr) {
    char* device_split_name = std::strtok(const_cast<char*>(options_.accelerator_name), ",");
    while(device_split_name) {
        NeuronDevice* mtk_device = nullptr;
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                        "NeuronCompilation_createForDevices: %s",
                        device_split_name);
        if (GetAcceleratorDevice(neuronapi_, context, device_split_name, &mtk_device) == kTfLiteOk) {
          device[device_count] = mtk_device;
          device_count++;
        }
        device_split_name = std::strtok(NULL, ",");
    }
  }
  if (device_count) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_createForDevices(nn_model_.get(), device,
                                                       device_count, &compilation),
        "creating Neuron compilation user specified device");
  } else {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_create(nn_model_.get(), &compilation),
        "creating Neuron compilation");
  }

  int result = neuronapi_->NeuronCompilation_setPreference(
      compilation, options_.execution_preference);
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "NeuronCompilation_setPreference: %d",
                  options_.execution_preference);
  if (result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, result,
                                      "setting compilation preferences");
  result = neuronapi_->NeuronCompilation_setPriority(
      compilation, options_.execution_priority);
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_setPriority: %d",
                  options_.execution_priority);
  if (result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, result,
                                      "setting compilation priority");
  if (neuronapi_->NeuronCompilation_setOptimizationHint != nullptr) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_setOptimizationHint: %d",
                  options_.optimization_hint);
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setOptimizationHint(
            compilation, options_.optimization_hint),
        "set optimization hint");
  }
  if (options_.cache_dir != nullptr && !nn_compilation_cache_token_.empty()) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_setCaching: %s",
                  options_.cache_dir);
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setCaching(
            compilation, options_.cache_dir,
            nn_compilation_cache_token_.data()),
        "set optimization hint");
  }

  const char* default_compile_options;
  if (options_.execution_preference == ExecutionPreference::kLowPower ||
          options_.execution_preference == ExecutionPreference::kSustainedSpeed) {
      default_compile_options = "--opt-bw";
  } else {
      default_compile_options = "--opt 3";
  }
  if (neuronapi_->android_sdk_version > kMinSdkVersionForNeuron13) {
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_setDefaultCompileOptions: %s",
                  default_compile_options);
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setOptimizationString(compilation, default_compile_options),
        "set default compile options");

    if (options_.compile_options != nullptr) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_setCompileOptions: %s",
                    options_.compile_options);
      RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
          context,
          neuronapi_->NeuronCompilation_setOptimizationString(compilation, options_.compile_options),
          "set compile options");
    }
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronCompilation_max partition: %d",
                options_.max_number_delegated_partitions);
  uint32_t apu_mem_size = 0;
  if (neuronapi_->Neuron_getL1MemorySizeKb(&apu_mem_size) == NEURON_NO_ERROR &&
      apu_mem_size > 0) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronCompilation_setL1MemorySizeKb(compilation,
                                                        apu_mem_size),
        "set apu l1 memory size");
  }

  if (options_.execution_preference == ExecutionPreference::kLowPower) {
    perf_handle_ = acquirePerformanceLock(perf_handle_, LOW_POWER_MODE, 2000);
  } else {
    perf_handle_ = acquirePerformanceLock(perf_handle_, FAST_COMPILE_MODE, 2000);
  }
  const int finish_result = neuronapi_->NeuronCompilation_finish(compilation);
  releasePerformanceLock(perf_handle_);
  perf_handle_ = 0;
  if (finish_result != NEURON_NO_ERROR) {
    neuronapi_->NeuronCompilation_free(compilation);
    compilation = nullptr;
  }

  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(context, finish_result,
                                      "completing Neuron compilation");
  nn_compilation_.reset(compilation);

#ifdef NEURON_REUSABLE_EXECUTION
  NeuronExecution* execution = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution),
      "creating Neuron execution");
  nn_execution_.reset(execution);
#endif

  return kTfLiteOk;
}

TfLiteStatus NeuronDelegateKernel::Eval(TfLiteContext* context,
                                        TfLiteNode* node) {
#ifdef NEURON_REUSABLE_EXECUTION
  NeuronExecution* execution = nn_execution_.get();
#else
  NeuronExecution* execution = nullptr;
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronExecution_create(nn_compilation_.get(), &execution),
      "creating Neuron execution");
  std::unique_ptr<NeuronExecution, NNFreeExecution> execution_unique_ptr(
      execution, NNFreeExecution(neuronapi_));
#endif

  // Set the input tensor buffers. Note: we access tflite tensors using
  // absolute indices but Neuron indices inputs by relative indices.
  int relative_input_index = 0;
  size_t input_offset = 0;
  TfLiteNeuronDelegateGetTensorMemoryMap(node->delegate, &tensor_memory_map_kernel_);
  for (auto absolute_input_index : TfLiteIntArrayView(node->inputs)) {
    if (absolute_input_index == kTfLiteOptionalTensor) {
      continue;
    }

    NeuronOperandType input_nn_operand_type;
    NeuronOperandType* input_nn_operand_type_ptr = nullptr;
    TfLiteTensor* tensor = &context->tensors[absolute_input_index];
    TfLiteType nn_type_equivalent =
        operand_mapping_.lite_index_to_neuron_type_conversion(
            absolute_input_index);
    if (tensor->allocation_type != kTfLiteMmapRo) {
      if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
          tensor->buffer_handle < tensor_memory_map_kernel_.size()) {
        if (need_set_memory) {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setInputFromMemory(
                    execution, relative_input_index, input_nn_operand_type_ptr,
                    tensor_memory_map_kernel_.at(tensor->buffer_handle), 0,
                    tensor->bytes),
                "associating Neuron execution input with a memory object");
        }
        relative_input_index++;
        continue;
      }
      int tensor_size = 0;
      if (nn_type_equivalent != kTfLiteNoType) {
        const auto num_elements = NumElements(tensor);
        uint8_t* input_ptr = nn_input_memory_->get_data_ptr() + input_offset;
        if (tensor->type == kTfLiteUInt8 &&
            nn_type_equivalent == kTfLiteInt32) {
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t>(tensor->data.uint8[i]);
          }
        } else if (tensor->type == kTfLiteInt8 &&
                   nn_type_equivalent == kTfLiteUInt8) {
          // Explicitly convert int8 values to uint8 values.
          for (int i = 0; i < num_elements; ++i) {
            input_ptr[i] = static_cast<const uint8_t>(
                static_cast<int32_t>(tensor->data.int8[i]) + 128);
          }
        } else if (tensor->type == kTfLiteInt8 &&
                   nn_type_equivalent == kTfLiteInt32) {
          for (int i = 0; i < num_elements; ++i) {
            reinterpret_cast<int32_t*>(input_ptr)[i] =
                static_cast<const int32_t>(tensor->data.int8[i]) + 128;
          }
        } else {
          context->ReportError(
              context,
              "Neuron Delegate: unsupported tensor types conversion: "
              "from type code %d to type code %d.\n",
              tensor->type, nn_type_equivalent);
          return kTfLiteError;
        }
        size_t type_size;
        TF_LITE_ENSURE_OK(
            context, GetSizeOfType(context, nn_type_equivalent, &type_size));
        tensor_size = NumElements(tensor) * type_size;
        if (nn_input_memory_->get_handle()) {
            if (need_set_memory) {
                RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                    context,
                    neuronapi_->NeuronExecution_setInputFromMemory(
                        execution, relative_input_index, input_nn_operand_type_ptr,
                        nn_input_memory_->get_handle(), input_offset, tensor_size),
                    "associating Neuron execution input with a memory object");
            }
        } else {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setInput(
                    execution, relative_input_index, input_nn_operand_type_ptr,
                    input_ptr, tensor_size),
                "associating Neuron execution input with a memory object");
        }
      } else {
        // copy data to pre-allocated shared memory.
        memcpy(nn_input_memory_->get_data_ptr() + input_offset,
               tensor->data.raw, tensor->bytes);
        if (nn_input_memory_->get_handle()) {
            if (need_set_memory) {
                RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                    context,
                    neuronapi_->NeuronExecution_setInputFromMemory(
                        execution, relative_input_index, input_nn_operand_type_ptr,
                        nn_input_memory_->get_handle(), input_offset, tensor->bytes),
                    "associating Neuron execution input with a memory object");
            }
        } else {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setInput(
                    execution, relative_input_index, input_nn_operand_type_ptr,
                    nn_input_memory_->get_data_ptr() + input_offset, tensor->bytes),
                "associating Neuron execution input with a memory object");
        }
        tensor_size = tensor->bytes;
      }

      input_offset += tensor_size;
      input_offset += getNumPaddingBytes(tensor_size);
      relative_input_index++;
    }
  }

  // Set the output tensor buffers.
  int relative_output_index = 0;
  size_t output_offset = 0;
  for (auto output_index : TfLiteIntArrayView(node->outputs)) {
    // If the Neuron implementation doesn't have some of the outputs
    // they are left unmapped and we should not try to read their value here
    if (operand_mapping_.lite_index_to_neuron(output_index) == -1) {
      continue;
    }

    NeuronOperandType output_nn_operand_type;
    NeuronOperandType* output_nn_operand_type_ptr = nullptr;
    TfLiteTensor* tensor = &context->tensors[output_index];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->buffer_handle < tensor_memory_map_kernel_.size()) {
      if (need_set_memory) {
          RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
              context,
              neuronapi_->NeuronExecution_setOutputFromMemory(
                  execution, relative_output_index, output_nn_operand_type_ptr,
                  tensor_memory_map_kernel_.at(tensor->buffer_handle), 0,
                  tensor->bytes),
              "associating NNAPI execution output to a memory object");
      }
    } else {
        if (nn_output_memory_->get_handle()) {
            if (need_set_memory) {
                RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                    context,
                    neuronapi_->NeuronExecution_setOutputFromMemory(
                        execution, relative_output_index, output_nn_operand_type_ptr,
                        nn_output_memory_->get_handle(), output_offset, tensor->bytes),
                    "associating NNAPI execution output to a memory object");
            }
        } else {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setOutput(
                    execution, relative_output_index, output_nn_operand_type_ptr,
                    nn_output_memory_->get_data_ptr() + output_offset, tensor->bytes),
                "associating NNAPI execution output to a memory object");
        }
        output_offset += tensor->bytes;
        output_offset += getNumPaddingBytes(tensor->bytes);
    }
    relative_output_index++;
  }

  // The state_out of previous invocation need to be mapped to state_in of
  // current invocation.
  for (size_t i = 0; i < model_state_tfl_inputs_.size(); i++) {
    int state_tensor_idx = model_state_tfl_inputs_[i];
    TfLiteTensor* tensor = &context->tensors[state_tensor_idx];
    if (nn_output_memory_->get_handle()) {
        if (need_set_memory) {
            RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
                context,
                neuronapi_->NeuronExecution_setOutputFromMemory(
                    execution, relative_output_index, nullptr,
                    nn_output_memory_->get_handle(), output_offset, tensor->bytes),
                "associating Neuron execution output to a buffer");
        }
    } else {
        RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
            context,
            neuronapi_->NeuronExecution_setOutput(
                execution, relative_output_index, nullptr,
                nn_output_memory_->get_data_ptr() + output_offset, tensor->bytes),
            "associating Neuron execution output to a buffer");
    }
    output_offset += tensor->bytes;
    output_offset += getNumPaddingBytes(tensor->bytes);
    relative_output_index++;
  }

  // Use synchronous execution
  uint32_t boost_duration =
      options_.boost_duration == 0 ? 2000 : options_.boost_duration;
  PERFORMANCE_MODE_E performance_mode = FAST_SINGLE_ANSWER_MODE;
  if (options_.execution_preference == ExecutionPreference::kSustainedSpeed) {
    performance_mode = SUSTAINED_SPEED_MODE;
  } else if (options_.execution_preference == ExecutionPreference::kLowPower) {
    performance_mode = LOW_POWER_MODE;
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Neuron set boost duration: %d",
                options_.boost_duration);
  perf_handle_ =
      acquirePerformanceLock(perf_handle_, performance_mode, boost_duration);
  if (options_.boost_value >= 0) {
    RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
        context,
        neuronapi_->NeuronExecution_setBoostHint(execution,
                                                 options_.boost_value),
        "setting computation boost hint");
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronExecution_setBoostHint: %d",
                    options_.boost_value);
  }
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronExecution_compute(execution),
      "running computation");
  // copy results from shared memory to the destination.
  output_offset = 0;
  for (auto output_index : TfLiteIntArrayView(node->outputs)) {
    TfLiteTensor* tensor = &context->tensors[output_index];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
      continue;
    }
    TfLiteType nn_type_equivalent =
        operand_mapping_.lite_index_to_neuron_type_conversion(output_index);
    if (tensor->type == kTfLiteInt8 && nn_type_equivalent == kTfLiteUInt8) {
      // Explicitly convert uint8 values to int8 values.
      uint8_t* output_ptr = reinterpret_cast<uint8_t*>(
          nn_output_memory_->get_data_ptr() + output_offset);
      const auto num_elements = NumElements(tensor);
      for (int i = 0; i < num_elements; ++i) {
        output_ptr[i] =
            static_cast<uint8_t>(static_cast<int32_t>(output_ptr[i]) - 128);
      }
    }
    memcpy(tensor->data.raw, nn_output_memory_->get_data_ptr() + output_offset,
           tensor->bytes);
    output_offset += tensor->bytes;
    output_offset += getNumPaddingBytes(tensor->bytes);
  }

  need_set_memory = false;
  // copy output of all output tensors in feedback_loops_ into the
  // associated input
  for (auto feedback_loop : feedback_loops_) {
    int output_tensor_idx;
    int input_tensor_idx;
    std::tie(output_tensor_idx, input_tensor_idx) = feedback_loop;
    TfLiteTensor* src =
        &context->tensors[node->outputs->data[output_tensor_idx]];
    TfLiteTensor* dest =
        &context->tensors[node->inputs->data[input_tensor_idx]];

    memcpy(dest->data.raw, src->data.raw, src->bytes);
  }

  return kTfLiteOk;
}

void NeuronDelegateKernel::AddDequantizeOperatorsWhereNeeded(
    const TfLiteContext* context, int builtin_code, const TfLiteNode* node,
    NeuronOpBuilder* builder) {
  int input_tensor_index = -1;
  std::vector<int> inputs_to_potentially_dequantize;

  switch (builtin_code) {
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinFullyConnected: {
      input_tensor_index = 0;
      // Weights and bias are inputs #1 and #2 respectively and may require
      // dequantization.
      inputs_to_potentially_dequantize = {1, 2};
      break;
    }
    case kTfLiteBuiltinLstm: {
      input_tensor_index = 0;
      inputs_to_potentially_dequantize = {1,  2,  3,  4,  5,  6,  7,
                                          8,  9,  10, 11, 12, 13, 14,
                                          15, 16, 17, 20, 21, 22, 23};
      break;
    }
    default:
      return;
  }

  int tensor_id = node->inputs->data[input_tensor_index];
  if (tensor_id < 0) return;

  // Nothing to do if the input is not floating-point.
  if (!IsFloat(context->tensors[tensor_id].type)) return;

  for (int i : inputs_to_potentially_dequantize) {
    if (i < 0 || i >= node->inputs->size) continue;  // Ignore invalid index.
    tensor_id = node->inputs->data[i];
    if (tensor_id < 0) continue;  // Ignore optional input.

    const TfLiteType type = context->tensors[tensor_id].type;
    // Nothing to do for this tensor if it's not quantized.
    if (!IsQuantized(type)) continue;

    // Insert Dequantize operator if it hasn't been done already and change
    // the node's input accordingly.
    builder->AddDequantize(i, node->inputs->data[i], type);
  }
}

TfLiteStatus NeuronDelegateKernel::AddOpsAndTensors(TfLiteContext* context) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "AddOpsAndTensors");
  DequantizeMapping dequantize_mapping;
  // The operand builder allows creating a single op. It is created outside
  // the for loop to avoid reallocating the vectors.
  NeuronOpBuilder builder(neuronapi_, context, &operand_mapping_,
                          &dequantize_mapping, nn_model_.get());
  if(options_.neuron_model) {
      operand_mapping_.set_neuron_current_index(*options_.current_neuron_index);
      for (int i : static_cast<Subgraph*>(context->impl_)->inputs()) {
        operand_mapping_.set_tflite_index_to_neuron(i, *options_.neuron_input_index++);
      }

      for (int i : static_cast<Subgraph*>(context->impl_)->outputs()) {
        operand_mapping_.set_tflite_index_to_neuron(i, *options_.neuron_output_index++);
      }
  }
  // Add Tensors.
  for (auto node_index : nodes_) {
    // Obtain the op and registration.
    TfLiteNode* node;
    TfLiteRegistration* reg;
    TF_LITE_ENSURE_STATUS(
        context->GetNodeAndRegistration(context, node_index, &node, &reg));
    /// M: NeuroPilot {@
    bool skip = false;
    if (static_cast<TfLiteBuiltinOperator>(reg->builtin_code) ==
        kTfLiteBuiltinCustom &&
        reg->custom_name != nullptr &&
        strcmp(reg->custom_name, "MTK_OPT") == 0) {
      for (size_t idx = node_index + 1; idx < nodes_.size(); idx++) {
        TfLiteNode* target_node;
        TfLiteRegistration* target_reg;
        context->GetNodeAndRegistration(context, idx,
                                        &target_node, &target_reg);
        // If the MTK OP is connected by multiple OPs
        // ==> The MTK OP node should be put into fusion list multiple times.
        for (auto m = 0; m < target_node->inputs->size; m++) {
          if (node->outputs->data[0] == target_node->inputs->data[m]) {
            fusion_mtk_op_list_.push_back(*node);
          }
        }
      }
      skip = true;
    } else {
      if (!fusion_mtk_op_list_.empty()) {
        bool erased = false;
        std::vector<TfLiteNode>::iterator it = fusion_mtk_op_list_.begin();
        while (it != fusion_mtk_op_list_.end()) {
          erased = false;
          for (auto j = 0; j < node->inputs->size; j++) {
            if (it->outputs->data[0] == node->inputs->data[j]) {
              TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "Change builtin OP(%d):%s's input index:%d to index:%d",
                  reg->builtin_code,
                  EnumNameBuiltinOperator(
                    static_cast<BuiltinOperator>(reg->builtin_code)),
                  node->inputs->data[j],
                  it->inputs->data[0]);
              // Set the input tensor of mtk OP as the input tensor
              // of previous OP
              node->inputs->data[j] = it->inputs->data[0];
              it = fusion_mtk_op_list_.erase(it);
              erased = true;
            }
          }
          if (!erased) {
            ++it;
          }
        }
      }
    }

    if (skip) {
      continue;
    }

    // Delegate PACK by lowering it into CONCAT + RESHAPE.
    if (reg->builtin_code == kTfLiteBuiltinPack) {
      TF_LITE_ENSURE_STATUS(builder.TransformPackIntoSupportedOps(node, reg));
      continue;
    }
    // Set same quantization params of input & output tensors for invalid MirrorPad
    if(reg->builtin_code == kTfLiteBuiltinMirrorPad){
      auto input_index = node->inputs->data[0];
      auto output_index = node->outputs->data[0];
      TfLiteTensor input_tensor = context->tensors[input_index];
      TfLiteTensor& output_tensor = context->tensors[output_index];
      if (IsQuantized(input_tensor.type) && IsQuantized(output_tensor.type)){
        if (input_tensor.quantization.type == kTfLiteAffineQuantization){
          auto* src_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
              input_tensor.quantization.params);
          const size_t num_scales = src_quantization->scale->size;
          auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
              malloc(sizeof(TfLiteAffineQuantization)));
          affine_quantization->scale = TfLiteFloatArrayCreate(num_scales);
          affine_quantization->zero_point = TfLiteIntArrayCreate(num_scales);
          for (size_t i = 0; i < num_scales; ++i) {
            affine_quantization->scale->data[i] =
                src_quantization->scale->data[i];
            affine_quantization->zero_point->data[i] =
                src_quantization->zero_point->data[i];
          }
          affine_quantization->quantized_dimension =
              src_quantization->quantized_dimension;
          output_tensor.quantization.params =
              reinterpret_cast<void*>(affine_quantization);
          output_tensor.quantization.type = input_tensor.quantization.type;
          output_tensor.params.scale = input_tensor.params.scale;
          output_tensor.params.zero_point = input_tensor.params.zero_point;
        }
      }
    }
    /// M: NeuroPilot @}

    const bool hybrid_op = IsHybridOperator(context, reg->builtin_code, node);
    const bool scalar_as_tensor = IsScalarInputSupported(reg->builtin_code);
    //No need int8 conversion after sdk version 30
    const bool need_int8_conversion = false;
    const bool use_int8_asymm_signed = true;

    int input_tensor_flags = 0;
    if (scalar_as_tensor) {
      input_tensor_flags |= NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
    }
    if (use_int8_asymm_signed) {
      input_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }
    // h_swish will be lowered into supported Neuron operations.
    if (reg->builtin_code == kTfLiteBuiltinHardSwish) {
      builder.AddHardSwish(node->inputs->data[0], node->outputs->data[0],
                           need_int8_conversion);
      continue;
    }
    bool is_asymm_to_symm = false;
    if (reg->builtin_code == kTfLiteBuiltinCustom && strncmp(reg->custom_name, "MTK", 3) == 0) {
        for (int mtk_ext_input_pos = 0; mtk_ext_input_pos < node->inputs->size; ++mtk_ext_input_pos) {
            const auto input_index = node->inputs->data[mtk_ext_input_pos];
            TfLiteTensor* tensor = &context->tensors[input_index];
            if (tensor->type == kTfLiteInt8 && tensor->allocation_type != kTfLiteMmapRo) {
                is_asymm_to_symm = false;
                break;
            } else if (tensor->type == kTfLiteInt8 && tensor->allocation_type == kTfLiteMmapRo)  {
                is_asymm_to_symm = true;
            }
        }
    }

    // Map inputs to Neuron tensor indices.
    for (int input_pos = 0; input_pos < node->inputs->size; ++input_pos) {
      if (reg->builtin_code == kTfLiteBuiltinTransposeConv) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinFullyConnected &&
          node->inputs->data[input_pos] == kTfLiteOptionalTensor) {
        // skip optional bias and handle it during mapping
        continue;
      }
      const auto input_index = node->inputs->data[input_pos];
      if (need_int8_conversion &&
          (input_pos == 0 ||
           reg->builtin_code == kTfLiteBuiltinFullyConnected ||
           reg->builtin_code == kTfLiteBuiltinConv2d ||
           reg->builtin_code == kTfLiteBuiltinDepthwiseConv2d ||
           reg->builtin_code == kTfLiteBuiltinAdd ||
           reg->builtin_code == kTfLiteBuiltinMul ||
           reg->builtin_code == kTfLiteBuiltinSub ||
           reg->builtin_code == kTfLiteBuiltinConcatenation ||
           reg->builtin_code == kTfLiteBuiltinMaximum ||
           reg->builtin_code == kTfLiteBuiltinMinimum ||
           reg->builtin_code == kTfLiteBuiltinLeakyRelu ||
           reg->builtin_code == kTfLiteBuiltinLess ||
           reg->builtin_code == kTfLiteBuiltinLessEqual ||
           reg->builtin_code == kTfLiteBuiltinPrelu ||
           reg->builtin_code == kTfLiteBuiltinGreater ||
           reg->builtin_code == kTfLiteBuiltinGreaterEqual ||
           reg->builtin_code == kTfLiteBuiltinEqual ||
           reg->builtin_code == kTfLiteBuiltinNotEqual ||
           reg->builtin_code == kTfLiteBuiltinSelect)) {
        // Only selected inputs require int8 conversion.
        TF_LITE_ENSURE_STATUS(builder.AddTensorInput(
            input_index, hybrid_op,
            input_tensor_flags | NN_TENSOR_FLAG_INT8_CONVERSION));
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmFullKernel(node) &&
          input_pos >= 20) {
        // Skip layer normalization weights. They are added in the Map
        // function (after all the other inputs added there) since layer
        // normalization weights are the last four inputs of the LSTM op in
        // Neuron.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmBasicKernel(node)) {
        // Configuring all inputs in the Map function
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinUnidirectionalSequenceLstm) {
        if (input_pos >= 20) {
          // Skip layer normalization weights. They are added in the Map
          // function (after all the other inputs added there) since layer
          // normalization weights are the last four inputs of the
          // unidirectional sequence LSTM op in Neuron.
          continue;
        }
        if (input_index == kTfLiteOptionalTensor) {
          TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
          continue;
        }
      }
      if ((reg->builtin_code == kTfLiteBuiltinSplit) &&
          (input_index == node->inputs->data[0])) {
        // Skip the axis input tensor; it will be added as a scalar operand
        // by the Map() mapping.
        continue;
      }
      if (reg->builtin_code == kTfLiteBuiltinTransposeConv ||
          (reg->builtin_code == kTfLiteBuiltinCustom &&
           reg->custom_name == "MTK_TRANSPOSE_CONV")) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      }

      // Pad and Padv2 have an optional parameter for a pad value which has
      // to be converted to a scalar type in Neuron.
      if ((reg->builtin_code == kTfLiteBuiltinPadv2 ||
           reg->builtin_code == kTfLiteBuiltinPad) &&
          node->inputs->size == 3 && input_pos == 2) {
        const int constant_value_id = node->inputs->data[2];
        if (constant_value_id == kTfLiteOptionalTensor) {
          continue;
        }
        const TfLiteTensor constant_value = context->tensors[constant_value_id];

        switch (constant_value.type) {
          case kTfLiteFloat32:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarFloat32Operand(*constant_value.data.f);
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_FLOAT32);
            }
            break;
          case kTfLiteUInt8:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*constant_value.data.uint8));
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_INT32);
            }
            break;
          case kTfLiteInt8:
            if (constant_value.allocation_type == kTfLiteMmapRo) {
              builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*constant_value.data.int8) + 128);
            } else {
              builder.AddSingleValueTensorAsScalarOperand(constant_value_id,
                                                          NEURON_INT32);
            }
            break;
          default:
            context->ReportError(context,
                                 "Unsupported type of pad value for pad_v2\n");
            return kTfLiteError;
        }
        continue;
      }

      if (input_index == kTfLiteOptionalTensor &&
          (reg->builtin_code == kTfLiteBuiltinLstm ||
           reg->builtin_code == kTfLiteBuiltinSvdf ||
           reg->builtin_code == kTfLiteBuiltinBidirectionalSequenceLstm)) {
        // properly handle the optional tensor for LSTM and SVDF.
        // currently only support float32.
        TF_LITE_ENSURE_STATUS(builder.AddVectorFloat32Operand(nullptr, 0));
      } else if (reg->builtin_code == kTfLiteBuiltinResizeBilinear ||
                 reg->builtin_code == kTfLiteBuiltinResizeNearestNeighbor) {
        if (input_pos == 0) {
          // Only the first input tensor is added. The second one,
          // specifying the output height and width, is not added and
          // instead the height and width will be added individually as
          // scalars by the mapping function returned by Map().
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        }
      } else if (reg->builtin_code == kTfLiteBuiltinTopkV2 && input_pos > 0) {
        // The K parameter tensor is not handled here but by the functor
        // returned by Map, the input tensor is instead added in
        // the else clause below
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinGather) {
        // Everything is added during Map since input tensors
        // have different order.
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinExpandDims &&
                 input_pos == 1) {
        // The axis param is added during Map
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinBatchToSpaceNd &&
                 input_pos == 2) {
        // Neuron does not support crops.
        // The Map function will check if all crops are zero.
        continue;
      } else if (reg->builtin_code == kTfLiteBuiltinArgMin ||
                 reg->builtin_code == kTfLiteBuiltinArgMax) {
        // The first input tensor is added as is. The second one, specifying
        // the axis, needs to be converted to a scalar since TFLite uses a
        // tensor but Neuron uses a scalar as the axis.
        if (input_pos == 0) {
          TF_LITE_ENSURE_STATUS(builder.AddTensorInput(input_index, hybrid_op));
        } else {
          const int axis_id = node->inputs->data[1];
          const TfLiteTensor& axis_tensor = context->tensors[axis_id];
          switch (axis_tensor.type) {
            case kTfLiteInt32:
              if (axis_tensor.allocation_type == kTfLiteMmapRo) {
                TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                    static_cast<int32_t>(*axis_tensor.data.i32)));
              } else {
                TF_LITE_ENSURE_STATUS(
                    builder.AddSingleValueTensorAsScalarOperand(axis_id,
                                                                NEURON_INT32));
              }
              break;
            case kTfLiteInt64:
              // Map() function already makes sure int64 input is constant.
              TF_LITE_ENSURE_STATUS(builder.AddScalarInt32Operand(
                  static_cast<int32_t>(*axis_tensor.data.i64)));
              break;
            default:
              return kTfLiteError;
          }
        }
      } else {
        if (is_asymm_to_symm) {
            TfLiteTensor* tensor = &context->tensors[input_index];
            if (tensor->type == kTfLiteInt8 && tensor->allocation_type == kTfLiteMmapRo) {
                input_tensor_flags &= ~NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
            }
        }
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorInput(input_index, hybrid_op, input_tensor_flags));
      }
    }

    // If we have target accelerators the target SDK version might be
    // different than the current android version.
    int android_sdk_version = neuronapi_->android_sdk_version;

    // Get op type and operands
    // Fails if the Validate function failed
    NeuronOperationType nn_op_type;
    TF_LITE_ENSURE_STATUS(Map(context, reg->builtin_code, reg->version,
                              reg->custom_name, android_sdk_version,
                              {context, &builder, node, &model_state_outputs_,
                               &model_state_tfl_inputs_, &feedback_loops_},
                              &nn_op_type));

    // Map outputs to Neuron tensor indices.
    int output_tensor_flags = 0;
    if (need_int8_conversion) {
      output_tensor_flags |= NN_TENSOR_FLAG_INT8_CONVERSION;
    }
    if (use_int8_asymm_signed) {
      output_tensor_flags |= NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED;
    }
    // fc_nn_intermediate_output_index is used to indicate whether additional
    // RESHAPE op is needed.
    int fc_nn_intermediate_output_index = -1;
    for (int output_pos = 0; output_pos < node->outputs->size; ++output_pos) {
      const auto output_index = node->outputs->data[output_pos];

      // Outputs for  basic LSTM cell are set in the Map function since
      if (reg->builtin_code == kTfLiteBuiltinLstm && isLstmBasicKernel(node)) {
        continue;
      }
      // Handle FC with keep_num_dims==true.
      if (reg->builtin_code == kTfLiteBuiltinFullyConnected &&
          reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data)
              ->keep_num_dims) {
        auto& output_tensor = context->tensors[output_index];

        int num_units = output_tensor.dims->data[output_tensor.dims->size - 1];
        std::vector<uint32_t> output_dims(2);
        output_dims[0] = NumElements(output_tensor.dims) / num_units;
        output_dims[1] = num_units;
        TF_LITE_ENSURE_STATUS(builder.AddIntermediateOutputTensor(
            output_tensor.type, output_dims.size(), output_dims.data(),
            output_tensor.params.scale, output_tensor.params.zero_point,
            &fc_nn_intermediate_output_index));
      } else {
        TF_LITE_ENSURE_STATUS(
            builder.AddTensorOutput(output_index, output_tensor_flags));
      }
    }

    // Dequantize operators may have to be added in case inputs are to be
    // floating-point.
    AddDequantizeOperatorsWhereNeeded(context, reg->builtin_code, node,
                                      &builder);
    //bool mtkCustomNode = reg->builtin_code == kTfLiteBuiltinCustom &&
    //                     mtk::IsMtkCustomNode(reinterpret_cast<void*>(node));
    //if (!mtkCustomNode) {
    //  builder.FinalizeAddOperation(nn_op_type);
    //} else {
    //  builder.FinalizeAddOperationExtension(node);
    //}
    TF_LITE_ENSURE_STATUS(builder.FinalizeAddOperation(nn_op_type));
    if (fc_nn_intermediate_output_index > -1) {
        TF_LITE_ENSURE_STATUS(builder.AppendReshape(
            fc_nn_intermediate_output_index, node->outputs->data[0]));
    }
  }
  if (options_.neuron_model) {
      *options_.current_neuron_index = operand_mapping_.get_neuron_current_index();
      *options_.neuron_model = nn_model_.release();
      return kTfLiteError;
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "AddOpsAndTensors done");
  return kTfLiteOk;
}

TfLiteStatus NeuronDelegateKernel::BuildGraph(
    TfLiteContext* context, const TfLiteIntArray* input_tensors,
    const TfLiteIntArray* output_tensors) {
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "BuildGraph");
  // Build the ops and tensors.
  TF_LITE_ENSURE_STATUS(AddOpsAndTensors(context));

  // Map input and output tensor indices to Neuron
  std::vector<uint32_t> inputs;
  inputs.reserve(input_tensors->size);
  std::vector<uint32_t> outputs;
  outputs.reserve(output_tensors->size);

  size_t total_input_byte_size = 0;
  // Make the TensorFlow Lite inputs and outputs to neuron indices.
  for (int i : TfLiteIntArrayView(input_tensors)) {
    // Constant tensors are not Neuron inputs.
    if (i != kTfLiteOptionalTensor &&
        context->tensors[i].allocation_type != kTfLiteMmapRo &&
        // The delegate might not have mapped this input (this can
        // happen if one tensor is split in several ones)
        operand_mapping_.lite_index_to_neuron(i) != -1) {
      inputs.push_back(operand_mapping_.lite_index_to_neuron(i));
      if (context->tensors[i].buffer_handle != kTfLiteNullBufferHandle) {
        continue;
      }
      const TfLiteType nn_type_conversion =
          operand_mapping_.lite_index_to_neuron_type_conversion(i);
      int tensor_size = 0;
      if (nn_type_conversion == kTfLiteNoType) {
        tensor_size = context->tensors[i].bytes;
      } else {
        size_t type_size;
        TF_LITE_ENSURE_OK(
            context, GetSizeOfType(context, nn_type_conversion, &type_size));
        tensor_size = NumElements(&context->tensors[i]) * type_size;
      }
      total_input_byte_size += tensor_size;
      total_input_byte_size += getNumPaddingBytes(tensor_size);
    }
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "total_input_byte_size: %d",
                  total_input_byte_size);

  size_t total_output_byte_size = 0;
  for (int i : TfLiteIntArrayView(output_tensors)) {
    const int output_tensor_neuron_index =
        operand_mapping_.lite_index_to_neuron(i);
    // Unmapped outputs are not added
    if (output_tensor_neuron_index != -1) {
      outputs.push_back(output_tensor_neuron_index);
    }
    if (context->tensors[i].buffer_handle != kTfLiteNullBufferHandle) {
      continue;
    }
    total_output_byte_size += context->tensors[i].bytes;
    total_output_byte_size += getNumPaddingBytes(context->tensors[i].bytes);
  }
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "total_output_byte_size: %d",
                  total_output_byte_size);

  // Add state output tensors as model outputs.
  for (int i : model_state_outputs_) {
    outputs.push_back(i);
  }
  // Tell Neuron to declare inputs/outputs
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronModel_identifyInputsAndOutputs(
          nn_model_.get(), inputs.size(), inputs.data(), outputs.size(),
          outputs.data()),
      "identifying model inputs and outputs");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "NeuronModel_identifyInputsAndOutputs");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "NeuronModel_identifyInputsAndOutputs allow fp16 %d", options_.allow_fp16);
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context,
      neuronapi_->NeuronModel_relaxComputationFloat32toFloat16(
          nn_model_.get(), options_.allow_fp16),
      "set relaxed computation mode for fp32 if possible");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                  "NeuronModel_relaxComputationFloat32toFloat16: %d",
                  options_.allow_fp16);
  RETURN_TFLITE_ERROR_IF_NEURON_ERROR(
      context, neuronapi_->NeuronModel_finish(nn_model_.get()),
      "finalizing the model");
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "NeuronModel_finish");
  // Create shared memory pool for inputs and outputs.
  nn_input_memory_.reset(
      new NNMemory(neuronapi_, "input_pool", total_input_byte_size, options_.use_ahwb, options_.use_ion));
  nn_output_memory_.reset(
      new NNMemory(neuronapi_, "output_pool", total_output_byte_size, options_.use_ahwb, options_.use_ion));

  TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "BuildGraph done");
  return kTfLiteOk;
}

}  // namespace neuron
}  // namespace tflite
