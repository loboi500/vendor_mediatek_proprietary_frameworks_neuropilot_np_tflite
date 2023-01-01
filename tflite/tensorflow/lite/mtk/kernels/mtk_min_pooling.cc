/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/mtk/kernels/internal/reference/mtk_reference_ops.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

#include "tensorflow/lite/mtk/kernels/mtk_ops.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"

#include "flatbuffers/flexbuffers.h"

namespace tflite {
namespace ops {
namespace mtk {
namespace pooling {

// This file has two implementation of each pooling op.
enum KernelType {
  kReference,
  kGenericOptimized,
};

enum PoolType {
  kMin,
};


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  auto parse_padding = [](Padding padding) {
    switch (padding) {
      case Padding_SAME:
        return kTfLitePaddingSame;
      case Padding_VALID:
        return kTfLitePaddingValid;
    }
    return kTfLitePaddingUnknown;
  };
  auto parse_activation = [](ActivationFunctionType activation) {
    switch (activation) {
      case ActivationFunctionType_NONE:
        return kTfLiteActNone;
      case ActivationFunctionType_RELU:
        return kTfLiteActRelu;
      case ActivationFunctionType_RELU_N1_TO_1:
        return kTfLiteActReluN1To1;
      case ActivationFunctionType_RELU6:
        return kTfLiteActRelu6;
      case ActivationFunctionType_TANH:
        return kTfLiteActTanh;
      case ActivationFunctionType_SIGN_BIT:
        return kTfLiteActSignBit;
    }
    return kTfLiteActNone;
  };

  OpData* data = new OpData;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->padding = parse_padding(static_cast<Padding>(m["PaddingType"].AsInt64()));
  data->stride_width = m["stride_width"].AsInt64();
  data->stride_height = m["stride_height"].AsInt64();
  data->kwidth = m["kwidth"].AsInt64();
  data->kheight = m["kheight"].AsInt64();
  data->activation = parse_activation(static_cast<ActivationFunctionType>(m["activation"].AsInt64()));

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <PoolType pool_type>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  const TfLiteType input_type = input->type;
  const TfLiteType output_type = output->type;

  if (input_type == kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, input->type, output->type);
  }
  else {
    TF_LITE_ENSURE(context, input_type == kTfLiteUInt8 ||
                            input_type == kTfLiteInt8 ||
                            input_type == kTfLiteInt16);
    TF_LITE_ENSURE(context, output_type == kTfLiteUInt8 ||
                            output_type == kTfLiteInt8 ||
                            output_type == kTfLiteInt16);
  }

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = data->padding;

  auto compute_out_size = [padding](int image_size, int filter_size,
                                    int stride) -> int {
    return padding == kTfLitePaddingSame
               ? (image_size + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (image_size - filter_size + stride) / stride
                     : 0;
  };

  int out_width =
      compute_out_size(width, data->kwidth, data->stride_width);
  int out_height =
      compute_out_size(height, data->kheight, data->stride_height);

  data->paddingValues.height = ComputePadding(data->stride_height, 1, height,
                                        data->kheight, out_height);
  data->paddingValues.width = ComputePadding(data->stride_width, 1, width,
                                       data->kwidth, out_width);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;

  return context->ResizeTensor(context, output, output_size);
}


template <KernelType kernel_type>
void MinEvalFloat(TfLiteContext* context, TfLiteNode* node, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(data->activation, &activation_min,
                           &activation_max);
#define TF_LITE_MIN_POOL(type)                                                    \
  MtkPoolParams op_params;                                                   \
  op_params.stride_height = data->stride_height;                                  \
  op_params.stride_width = data->stride_width;                                    \
  op_params.filter_height = data->kheight;                                        \
  op_params.filter_width = data->kwidth;                                          \
  op_params.padding_values.height = data->paddingValues.height;                   \
  op_params.padding_values.width = data->paddingValues.width;                     \
  op_params.float_activation_min = activation_min;                                \
  op_params.float_activation_max = activation_max;                                \
  reference_ops_mtk::MinPool(                                                     \
      op_params,                                                                  \
      GetTensorShape(input), GetTensorData<float>(input),                         \
      GetTensorShape(output), GetTensorData<float>(output))

  TF_LITE_MIN_POOL();

#undef TF_LITE_MIN_POOL
}

template <KernelType kernel_type>
void MinEvalQuantized(TfLiteContext* context, TfLiteNode* node, OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeQuantized(context, data->activation, output,
                                    &activation_min, &activation_max);

  const double input_scale = input->params.scale;
  const double output_scale = output->params.scale;
  const double real_output_multiplier = input_scale / output_scale;
  int32 output_multiplier;
  int output_shift;
  QuantizeMultiplier(real_output_multiplier, &output_multiplier, &output_shift);

  MtkPoolParams op_params;
  op_params.stride_height = data->stride_height;
  op_params.stride_width = data->stride_width;
  op_params.filter_height = data->kheight;
  op_params.filter_width = data->kwidth;
  op_params.padding_values.height = data->paddingValues.height;
  op_params.padding_values.width = data->paddingValues.width;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;

#define TF_LITE_MIN_POOL_REQUANT(input_type)                                \
  reference_ops_mtk::MinPoolRequantize<input_type>(                         \
      op_params,                                                            \
      GetTensorShape(input), GetTensorData<uint8_t>(input), input->type,    \
      GetTensorShape(output), GetTensorData<uint8_t>(output), output->type)

  if (input->type == kTfLiteUInt8) {
    TF_LITE_MIN_POOL_REQUANT(uint8_t);
  } else if (input->type == kTfLiteInt8) {
    TF_LITE_MIN_POOL_REQUANT(int8_t);
  } else if (input->type == kTfLiteInt16) {
    TF_LITE_MIN_POOL_REQUANT(int16_t);
  }

#undef TF_LITE_MAX_POOL_REQUANT
}

template <KernelType kernel_type>
TfLiteStatus MinEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      MinEvalFloat<kernel_type>(context, node, data, input, output);
      break;
    case kTfLiteInt16:
    case kTfLiteInt8:
    case kTfLiteUInt8:
      MinEvalQuantized<kernel_type>(context, node, data, input, output);
      break;
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

#define CHECK_NN(x)                                         \
  if (x != ANEURALNETWORKS_NO_ERROR) {                      \
    exit(1);                                                \
  }

int32_t add_ann_params(ANeuralNetworksModel* nn_model,
                       std::vector<uint32_t>& augmented_inputs,
                       uint32_t& next_id,
                       void* data) {
  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin =
      reinterpret_cast<ops::mtk::pooling::OpData*>(data);
  add_scalar_int32(builtin->padding);
  add_scalar_int32(builtin->stride_width);
  add_scalar_int32(builtin->stride_height);
  add_scalar_int32(builtin->kwidth);
  add_scalar_int32(builtin->kheight);
  add_scalar_int32(builtin->activation);
  return -1;
}

#define CHECK_NEURON(x)                                         \
  if (x != NEURON_NO_ERROR) {                      \
    exit(1);                                                \
  }

int32_t add_neuron_params(NeuronModel* nn_model,
                       std::vector<uint32_t>& augmented_inputs,
                       uint32_t& next_id,
                       void* data) {
  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    NeuronOperandType operand_type{.type = NEURON_INT32};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin =
      reinterpret_cast<ops::mtk::pooling::OpData*>(data);
  add_scalar_int32(builtin->padding);
  add_scalar_int32(builtin->stride_width);
  add_scalar_int32(builtin->stride_height);
  add_scalar_int32(builtin->kwidth);
  add_scalar_int32(builtin->kheight);
  add_scalar_int32(builtin->activation);
  return -1;
}

}  // namespace pooling

TfLiteRegistration* Register_MIN_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMin>,
                                 pooling::MinEval<pooling::kReference>};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTK_MIN_POOL_2D", pooling::add_ann_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTK_MIN_POOL_2D", pooling::add_neuron_params);
  return &r;
}

TfLiteRegistration* Register_MTK_MIN_POOL_2D() {
  return Register_MIN_POOL_REF();
}

}  // namespace mtk
}  // namespace ops
}  // namespace tflite

