/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/mtk/kernels/internal/reference/mtk_reference_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/mtk/kernels/mtk_ops.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"

namespace tflite {
namespace ops {
namespace mtk {
namespace roi_align {

constexpr int kDataInputTensor = 0;
constexpr int kBoxTensor = 1;
constexpr int kBoxIndexTensor = 2;
constexpr int kOutputSizeTensor = 3;
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* data = new OpData;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->height_ratio = m["height_ratio"].AsDouble();
  data->width_ratio = m["width_ratio"].AsDouble();
  data->sampling_ratio_height = m["sampling_ratio_height"].AsInt64();
  data->sampling_ratio_width = m["sampling_ratio_width"].AsInt64();
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus RoiAlignPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // check node input/output
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  const TfLiteTensor* box = GetInput(context, node, kBoxTensor);
  const TfLiteTensor* box_index = GetInput(context, node, kBoxIndexTensor);
  const TfLiteTensor* output_size = GetInput(context, node, kOutputSizeTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // check input dimension
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(box), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(box, 1), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(box_index), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(box_index, 0), SizeOfDimension(box, 0));
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_size), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(output_size, 0), 2);

  // update output tensor size
  TF_LITE_ENSURE(context, IsConstantTensor(output_size));
  const int32* output_size_data = GetTensorData<int32>(output_size);
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(4);
  output_size_array->data[0] = SizeOfDimension(box, 0);
  output_size_array->data[1] = output_size_data[0];
  output_size_array->data[2] = output_size_data[1];
  output_size_array->data[3] = SizeOfDimension(input, 3);
  return context->ResizeTensor(context, output, output_size_array);
}

TfLiteStatus RoiAlignEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTK_ROI_ALIGN operator is not implemented.");
  return kTfLiteError;
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

  auto add_scalar_float32 = [&nn_model, &augmented_inputs,
                             &next_id](int value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_FLOAT32};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(float)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtk::roi_align::OpData*>(data);
#if 0
  add_scalar_float32(builtin->extrapolation_value);
  add_scalar_int32(builtin->kheight);
  add_scalar_int32(builtin->kwidth);
#endif
  return ::tflite::mtk::Hash("roialignmtk");
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

  auto add_scalar_float32 = [&nn_model, &augmented_inputs,
                             &next_id](int value) {
    NeuronOperandType operand_type{.type = NEURON_FLOAT32};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(float)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtk::roi_align::OpData*>(data);
#if 0
  add_scalar_float32(builtin->extrapolation_value);
  add_scalar_int32(builtin->kheight);
  add_scalar_int32(builtin->kwidth);
#endif
  return ::tflite::mtk::Hash("roialignmtk");
}

}  // namespace roi_align

TfLiteRegistration* Register_MTK_ROI_ALIGN_REF() {
  static TfLiteRegistration r = {roi_align::Init,
                                 roi_align::Free,
                                 roi_align::RoiAlignPrepare,
                                 roi_align::RoiAlignEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTK_ROI_ALIGN", roi_align::add_ann_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTK_ROI_ALIGN", roi_align::add_neuron_params);
  return &r;
}

TfLiteRegistration* Register_MTK_ROI_ALIGN() {
    return Register_MTK_ROI_ALIGN_REF();
}

}  // namespace mtk
}  // namespace ops
}  // namespace tflite
