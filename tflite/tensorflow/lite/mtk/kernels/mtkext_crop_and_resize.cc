/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 */
/* MediaTek Inc. (C) 2019. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/mtk/kernels/mtkext_ops.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "flatbuffers/flexbuffers.h"

#define LOG_TAG "MtkExtCropAndResize"

namespace tflite {
namespace ops {
namespace mtkext {
namespace crop_and_resize {

constexpr int kDataInputTensor = 0;
constexpr int kBoxTensor = 1;
constexpr int kBoxIndexTensor = 2;
constexpr int kOutputSizeTensor = 3;
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {

  auto parse_method = [](std::string method) {
    if (method.compare("bilinear") == 0) {
      return kCropAndResizeMethodBilinear;
    } else if (method.compare("nearest") == 0) {
      return kCropAndResizeMethodNearest;
    } else {
      return kCropAndResizeMethodUnknown;
    }
  };

  OpData* data = new OpData;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  data->extrapolation_value = m["extrapolation_value"].AsDouble();
  data->method = parse_method(m["method"].AsString().str());
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDataInputTensor, &input));
  const TfLiteTensor* box;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBoxTensor, &box));
  const TfLiteTensor* box_index;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kBoxIndexTensor, &box_index));
  const TfLiteTensor* output_size;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kOutputSizeTensor, &output_size));

  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(box), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(box, 1), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(box_index), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(box_index, 0), SizeOfDimension(box, 0));
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_size), 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(output_size, 0), 2);

  TF_LITE_ENSURE(context, output_size->type == kTfLiteInt32);
  TF_LITE_ENSURE(context, IsConstantTensor(output_size));

  const int32* output_size_data = GetTensorData<int32>(output_size);
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(4);
  output_size_array->data[0] = SizeOfDimension(box, 0);
  output_size_array->data[1] = output_size_data[0];
  output_size_array->data[2] = output_size_data[1];
  output_size_array->data[3] = SizeOfDimension(input, 3);
  return context->ResizeTensor(context, output, output_size_array);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_CROP_AND_RESIZE operator is not implemented.\n");
  return kTfLiteError;
}

#define CHECK_NEURON(x)                                                 \
    if (x != NEURON_NO_ERROR) {                                     \
        TFLITE_MTK_LOG_ERROR("Aborting since NN returned failure.");\
        exit(1);                                                    \
    }

int32_t add_neuron_params(NeuronModel* nn_model,
                                  std::vector<uint32_t>& augmented_inputs,
                                  uint32_t& next_id,
                                  void* data) {
  auto add_scalar_float32 = [&nn_model, &augmented_inputs,
                           &next_id](float value) {
    NeuronOperandType operand_type{.type = NEURON_FLOAT32};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(float)))
    augmented_inputs.push_back(next_id++);
  };

  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    NeuronOperandType operand_type{.type = NEURON_INT32};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::crop_and_resize::OpData*>(data);
  add_scalar_float32(builtin->extrapolation_value);
  add_scalar_int32(builtin->method);

  return -1;
}

#define CHECK_NN(x)                                                 \
    if (x != ANEURALNETWORKS_NO_ERROR) {                                     \
        TFLITE_MTK_LOG_ERROR("Aborting since NN returned failure.");\
        exit(1);                                                    \
    }

int32_t add_ann_params(ANeuralNetworksModel* nn_model,
                                  std::vector<uint32_t>& augmented_inputs,
                                  uint32_t& next_id,
                                  void* data) {
  auto add_scalar_float32 = [&nn_model, &augmented_inputs,
                           &next_id](float value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_FLOAT32};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(float)))
    augmented_inputs.push_back(next_id++);
  };

  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::crop_and_resize::OpData*>(data);
  add_scalar_float32(builtin->extrapolation_value);
  add_scalar_int32(builtin->method);

  return -1;
}

}  // namespace crop_and_resize

TfLiteRegistration* Register_MTKEXT_CROP_AND_RESIZE() {
  static TfLiteRegistration r = {crop_and_resize::Init,
                                 crop_and_resize::Free,
                                 crop_and_resize::Prepare,
                                 crop_and_resize::Eval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_CROP_AND_RESIZE", crop_and_resize::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_CROP_AND_RESIZE", crop_and_resize::add_ann_params);
  return &r;
}

}  // namespace mtkext
}  // namespace ops
}  // namespace tflite
