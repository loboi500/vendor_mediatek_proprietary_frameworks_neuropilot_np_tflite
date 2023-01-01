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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/mtk/kernels/mtkext_ops.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "flatbuffers/flexbuffers.h"

#define LOG_TAG "MtkExtPooling3d"

namespace tflite {
namespace ops {
namespace mtkext {
namespace pooling_3d {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {

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
  data->padding = parse_padding(static_cast<Padding>(m["padding"].AsInt64()));
  data->stride_width = m["stride_w"].AsInt64();
  data->stride_height = m["stride_h"].AsInt64();
  data->stride_depth = m["stride_d"].AsInt64();
  data->filter_width = m["filter_width"].AsInt64();
  data->filter_height = m["filter_height"].AsInt64();
  data->filter_depth = m["filter_depth"].AsInt64();
  data->activation = parse_activation(static_cast<ActivationFunctionType>(
    m["fused_activation_function"].AsInt64()));
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Pool3DPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 5);
  int batches = SizeOfDimension(input, 0);
  int depth = SizeOfDimension(input, 1);
  int height = SizeOfDimension(input, 2);
  int width = SizeOfDimension(input, 3);
  int channels_out = SizeOfDimension(input, 4);

  auto padding = data->padding;
  int out_width = ComputeOutSize(padding, width, data->filter_width, data->stride_width, 1);
  int out_height = ComputeOutSize(padding, height, data->filter_height, data->stride_height, 1);
  int out_depth = ComputeOutSize(padding, depth, data->filter_depth, data->stride_depth, 1);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(5);
  output_size->data[0] = batches;
  output_size->data[1] = out_depth;
  output_size->data[2] = out_height;
  output_size->data[3] = out_width;
  output_size->data[4] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus AvgPool3DEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_AVERAGE_POOL_3D operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus MaxPool3DEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_MAX_POOL_3D operator is not implemented.\n");
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
  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    NeuronOperandType operand_type{.type = NEURON_INT32};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::pooling_3d::OpData*>(data);
  add_scalar_int32(builtin->padding);
  add_scalar_int32(builtin->stride_width);
  add_scalar_int32(builtin->stride_height);
  add_scalar_int32(builtin->stride_depth);
  add_scalar_int32(builtin->filter_width);
  add_scalar_int32(builtin->filter_height);
  add_scalar_int32(builtin->filter_depth);
  add_scalar_int32(builtin->activation);

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
  auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                           &next_id](int value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(int32_t)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::pooling_3d::OpData*>(data);
  add_scalar_int32(builtin->padding);
  add_scalar_int32(builtin->stride_width);
  add_scalar_int32(builtin->stride_height);
  add_scalar_int32(builtin->stride_depth);
  add_scalar_int32(builtin->filter_width);
  add_scalar_int32(builtin->filter_height);
  add_scalar_int32(builtin->filter_depth);
  add_scalar_int32(builtin->activation);

  return -1;
}

}  // namespace pooling_3d

TfLiteRegistration* Register_MTKEXT_AVERAGE_POOL_3D() {
  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::Pool3DPrepare,
                                 pooling_3d::AvgPool3DEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_AVERAGE_POOL_3D", pooling_3d::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_AVERAGE_POOL_3D", pooling_3d::add_ann_params);
  return &r;
}

TfLiteRegistration* Register_MTKEXT_MAX_POOL_3D() {
  static TfLiteRegistration r = {pooling_3d::Init, pooling_3d::Free,
                                 pooling_3d::Pool3DPrepare,
                                 pooling_3d::MaxPool3DEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_MAX_POOL_3D", pooling_3d::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_MAX_POOL_3D", pooling_3d::add_ann_params);
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
