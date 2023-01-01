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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/mtk/kernels/mtkext_ops.h"
#include "tensorflow/lite/mtk/mtk_utils.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "flatbuffers/flexbuffers.h"

#define LOG_TAG "MtkExtReduce"

namespace tflite {
namespace ops {
namespace mtkext {
namespace reduce {

constexpr int kInputTensor = 0;
constexpr int kAxesTensor = 1;
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* data = new OpData;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  data->keep_dims = m["keep_dims"].AsBool();
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* axes;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxesTensor, &axes));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TF_LITE_ENSURE_TYPES_EQ(context, axes->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, IsConstantTensor(axes));

  const int* axes_data = GetTensorData<int>(axes);
  int input_num_dims = NumDimensions(input);
  int num_axes = NumElements(axes);
  int num_output_dims;
  if (data->keep_dims) {
    num_output_dims = NumDimensions(input);
  } else {
    int num_reduce_axes = NumElements(axes);
    for (int i = 0; i < num_axes; ++i) {
      int current = axes_data[i];
      if (current < 0) {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j) {
        int previous = axes_data[j];
        if (previous < 0) {
          previous += input_num_dims;
        }
        if (current == previous) {
          num_reduce_axes --;
          break;
        }
      }
    }
    num_output_dims = NumDimensions(input) - num_reduce_axes;
  }

  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(num_output_dims);
  int num_skip_axes = 0;
  for (int idx = 0; idx < input_num_dims; ++idx) {
    bool is_match = false;
    for (int axis_idx = 0; axis_idx < num_axes; ++axis_idx) {
      if (axes_data[axis_idx] == idx || axes_data[axis_idx] + input_num_dims == idx) {
        is_match = true;
        break;
      }
    }
    if (is_match) {
      if (data->keep_dims) {
        output_dims->data[idx - num_skip_axes] = 1;
      } else {
        num_skip_axes ++;
      }
    } else {
      output_dims->data[idx - num_skip_axes] = SizeOfDimension(input, idx);
    }
  }
  return context->ResizeTensor(context, output, output_dims);
}

TfLiteStatus ReduceMinEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_REDUCE_MIN operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus ReduceMaxEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_REDUCE_MAX operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus SumEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_SUM operator is not implemented.\n");
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
  auto add_scalar_bool = [&nn_model, &augmented_inputs,
                           &next_id](bool value) {
    NeuronOperandType operand_type{.type = NEURON_BOOL};
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_addOperand(nn_model, &operand_type))
    CHECK_NEURON(NeuronApiImplementation()->NeuronModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(bool)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::reduce::OpData*>(data);
  add_scalar_bool(builtin->keep_dims);

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
  auto add_scalar_bool = [&nn_model, &augmented_inputs,
                           &next_id](bool value) {
    ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_BOOL};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
    CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                  sizeof(bool)))
    augmented_inputs.push_back(next_id++);
  };

  auto builtin = reinterpret_cast<ops::mtkext::reduce::OpData*>(data);
  add_scalar_bool(builtin->keep_dims);

  return -1;
}

}  // namespace reduce


TfLiteRegistration* Register_MTKEXT_REDUCE_MIN() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::Prepare, reduce::ReduceMinEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_REDUCE_MIN", reduce::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_REDUCE_MIN", reduce::add_ann_params);
  return &r;
}

TfLiteRegistration* Register_MTKEXT_REDUCE_MAX() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::Prepare, reduce::ReduceMaxEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_REDUCE_MAX", reduce::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_REDUCE_MAX", reduce::add_ann_params);
  return &r;
}

TfLiteRegistration* Register_MTKEXT_SUM() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::Prepare, reduce::SumEval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_SUM", reduce::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_SUM", reduce::add_ann_params);
  return &r;
}

}  // namespace mtkext
}  // namespace ops
}  // namespace tflite
