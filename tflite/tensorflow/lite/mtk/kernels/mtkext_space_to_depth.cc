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

#define LOG_TAG "MtkExtSpaceToDepth"

namespace tflite {
namespace ops {
namespace mtkext {
namespace space_to_depth {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* data = new OpData;

  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();

  data->block_size = m["block_size"].AsInt64();
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);

  const int block_size = data->block_size;
  const int input_height = SizeOfDimension(input, 1);
  const int input_width = SizeOfDimension(input, 2);
  TF_LITE_ENSURE_EQ(context, input_height % block_size, 0);
  TF_LITE_ENSURE_EQ(context, input_width % block_size, 0);

  int output_height = input_height / block_size;
  int output_width = input_width / block_size;

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = SizeOfDimension(input, 0);
  output_size->data[1] = output_height;
  output_size->data[2] = output_width;
  output_size->data[3] = SizeOfDimension(input, 3) * (block_size * block_size);

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_SPACE_TO_DEPTH operator is not implemented.\n");
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

  auto builtin = reinterpret_cast<ops::mtkext::space_to_depth::OpData*>(data);
  add_scalar_int32(builtin->block_size);
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

  auto builtin = reinterpret_cast<ops::mtkext::space_to_depth::OpData*>(data);
  add_scalar_int32(builtin->block_size);
  return -1;
}


}  // namespace space_to_depth


TfLiteRegistration* Register_MTKEXT_SPACE_TO_DEPTH() {
  static TfLiteRegistration r = {space_to_depth::Init,
                                 space_to_depth::Free,
                                 space_to_depth::Prepare,
                                 space_to_depth::Eval};
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFuncNeuron(
      "MTKEXT_SPACE_TO_DEPTH", space_to_depth::add_neuron_params);
  ::tflite::mtk::CustomOpHelper::GetInstance().SetMtkExtOpParameterFunc(
      "MTKEXT_SPACE_TO_DEPTH", space_to_depth::add_ann_params);
  return &r;
}

}  // namespace mtkext
}  // namespace ops
}  // namespace tflite
