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
#include "flatbuffers/flexbuffers.h"

namespace tflite {
namespace ops {
namespace mtkext {
namespace comparison {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteIntArray* output_size = nullptr;
  TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(context, input1, input2, &output_size));
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus EqualEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_EQUAL operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus GreaterEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_GREATER operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus GreaterEqualEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_GREATER_EQUAL operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus LessEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_LESS operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus LessEqualEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_LESS_EQUAL operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus NotEqualEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_NOT_EQUAL operator is not implemented.\n");
  return kTfLiteError;
}

}  // namespace div


TfLiteRegistration* Register_MTKEXT_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::EqualEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_GREATER() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::GreaterEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_GREATER_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::GreaterEqualEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_LESS() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::LessEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_LESS_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::LessEqualEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_NOT_EQUAL() {
  static TfLiteRegistration r = {nullptr, nullptr, comparison::Prepare, comparison::NotEqualEval};
  return &r;
}


}  // namespace mtkext
}  // namespace ops
}  // namespace tflite

