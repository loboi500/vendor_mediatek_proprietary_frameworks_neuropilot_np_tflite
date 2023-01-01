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

namespace tflite {
namespace ops {
namespace mtkext {
namespace unary {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus AbsEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_ABS operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus CosEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_COS operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_ELU operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus ExpEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_EXP operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_HARD_SWISH operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus LogEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_LOG operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus NegEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_NEG operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_RELU operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_RELU6 operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus RsqrtEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_RSQRT operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus SinEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_SIN operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus SqrtEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_SQRT operator is not implemented.\n");
  return kTfLiteError;
}

TfLiteStatus SquareEval(TfLiteContext* context, TfLiteNode* node) {
  context->ReportError(context, "MTKEXT_SQUARE operator is not implemented.\n");
  return kTfLiteError;
}

}  // namespace abs


TfLiteRegistration* Register_MTKEXT_ABS() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::AbsEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_COS() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::CosEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_ELU() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::EluEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_EXP() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::ExpEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_HARD_SWISH() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::HardSwishEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_LOG() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::LogEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_NEG() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::NegEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_RELU() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::ReluEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_RELU6() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::Relu6Eval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_RSQRT() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::RsqrtEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_SIN() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::SinEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_SQRT() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::SqrtEval};
  return &r;
}

TfLiteRegistration* Register_MTKEXT_SQUARE() {
  static TfLiteRegistration r = {nullptr, nullptr, unary::Prepare, unary::SquareEval};
  return &r;
}

}  // namespace mtkext
}  // namespace ops
}  // namespace tflite
