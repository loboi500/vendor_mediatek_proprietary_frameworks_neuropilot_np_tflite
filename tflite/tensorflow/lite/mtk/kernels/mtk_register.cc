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
#define LOG_TAG "MtkBuiltinOpResolver"
#include "tensorflow/lite/mtk/kernels/mtk_register.h"
#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "tensorflow/lite/util.h"
#include <iostream>

namespace tflite {
namespace ops {

namespace mtk {
TfLiteRegistration* Register_MTK_ABS();
TfLiteRegistration* Register_MTK_AXIS_ALIGNED_BBOX_TRANSFORM();
TfLiteRegistration* Register_MTK_BOX_WITH_NMS_LIMIT();
TfLiteRegistration* Register_MTK_CHANNEL_SHUFFLE();
TfLiteRegistration* Register_MTK_CROP_AND_RESIZE();
TfLiteRegistration* Register_MTK_DEPTH_TO_SPACE();
TfLiteRegistration* Register_MTK_ELU();
TfLiteRegistration* Register_MTK_GENERATE_PROPOSALS();
TfLiteRegistration* Register_MTK_LAYER_NORMALIZATION();
TfLiteRegistration* Register_MTK_MIN_POOL_2D();
TfLiteRegistration* Register_MTK_OPT();
TfLiteRegistration* Register_MTK_QUANTIZE_REF();
TfLiteRegistration* Register_MTK_REQUANTIZE();
TfLiteRegistration* Register_MTK_REVERSE();
TfLiteRegistration* Register_MTK_ROI_ALIGN();
TfLiteRegistration* Register_MTK_TRANSPOSE_CONV_REF();
TfLiteRegistration* Register_MTK_CUSTOM_OP();
}  // namespace mtk

// MTKEXT Operators
namespace mtkext {
TfLiteRegistration* Register_MTKEXT_ABS();
TfLiteRegistration* Register_MTKEXT_ARG_MAX();
TfLiteRegistration* Register_MTKEXT_ARG_MIN();
TfLiteRegistration* Register_MTKEXT_AVERAGE_POOL_3D();
TfLiteRegistration* Register_MTKEXT_CAST();
TfLiteRegistration* Register_MTKEXT_CONV_2D();
TfLiteRegistration* Register_MTKEXT_CONV_3D();
TfLiteRegistration* Register_MTKEXT_COS();
TfLiteRegistration* Register_MTKEXT_CROP_AND_RESIZE();
TfLiteRegistration* Register_MTKEXT_DEPTH_TO_SPACE();
TfLiteRegistration* Register_MTKEXT_DEPTHWISE_CONV_2D();
TfLiteRegistration* Register_MTKEXT_DEPTHWISE_CONV_3D();
TfLiteRegistration* Register_MTKEXT_DIV();
TfLiteRegistration* Register_MTKEXT_ELU();
TfLiteRegistration* Register_MTKEXT_EQUAL();
TfLiteRegistration* Register_MTKEXT_EXP();
TfLiteRegistration* Register_MTKEXT_FULLY_CONNECTED();
TfLiteRegistration* Register_MTKEXT_GATHER_ND();
TfLiteRegistration* Register_MTKEXT_GREATER();
TfLiteRegistration* Register_MTKEXT_GREATER_EQUAL();
TfLiteRegistration* Register_MTKEXT_HARD_SWISH();
TfLiteRegistration* Register_MTKEXT_L2_NORMALIZATION();
TfLiteRegistration* Register_MTKEXT_LESS();
TfLiteRegistration* Register_MTKEXT_LESS_EQUAL();
TfLiteRegistration* Register_MTKEXT_LOG();
TfLiteRegistration* Register_MTKEXT_MAX_POOL_3D();
TfLiteRegistration* Register_MTKEXT_NEG();
TfLiteRegistration* Register_MTKEXT_NOT_EQUAL();
TfLiteRegistration* Register_MTKEXT_POW();
TfLiteRegistration* Register_MTKEXT_PRELU();
TfLiteRegistration* Register_MTKEXT_REDUCE_MAX();
TfLiteRegistration* Register_MTKEXT_REDUCE_MIN();
TfLiteRegistration* Register_MTKEXT_RELU();
TfLiteRegistration* Register_MTKEXT_RELU6();
TfLiteRegistration* Register_MTKEXT_REQUANTIZE();
TfLiteRegistration* Register_MTKEXT_RESIZE_BILINEAR();
TfLiteRegistration* Register_MTKEXT_ROI_ALIGN();
TfLiteRegistration* Register_MTKEXT_RSQRT();
TfLiteRegistration* Register_MTKEXT_SIN();
TfLiteRegistration* Register_MTKEXT_SPACE_TO_DEPTH();
TfLiteRegistration* Register_MTKEXT_SQRT();
TfLiteRegistration* Register_MTKEXT_SQUARE();
TfLiteRegistration* Register_MTKEXT_SQUARED_DIFFERENCE();
TfLiteRegistration* Register_MTKEXT_SUM();
TfLiteRegistration* Register_MTKEXT_TILE();
TfLiteRegistration* Register_MTKEXT_TRANSPOSE_CONV_2D();
TfLiteRegistration* Register_MTKEXT_TRANSPOSE_CONV_3D();
}  // namespace mtkext

namespace builtin {
MtkBuiltinOpResolver::MtkBuiltinOpResolver() {
  BuiltinOpResolver::AddCustom("MTK_ABS",
                              mtk::Register_MTK_ABS());
  BuiltinOpResolver::AddCustom("MTK_AXIS_ALIGNED_BBOX_TRANSFORM",
                              mtk::Register_MTK_AXIS_ALIGNED_BBOX_TRANSFORM());
  BuiltinOpResolver::AddCustom("MTK_BOX_WITH_NMS_LIMIT",
                              mtk::Register_MTK_BOX_WITH_NMS_LIMIT());
  BuiltinOpResolver::AddCustom("MTK_CHANNEL_SHUFFLE",
                              mtk::Register_MTK_CHANNEL_SHUFFLE());
  BuiltinOpResolver::AddCustom("MTK_CROP_AND_RESIZE",
                              mtk::Register_MTK_CROP_AND_RESIZE());
  BuiltinOpResolver::AddCustom("MTK_DEPTH_TO_SPACE",
                              mtk::Register_MTK_DEPTH_TO_SPACE());
  BuiltinOpResolver::AddCustom("MTK_ELU",
                              mtk::Register_MTK_ELU());
  BuiltinOpResolver::AddCustom("MTK_GENERATE_PROPOSALS",
                              mtk::Register_MTK_GENERATE_PROPOSALS());
  BuiltinOpResolver::AddCustom("MTK_LAYER_NORMALIZATION",
                              mtk::Register_MTK_LAYER_NORMALIZATION());
  BuiltinOpResolver::AddCustom("MTK_MIN_POOL",
                              mtk::Register_MTK_MIN_POOL_2D());
  BuiltinOpResolver::AddCustom("MTK_OPT",
                              mtk::Register_MTK_OPT());
  BuiltinOpResolver::AddCustom("MTK_QUANTIZE",
                              mtk::Register_MTK_QUANTIZE_REF());
  BuiltinOpResolver::AddCustom("MTK_REQUANTIZE",
                              mtk::Register_MTK_REQUANTIZE());
  BuiltinOpResolver::AddCustom("MTK_REVERSE",
                              mtk::Register_MTK_REVERSE());
  BuiltinOpResolver::AddCustom("MTK_ROI_ALIGN",
                              mtk::Register_MTK_ROI_ALIGN());
  BuiltinOpResolver::AddCustom("MTK_TRANSPOSE_CONV",
                              mtk::Register_MTK_TRANSPOSE_CONV_REF());

  // MTKEXT Operators
  //
  BuiltinOpResolver::AddCustom("MTKEXT_ABS",
                               mtkext::Register_MTKEXT_ABS());
  BuiltinOpResolver::AddCustom("MTKEXT_ARG_MAX",
                               mtkext::Register_MTKEXT_ARG_MAX());
  BuiltinOpResolver::AddCustom("MTKEXT_ARG_MIN",
                               mtkext::Register_MTKEXT_ARG_MIN());
  BuiltinOpResolver::AddCustom("MTKEXT_AVERAGE_POOL_3D",
                               mtkext::Register_MTKEXT_AVERAGE_POOL_3D());
  BuiltinOpResolver::AddCustom("MTKEXT_CAST",
                               mtkext::Register_MTKEXT_CAST());
  BuiltinOpResolver::AddCustom("MTKEXT_CONV_2D",
                               mtkext::Register_MTKEXT_CONV_2D());
  BuiltinOpResolver::AddCustom("MTKEXT_CONV_3D",
                               mtkext::Register_MTKEXT_CONV_3D());
  BuiltinOpResolver::AddCustom("MTKEXT_COS",
                               mtkext::Register_MTKEXT_COS());
  BuiltinOpResolver::AddCustom("MTKEXT_CROP_AND_RESIZE",
                               mtkext::Register_MTKEXT_CROP_AND_RESIZE());
  BuiltinOpResolver::AddCustom("MTKEXT_DEPTH_TO_SPACE",
                               mtkext::Register_MTKEXT_DEPTH_TO_SPACE());
  BuiltinOpResolver::AddCustom("MTKEXT_DEPTHWISE_CONV_2D",
                               mtkext::Register_MTKEXT_DEPTHWISE_CONV_2D());
  BuiltinOpResolver::AddCustom("MTKEXT_DEPTHWISE_CONV_3D",
                               mtkext::Register_MTKEXT_DEPTHWISE_CONV_3D());
  BuiltinOpResolver::AddCustom("MTKEXT_DIV",
                               mtkext::Register_MTKEXT_DIV());
  BuiltinOpResolver::AddCustom("MTKEXT_ELU",
                               mtkext::Register_MTKEXT_ELU());
  BuiltinOpResolver::AddCustom("MTKEXT_EQUAL",
                               mtkext::Register_MTKEXT_EQUAL());
  BuiltinOpResolver::AddCustom("MTKEXT_EXP",
                               mtkext::Register_MTKEXT_EXP());
  BuiltinOpResolver::AddCustom("MTKEXT_FULLY_CONNECTED",
                               mtkext::Register_MTKEXT_FULLY_CONNECTED());
  BuiltinOpResolver::AddCustom("MTKEXT_GATHER_ND",
                               mtkext::Register_MTKEXT_GATHER_ND());
  BuiltinOpResolver::AddCustom("MTKEXT_GREATER",
                               mtkext::Register_MTKEXT_GREATER());
  BuiltinOpResolver::AddCustom("MTKEXT_GREATER_EQUAL",
                               mtkext::Register_MTKEXT_GREATER_EQUAL());
  BuiltinOpResolver::AddCustom("MTKEXT_HARD_SWISH",
                               mtkext::Register_MTKEXT_HARD_SWISH());
  BuiltinOpResolver::AddCustom("MTKEXT_L2_NORMALIZATION",
                               mtkext::Register_MTKEXT_L2_NORMALIZATION());
  BuiltinOpResolver::AddCustom("MTKEXT_LESS",
                               mtkext::Register_MTKEXT_LESS());
  BuiltinOpResolver::AddCustom("MTKEXT_LESS_EQUAL",
                               mtkext::Register_MTKEXT_LESS_EQUAL());
  BuiltinOpResolver::AddCustom("MTKEXT_LOG",
                               mtkext::Register_MTKEXT_LOG());
  BuiltinOpResolver::AddCustom("MTKEXT_MAX_POOL_3D",
                               mtkext::Register_MTKEXT_MAX_POOL_3D());
  BuiltinOpResolver::AddCustom("MTKEXT_NEG",
                               mtkext::Register_MTKEXT_NEG());
  BuiltinOpResolver::AddCustom("MTKEXT_NOT_EQUAL",
                               mtkext::Register_MTKEXT_NOT_EQUAL());
  BuiltinOpResolver::AddCustom("MTKEXT_POW",
                               mtkext::Register_MTKEXT_POW());
  BuiltinOpResolver::AddCustom("MTKEXT_PRELU",
                               mtkext::Register_MTKEXT_PRELU());
  BuiltinOpResolver::AddCustom("MTKEXT_REDUCE_MAX",
                               mtkext::Register_MTKEXT_REDUCE_MAX());
  BuiltinOpResolver::AddCustom("MTKEXT_REDUCE_MIN",
                               mtkext::Register_MTKEXT_REDUCE_MIN());
  BuiltinOpResolver::AddCustom("MTKEXT_RELU",
                               mtkext::Register_MTKEXT_RELU());
  BuiltinOpResolver::AddCustom("MTKEXT_RELU6",
                               mtkext::Register_MTKEXT_RELU6());
  BuiltinOpResolver::AddCustom("MTKEXT_REQUANTIZE",
                               mtkext::Register_MTKEXT_REQUANTIZE());
  BuiltinOpResolver::AddCustom("MTKEXT_RESIZE_BILINEAR",
                               mtkext::Register_MTKEXT_RESIZE_BILINEAR());
  BuiltinOpResolver::AddCustom("MTKEXT_ROI_ALIGN",
                               mtkext::Register_MTKEXT_ROI_ALIGN());
  BuiltinOpResolver::AddCustom("MTKEXT_RSQRT",
                               mtkext::Register_MTKEXT_RSQRT());
  BuiltinOpResolver::AddCustom("MTKEXT_SIN",
                               mtkext::Register_MTKEXT_SIN());
  BuiltinOpResolver::AddCustom("MTKEXT_SPACE_TO_DEPTH",
                               mtkext::Register_MTKEXT_SPACE_TO_DEPTH());
  BuiltinOpResolver::AddCustom("MTKEXT_SQRT",
                               mtkext::Register_MTKEXT_SQRT());
  BuiltinOpResolver::AddCustom("MTKEXT_SQUARE",
                               mtkext::Register_MTKEXT_SQUARE());
  BuiltinOpResolver::AddCustom("MTKEXT_SQUARED_DIFFERENCE",
                               mtkext::Register_MTKEXT_SQUARED_DIFFERENCE());
  BuiltinOpResolver::AddCustom("MTKEXT_SUM",
                               mtkext::Register_MTKEXT_SUM());
  BuiltinOpResolver::AddCustom("MTKEXT_TILE",
                               mtkext::Register_MTKEXT_TILE());
  BuiltinOpResolver::AddCustom("MTKEXT_TRANSPOSE_CONV_2D",
                               mtkext::Register_MTKEXT_TRANSPOSE_CONV_2D());
  BuiltinOpResolver::AddCustom("MTKEXT_TRANSPOSE_CONV_3D",
                               mtkext::Register_MTKEXT_TRANSPOSE_CONV_3D());
}

void MtkBuiltinOpResolver::AddOtherCustom(const ::tflite::Model* model) {
    const auto graph_ops = model->subgraphs()->Get(0)->operators();

    // Register unknown tflite custom ops before using tflite interpreter.
    for (uint32_t op_idx = 0; op_idx < graph_ops->size(); op_idx++) {
        const auto op = graph_ops->Get(op_idx);
        const auto opcode = model->operator_codes()->Get(op->opcode_index());

        const auto builtin_code = std::max(opcode->builtin_code(),
                static_cast<BuiltinOperator>(opcode->deprecated_builtin_code()));
        if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
            const char* name = opcode->custom_code()->c_str();

            if (FindOp(name, opcode->version()) == nullptr) {
                // Add unknown custom op into OpResolver.
                TfLiteRegistration* reg_op = tflite::ops::mtk::Register_MTK_CUSTOM_OP();
                reg_op->custom_name = strdup(name);
                AddCustom(name, reg_op);
            }
        }
    }
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
