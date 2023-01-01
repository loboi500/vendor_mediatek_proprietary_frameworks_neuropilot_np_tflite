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

#ifndef TENSORFLOW_CONTRIB_LITE_MTK_KERNELS_MTKEXT_OPS_H_
#define TENSORFLOW_CONTRIB_LITE_MTK_KERNELS_MTKEXT_OPS_H_

namespace tflite {
namespace ops {
namespace mtkext {

namespace crop_and_resize {

typedef enum {
  kCropAndResizeMethodUnknown = 0,
  kCropAndResizeMethodBilinear = 1,
  kCropAndResizeMethodNearest = 2,
} CropAndResizeMethod;

struct OpData {
  // For MTKEXT_CROP_AND_RESIZE version 1.
  float extrapolation_value;
  CropAndResizeMethod method;
};

}  // namespace crop_and_resize

namespace conv_2d {

struct OpData {
  // For MTKEXT_CONV_2D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
};

}  // namespace conv_2d

namespace conv_3d {

struct OpData {
  // For MTKEXT_CONV_3D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int stride_depth;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
  int dilation_depth_factor;
};

}  // namespace conv_3d

namespace depth_to_space {

struct OpData {
  // For MTKEXT_DEPTH_TO_SPACE version 1.
  int64_t block_size;
};

}  // namespace depth_to_space

namespace depthwise_conv_2d {

struct OpData {
  // For MTKEXT_DEPTHWISE_CONV_2D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
};

}  // namespace depthwise_conv_2d

namespace depthwise_conv_3d {

struct OpData {
  // For MTKEXT_DEPTHWISE_CONV_3D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int stride_depth;
  int depth_multiplier;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
  int dilation_depth_factor;
};

}  // namespace depthwise_conv_3d

namespace div {

struct OpData {
  // For MTKEXT_DIV version 1.
  TfLiteFusedActivation activation;
};

}  // namespace div

namespace fully_connected {

struct OpData {
  // For MTKEXT_FULLY_CONNECTED version 1.
  TfLiteFusedActivation activation;
  bool keep_num_dims;
};

}  // namespace fully_connected

namespace pooling_3d {

struct OpData {
  // For MTKEXT_AVERAGE_POOL_3D and MTKEXT_MAX_POOL_3D version 1.
  TfLitePadding padding;
  int filter_width;
  int filter_height;
  int filter_depth;
  int stride_width;
  int stride_height;
  int stride_depth;
  TfLiteFusedActivation activation;
};

}  // namespace pooling_3d

namespace reduce {

struct OpData {
  // For MTKEXT_REDUCE_MIN and MTKEXT_REDUCE_MAX version 1.
  bool keep_dims;
};

}  // namespace reduce

namespace resize_bilinear {

struct OpData {
  // For MTKEXT_RESIZE_BILINEAR version 1.
  bool align_corners;
  bool half_pixel_centers;
};

}  // namespace resize_bilinear

namespace roi_align {

struct OpData {
  // For MTKEXT_ROI_ALIGN version 1.
  int32_t sampling_ratio_height;
  int32_t sampling_ratio_width;
  float height_ratio;
  float width_ratio;
};

}  // namespace roi_align

namespace space_to_depth {

struct OpData {
  // For MTKEXT_SPACE_TO_DEPTH version 1.
  int64_t block_size;
};

}  // namespace space_to_depth

namespace transpose_conv_2d {

struct OpData {
  // For MTKEXT_TRANSPOSE_CONV_2D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
};

}  // namespace transpose_conv_2d

namespace transpose_conv_3d {

struct OpData {
  // For MTKEXT_TRANSPOSE_CONV_3D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int stride_depth;
  TfLiteFusedActivation activation;
  int dilation_width_factor;
  int dilation_height_factor;
  int dilation_depth_factor;
};

}  // namespace transpose_conv_3d

}  // namespace mtkext
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_MTK_KERNELS_MTKEXT_OPS_H_
