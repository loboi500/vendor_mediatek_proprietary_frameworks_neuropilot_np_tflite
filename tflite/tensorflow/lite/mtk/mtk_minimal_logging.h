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

#ifndef MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_
#define MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_

#include <cstdarg>
#include <string>
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/mtk/mtk_options.h"

namespace tflite {
namespace logging_internal {
namespace mtk {

inline void Log(LogSeverity severity, const char* tag,
    const char* format, ...) {
  static bool support = ::tflite::mtk::IsPrintLogSupported();
  if (!support && severity != TFLITE_LOG_ERROR) {
    return;
  }

  va_list args;
  va_start(args, format);
#ifdef __ANDROID__
  tflite::logging_internal::MinimalLogger::LogFormatted(severity, tag, format, args);
#else
  tflite::logging_internal::MinimalLogger::LogFormatted(severity, format, args);
#endif
  va_end(args);
}

}  // namespace mtk
}  // namespace logging_internal
}  // namespace tflite

#define TFLITE_MTK_LOG_INFO(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_INFO, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_WARN(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_WARNING, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_ERROR(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_ERROR, \
                                    LOG_TAG, format, ##__VA_ARGS__);

#define TFLITE_MTK_LOG_OP_DUMP(format, ...) \
  tflite::logging_internal::mtk::Log(tflite::TFLITE_LOG_INFO, \
                                    "TFLite OP Dump", format, ##__VA_ARGS__);

#endif  // MTK_TENSORFLOW_LITE_MINIMAL_LOGGING_H_
