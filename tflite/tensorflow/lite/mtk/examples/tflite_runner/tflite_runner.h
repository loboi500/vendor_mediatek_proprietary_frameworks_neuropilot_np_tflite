/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein is
 * confidential and proprietary to MediaTek Inc. and/or its licensors. Without
 * the prior written permission of MediaTek inc. and/or its licensors, any
 * reproduction, modification, use or disclosure of MediaTek Software, and
 * information contained herein, in whole or in part, shall be strictly
 * prohibited.
 *
 * MediaTek Inc. (C) 2017. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER
 * ON AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL
 * WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR
 * NONINFRINGEMENT. NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH
 * RESPECT TO THE SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY,
 * INCORPORATED IN, OR SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES
 * TO LOOK ONLY TO SUCH THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO.
 * RECEIVER EXPRESSLY ACKNOWLEDGES THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO
 * OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES CONTAINED IN MEDIATEK
 * SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK SOFTWARE
 * RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S
 * ENTIRE AND CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE
 * RELEASED HEREUNDER WILL BE, AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE
 * MEDIATEK SOFTWARE AT ISSUE, OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE
 * CHARGE PAID BY RECEIVER TO MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek
 * Software") have been modified by MediaTek Inc. All revisions are subject to
 * any receiver's applicable license agreements with MediaTek Inc.
 */
#ifndef __TFLITE_RUNNER_H__
#define __TFLITE_RUNNER_H__

#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "tensorflow/lite/minimal_logging.h"
#include <sstream>
#include <string>

#define LOG_TAG "TfliteRunner"

#define INFO "INFO"
#define ERROR "ERROR"
#define TFLITE_INFO "TFLITE_INFO"

class LOG {
public:
  std::stringstream input_str;
  std::string severity;

  LOG(std::string sv) { severity = sv; }

  ~LOG() {
    if (severity == INFO) {
      TFLITE_MTK_LOG_INFO("%s", input_str.str().c_str());
    } else if (severity == ERROR) {
      TFLITE_MTK_LOG_ERROR("%s", input_str.str().c_str());
    } else if (severity == TFLITE_INFO) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "%s", input_str.str().c_str());
    }
  }

  template <typename T> LOG &operator<<(T a) {
    input_str << a;
    return *this;
  }
};

typedef struct TFLiteRunnerHandle TFLiteRunnerHandle;

int TFLiteRunner_Create(TFLiteRunnerHandle **handle,
                        const std::string &model_path);
int TFLiteRunner_Invoke(TFLiteRunnerHandle *handle,
                        const std::vector<std::string> &xs_list,
                        const std::vector<std::string> &ys_list);
void TFLiteRunner_Free(TFLiteRunnerHandle *handle);

void TFLiteRunner_UseNnApiDelegate(TFLiteRunnerHandle *handle);

void TFLiteRunner_UseNeuronDelegate(TFLiteRunnerHandle *handle);

void TFLiteRunner_SetAllowThreshold(TFLiteRunnerHandle *handle, bool allow);

void TFLiteRunner_SetCacheDir(TFLiteRunnerHandle *handle,
                              const char *cache_dir);

void TFLiteRunner_SetEncryptionLevel(TFLiteRunnerHandle *handle,
                                     int encrryption_level);

void TFLiteRunner_SetPreference(TFLiteRunnerHandle *handle,
                                int execution_preference);

void TFLiteRunner_SetDisallowNnApiCpu(TFLiteRunnerHandle* handle, bool disallow_nnapi_cpu);

void TFLiteRunner_SetAcceleratorName(TFLiteRunnerHandle* handle, const char* accelerator_name);

void TFLiteRunner_SetExecutionPriority(TFLiteRunnerHandle* handle,  int execution_priority);

void TFLiteRunner_SetMaxCompilationTimeout(TFLiteRunnerHandle* handle,
                                           uint64_t max_compilation_timeout_duration_ns);

void TFLiteRunner_SetMaxExecutionTimeout(TFLiteRunnerHandle* handle,
                                         uint64_t max_execution_timeout_duration_ns);

void TFLiteRunner_SetMaxExecutionLoopTimeout(TFLiteRunnerHandle* handle,
                                             uint64_t max_execution_loop_timeout_duration_ns);

void TFLiteRunner_SetAllowFp16Precision(TFLiteRunnerHandle *handle, bool allow);

void TFLiteRunner_SetLoopCount(TFLiteRunnerHandle *handle, int32_t count);

void TFLiteRunner_PrintState(TFLiteRunnerHandle *handle);

void TFLiteRunner_SetNoSupportedOperationCheck(TFLiteRunnerHandle* handle, bool no_supported_operation_check);

void TFLiteRunner_SetInterpreterNumThreads(TFLiteRunnerHandle *handle,
                                           int32_t num);

void TFLiteRunner_SetShowOutput(TFLiteRunnerHandle *handle, bool allow);

void TFLiteRunner_SetSaveOutput(TFLiteRunnerHandle *handle, bool allow);

void TFLiteRunner_SetLowLatency(TFLiteRunnerHandle *handle, bool enableLowLatency);

void TFLiteRunner_SetDeepFusion(TFLiteRunnerHandle *handle, bool enableDeepFusion);

void TFLiteRunner_SetBatchProcessing(TFLiteRunnerHandle *handle, bool enableBatchProcessing);

void TFLiteRunner_SetBoostDuration(TFLiteRunnerHandle* handle,
                                             uint32_t duration);

#endif // __TFLITE_RUNNER_H__
