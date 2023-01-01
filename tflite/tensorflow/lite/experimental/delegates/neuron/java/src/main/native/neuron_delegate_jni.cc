/*
 * Copyright (C) 2021 MediaTek Inc., this file is modified on 02/26/2021
 * by MediaTek Inc. based on MIT License .
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the ""Software""), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <android/log.h>
#include <fcntl.h>
#include <jni.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>

#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate.h"

#define TAG "NeuronDelegateJni"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_com_mediatek_neuropilot_1S_neuron_NeuronDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jstring accelerator_name,
    jstring cache_dir, jstring model_token, jint max_delegated_partitions,
    jboolean allow_fp16, jint execution_priority,
    jboolean enable_low_latency, jboolean enable_deep_fusion, jboolean enable_batch_proccessing,
    jint boost_value, jint boost_duration, jstring compile_options, jboolean use_ahwb, jboolean use_ion) {
  auto options = TfLiteNeuronDelegateOptionsDefault();
  options.execution_preference = (ExecutionPreference)preference;
  if (accelerator_name) {
    options.accelerator_name = env->GetStringUTFChars(accelerator_name, NULL);
  }
  if (cache_dir) {
    options.cache_dir = env->GetStringUTFChars(cache_dir, NULL);
  }
  if (compile_options) {
      options.compile_options = env->GetStringUTFChars(compile_options, NULL);
  }
  if (model_token) {
    options.model_token = env->GetStringUTFChars(model_token, NULL);
  }

  if (max_delegated_partitions >= 0) {
    options.max_number_delegated_partitions = max_delegated_partitions;
  }

  if (allow_fp16) {
    options.allow_fp16 = allow_fp16;
  }

  options.execution_priority = (ExecutionPriority)execution_priority;

  if (enable_low_latency) {
      options.optimization_hint |= OptimizationHint::kOptimizationLowLatency;
  } else {
      options.optimization_hint &= ~OptimizationHint::kOptimizationLowLatency;
  }
  if (enable_deep_fusion) {
       options.optimization_hint |= OptimizationHint::kOptimizationDeepFusion;
  } else {
       options.optimization_hint &= ~OptimizationHint::kOptimizationDeepFusion;
  }
  if (enable_batch_proccessing) {
       options.optimization_hint |= OptimizationHint::kOptimizationBatchProcessor;
  } else {
       options.optimization_hint &= ~OptimizationHint::kOptimizationBatchProcessor;
  }
  options.boost_value = boost_value;
  options.boost_duration = boost_duration;

  options.use_ahwb = use_ahwb;
  options.use_ion = use_ion;

  return reinterpret_cast<jlong>(TfLiteNeuronDelegateCreate(&options));
}

JNIEXPORT void JNICALL Java_com_mediatek_neuropilot_1S_neuron_NeuronDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteNeuronDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
