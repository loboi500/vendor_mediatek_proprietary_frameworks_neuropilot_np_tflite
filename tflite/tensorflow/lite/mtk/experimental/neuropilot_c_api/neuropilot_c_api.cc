/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly
 * prohibited.
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
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY
 * ACKNOWLEDGES THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY
 * THIRD PARTY ALL PROPER LICENSES CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK
 * SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK SOFTWARE RELEASES MADE TO
 * RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR STANDARD OR OPEN
 * FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER
 * WILL BE, AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT
 * ISSUE, OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER
 * TO MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek
 * Software") have been modified by MediaTek Inc. All revisions are subject to
 * any receiver's applicable license agreements with MediaTek Inc.
 */

#define LOG_TAG "NeuroPilotTFLite"

#include "tensorflow/lite/mtk/experimental/neuropilot_c_api/neuropilot_c_api.h"
#include <fstream>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_kernel.h"
//#include "tensorflow/lite/mtk/experimental/neuropilot_c_api/neuropilot_trace.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/mtk/kernels/mtk_register.h"
#include "tensorflow/lite/mtk/mtk_helper.h"
#include "tensorflow/lite/mtk/mtk_minimal_logging.h"
#include "tensorflow/lite/mtk/mtk_utils.h"

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>
#include <iostream>
#include <android/hardware_buffer.h>

#define DEBUG "DEBUG"
#define ERROR "ERROR"

class LOG {
public:
    std::stringstream input_str;
    std::string severity;

    LOG(std::string sv) { severity = sv; }

    ~LOG() {
        if (severity == DEBUG) {
            TFLITE_MTK_LOG_INFO("%s", input_str.str().c_str());
        } else if (severity == ERROR) {
            TFLITE_MTK_LOG_ERROR("%s", input_str.str().c_str());
        }
    }

    template <typename T>
    LOG& operator<<(T a) {
        input_str << a;
        return *this;
    }
};

#ifndef UNUSED
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)
#endif

#define CHECK_UNEXPECTED_NULL(expr)                                                        \
    do {                                                                                   \
        if (expr == nullptr) {                                                             \
            LOG(ERROR) << "Check failed: [" << #expr << " != nullptr] "; \
            return ANEURALNETWORKS_UNEXPECTED_NULL;                                        \
        }                                                                                  \
    } while (0)

#define CHECK_UNEXPECTED_SIZE(expr)                                                \
    do {                                                                           \
        if (expr == false) {                                                       \
            LOG(ERROR) << "Check failed: tensor size not match"; \
            return ANEURALNETWORKS_BAD_DATA;                                       \
        }                                                                          \
    } while (0)

static std::mutex tfliteBufferMutex;

typedef struct {
    bool allowFp16PrecisionForFp32 = true;
    bool execParallel = true;
    bool useNNAPI = true;
    std::unordered_map<int32_t, std::vector<int>> inputDimensions;
    const char* cache_dir = nullptr;
    int encryption_level = 0;
    int execution_preference = ExecutionPreference::kUndefined;
    const char* accelerator_name = nullptr;
    bool disallow_nnapi_cpu = false;
    int execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;
    uint64_t max_compilation_timeout_duration_ns = 0;
    uint64_t max_execution_timeout_duration_ns = 0;
    uint64_t max_execution_loop_timeout_duration_ns = 0;
    uint32_t max_number_delegated_partitions = 0;
    bool use_neuron_delegate = false;
    int32_t optimization_hint = kOptimizationDefault;
    const char* compile_options = nullptr;
    int8_t boost_value = -1;
    uint32_t warmup_runs = 0;
    uint32_t boost_duration;
    bool no_supported_operation_check;
    bool use_ahwb = true;
    bool use_ion;
    NeuronModel** neuron_model = nullptr;
    uint32_t* neuron_input_index = nullptr;
    uint32_t* neuron_output_index = nullptr;
    uint32_t* current_neuron_index = nullptr;

    // below are gpu_delegate options
    bool use_tflite_gpu_delegate = false;
    int gpu_inference_usage =
        TfLiteGpuInferenceUsage::TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    int gpu_inference_priority1 =
        TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    int gpu_inference_priority2 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    int gpu_inference_priority3 = TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
} NpTFLiteOptions;

typedef struct {
    void* tflite;           // The TFLite instance which holds the tensor
    TFLiteBufferType type;  // Input or output tensor
    uint32_t index;         // Zero-based index
} NpTFLiteTensor;

class NpTFLite {
public:
    explicit NpTFLite(const char* modelPath, NpTFLiteOptions* options) {
        if (options != nullptr) {
            options_ = *options;
        }
        model_ = tflite::FlatBufferModel::BuildFromFile(modelPath);
        resolver_.AddOtherCustom(model_->GetModel());
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
    }

    NpTFLite(const char* modelPath, const std::vector<TFLiteCustomOpExt>& customOperations,
             NpTFLiteOptions* options) {
        if (options != nullptr) {
            options_ = *options;
        }
        model_ = tflite::FlatBufferModel::BuildFromFile(modelPath);

        for (const auto& customOp : customOperations) {
            LOG(DEBUG) << "Custom OP name " << customOp.op_name;
            TfLiteRegistration reg = {
                .init = customOp.init,
                .free = customOp.free,
                .prepare = customOp.prepare,
                .invoke = nullptr,
                .profiling_string = nullptr,
                .builtin_code = 32,
                .custom_name = customOp.op_name,
            };
            resolver_.AddCustom(customOp.op_name, &reg);
        }

        resolver_.AddOtherCustom(model_->GetModel());
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);

        for (auto& customOp : customOperations) {
            ::tflite::mtk::CustomOpHelper::GetInstance().SetParamsFunc(
                customOp.op_name, customOp.target_name, customOp.vendor_name, customOp.add_params);
        }
    }

    NpTFLite(const char* buffer, size_t bufferSize, NpTFLiteOptions* options) {
        if (options != nullptr) {
            options_ = *options;
        }
        const char* ptr = CloneModelBuffer(buffer, bufferSize) ? modelBuffer_ : buffer;
        model_ = tflite::FlatBufferModel::BuildFromBuffer(ptr, bufferSize);
        resolver_.AddOtherCustom(model_->GetModel());
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
    }

    NpTFLite(const char* buffer, size_t bufferSize,
             const std::vector<TFLiteCustomOpExt>& customOperations, NpTFLiteOptions* options) {
        if (options != nullptr) {
            options_ = *options;
        }

        const char* ptr = CloneModelBuffer(buffer, bufferSize) ? modelBuffer_ : buffer;
        model_ = tflite::FlatBufferModel::BuildFromBuffer(ptr, bufferSize);

        for (const auto& customOp : customOperations) {
            LOG(DEBUG) << "Custom OP name " << customOp.op_name;
            TfLiteRegistration reg = {
                .init = customOp.init,
                .free = customOp.free,
                .prepare = customOp.prepare,
                .invoke = nullptr,
                .profiling_string = nullptr,
                .builtin_code = 32,
                .custom_name = customOp.op_name,
            };
            resolver_.AddCustom(customOp.op_name, &reg);
        }

        resolver_.AddOtherCustom(model_->GetModel());
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);

        for (auto& customOp : customOperations) {
            ::tflite::mtk::CustomOpHelper::GetInstance().SetParamsFunc(
                customOp.op_name, customOp.target_name, customOp.vendor_name, customOp.add_params);
        }
    }

    TfLiteStatus BuildGraph() {
        LOG(DEBUG) << "BuildGraph, allowFp16PrecisionForFp32: "
                                       << options_.allowFp16PrecisionForFp32
                                       << " execParallel: " << options_.execParallel
                                       << " useNNAPI: " << options_.useNNAPI;

        // Resize input tensor. Must be called before UseNNAPI()
        for (auto& x : options_.inputDimensions) {
            LOG(DEBUG) << "Resize input " << x.first;
            for (auto y = x.second.begin(); y != x.second.end(); ++y) {
                LOG(DEBUG) << *y << " ";
            }

            interpreter_->ResizeInputTensor(interpreter_->inputs()[x.first], x.second);
        }

        // SetAllowFp16PrecisionForFp32() must be called before UseNNAPI()
        interpreter_->SetAllowFp16PrecisionForFp32(options_.allowFp16PrecisionForFp32);

        if (options_.use_neuron_delegate) {
            NeuronDelegateOptions options;
            options.cache_dir = options_.cache_dir;
            if (options_.execution_preference != ExecutionPreference::kUndefined) {
                options.execution_preference = (ExecutionPreference) options_.execution_preference;
            }
            options.execution_priority = (ExecutionPriority) options_.execution_priority;
            options.accelerator_name = options_.accelerator_name;
            options.optimization_hint = options_.optimization_hint;
            options.allow_fp16 = options_.allowFp16PrecisionForFp32;
            options.boost_value = options_.boost_value;
            options.boost_duration = options_.boost_duration;
            options.use_ahwb = options_.use_ahwb;
            options.use_ion = options_.use_ion;
            options.neuron_input_index = options_.neuron_input_index;
            options.neuron_output_index = options_.neuron_output_index;
            options.neuron_model = options_.neuron_model;
            options.current_neuron_index = options_.current_neuron_index;
            options.compile_options = options_.compile_options;
            options.max_number_delegated_partitions = options_.max_number_delegated_partitions;

            delegate_ = TfLiteNeuronDelegateCreate(&options);
        } else if (options_.useNNAPI) {
            tflite::StatefulNnApiDelegate::Options options;
            options.cache_dir = options_.cache_dir;
            if (options_.execution_preference != ExecutionPreference::kUndefined) {
                options.execution_preference =
                    (tflite::StatefulNnApiDelegate::Options::ExecutionPreference)
                        options_.execution_preference;
            }
            options.accelerator_name = options_.accelerator_name;
            options.disallow_nnapi_cpu = options_.disallow_nnapi_cpu;
            options.execution_priority = options_.execution_priority;
            options.max_compilation_timeout_duration_ns =
                options_.max_compilation_timeout_duration_ns;
            options.max_execution_timeout_duration_ns = options_.max_execution_timeout_duration_ns;
            options.max_execution_loop_timeout_duration_ns =
                options_.max_execution_loop_timeout_duration_ns;
            options.max_number_delegated_partitions = options_.max_number_delegated_partitions;

            delegate_ = new tflite::StatefulNnApiDelegate(options);
        } else if (options_.use_tflite_gpu_delegate) {
            TfLiteGpuDelegateOptionsV2 gpuDelegateOptions = TfLiteGpuDelegateOptionsV2Default();
            gpuDelegateOptions.inference_preference = options_.gpu_inference_usage;
            gpuDelegateOptions.inference_priority1 = options_.gpu_inference_priority1;
            gpuDelegateOptions.inference_priority2 = options_.gpu_inference_priority2;
            gpuDelegateOptions.inference_priority3 = options_.gpu_inference_priority3;

            if (options_.allowFp16PrecisionForFp32) {
                gpuDelegateOptions.is_precision_loss_allowed = 1;
            } else {
                gpuDelegateOptions.is_precision_loss_allowed = 0;
            }

            if (options_.max_number_delegated_partitions > 0) {
                gpuDelegateOptions.max_delegated_partitions =
                    options_.max_number_delegated_partitions;
            }

            delegate_ = TfLiteGpuDelegateV2Create(&gpuDelegateOptions);
        }
        if (delegate_) {
            TF_LITE_ENSURE_STATUS(interpreter_->ModifyGraphWithDelegate(delegate_));
        }
        TF_LITE_ENSURE_STATUS(interpreter_->AllocateTensors());
        return kTfLiteOk;
    }

    TfLiteStatus RebuildInterpreter() {
        interpreter_.reset();
        resolver_.AddOtherCustom(model_->GetModel());
        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
        return BuildGraph();
    }

    std::vector<float> GetDequantizedOutput(int outputTensorIndex) {
        // Transfer the output tensor index to actual tensor index
        int index = interpreter_->outputs()[outputTensorIndex];
        return Dequantize<uint8_t>(ExtractVector<uint8_t>(index), GetTensorScale(index),
                                   GetTensorZeroPoint(index));
    }

    static size_t getNumPaddingBytes(size_t byte_size) {
      size_t num_padding_bytes = 0;
      if (byte_size % kDefaultByteAlignmentForNeuron) {
        num_padding_bytes = kDefaultByteAlignmentForNeuron -
                            (byte_size % kDefaultByteAlignmentForNeuron);
      }
      return num_padding_bytes;
    }

    int SetBufferHandle(void** memory_data, int index, bool cacheable, int buffer_size) {

        uint64_t usage = 0;

        if (cacheable) {
            usage=AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
        } else {
            usage=AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;
        }

        buffer_size += getNumPaddingBytes(buffer_size);

        AHardwareBuffer_Desc desc {
            .width=static_cast<uint32_t>(buffer_size),
            .height=1,
            .layers=1,
            .format=AHARDWAREBUFFER_FORMAT_BLOB,
            .usage=usage,
            .stride=static_cast<uint32_t>(buffer_size),
        };
        AHardwareBuffer* buffer = nullptr;

        if (options_.use_neuron_delegate){

            NeuronMemory* memory = nullptr;

            if(AHardwareBuffer_allocate(&desc, &buffer) == 0) {
                NeuronApiImplementation()->NeuronMemory_createFromAHardwareBuffer(buffer, &memory);
            }

            AHardwareBuffer_lock(buffer, usage, -1, NULL, reinterpret_cast<void**>(memory_data));
            buffer_vector_.push_back(buffer);
            neuron_memory_vector_.push_back(memory);

            auto handle = TfLiteNeuronDelegateRegisterNeuronMemory(delegate_, memory);
            return interpreter_->SetBufferHandle(index, handle, delegate_);
        } else if (options_.useNNAPI) {

            ANeuralNetworksMemory* memory = nullptr;

            if(AHardwareBuffer_allocate(&desc, &buffer) == 0) {
                NnApiImplementation()->
                  ANeuralNetworksMemory_createFromAHardwareBuffer(buffer, &memory);
            }

            AHardwareBuffer_lock(buffer, usage, -1, NULL, reinterpret_cast<void**>(memory_data));
            buffer_vector_.push_back(buffer);
            nnapi_memory_vector_.push_back(memory);

            static tflite::StatefulNnApiDelegate::CopyToHostTensorFnPtr memory_callback =
                [](TfLiteTensor* tensor, ANeuralNetworksMemory* memory,
                   size_t memory_offset, size_t byte_size,
                   void* callback_context) -> TfLiteStatus {
              return kTfLiteOk;
            };

            auto handle = reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate_)->
              RegisterNnapiMemory(memory, memory_callback, nullptr);
            return interpreter_->SetBufferHandle(index, handle, delegate_);
        }

        return kTfLiteError;
    }

    int SetAhwb(AHardwareBuffer* buffer, int index) {

        if (options_.use_neuron_delegate){

            NeuronMemory* memory = nullptr;

            NeuronApiImplementation()->NeuronMemory_createFromAHardwareBuffer(buffer, &memory);

            neuron_memory_vector_.push_back(memory);

            auto handle = TfLiteNeuronDelegateRegisterNeuronMemory(delegate_, memory);
            return interpreter_->SetBufferHandle(index, handle, delegate_);
        } else if (options_.useNNAPI) {

            ANeuralNetworksMemory* memory = nullptr;

            NnApiImplementation()->ANeuralNetworksMemory_createFromAHardwareBuffer(buffer, &memory);

            nnapi_memory_vector_.push_back(memory);

            static tflite::StatefulNnApiDelegate::CopyToHostTensorFnPtr memory_callback =
                [](TfLiteTensor* tensor, ANeuralNetworksMemory* memory,
                   size_t memory_offset, size_t byte_size,
                   void* callback_context) -> TfLiteStatus {
              return kTfLiteOk;
            };

            auto handle = reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate_)->
              RegisterNnapiMemory(memory, memory_callback, nullptr);
            return interpreter_->SetBufferHandle(index, handle, delegate_);
        }

        return kTfLiteError;
    }

    ~NpTFLite() {
        for (auto &i: buffer_vector_) {
            AHardwareBuffer_unlock(i, nullptr);
            AHardwareBuffer_release(i);
        }
        buffer_vector_.clear();
        for (auto &i:neuron_memory_vector_) {
            NeuronApiImplementation()->NeuronMemory_free(i);
        }
        neuron_memory_vector_.clear();
        for (auto &i:nnapi_memory_vector_) {
            NnApiImplementation()->ANeuralNetworksMemory_free(i);
        }
        nnapi_memory_vector_.clear();
        if (modelBuffer_ != nullptr) {
            free(modelBuffer_);
            modelBuffer_ = nullptr;
        }
        if (options_.use_neuron_delegate) {
            TfLiteNeuronDelegateDelete(delegate_);
        } else if (options_.useNNAPI) {
            if (delegate_ != nullptr) {
                delete delegate_;
                delegate_ = nullptr;
            }
        } else if (options_.use_tflite_gpu_delegate) {
            TfLiteGpuDelegateV2Delete(delegate_);
        }
    }

    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
    tflite::ops::builtin::MtkBuiltinOpResolver resolver_;
    NpTFLiteOptions options_;
    TfLiteDelegate* delegate_ = nullptr;
    std::vector<AHardwareBuffer*> buffer_vector_;
    std::vector<NeuronMemory*> neuron_memory_vector_;
    std::vector<ANeuralNetworksMemory*> nnapi_memory_vector_;
    constexpr static size_t kDefaultByteAlignmentForNeuron = 128;


private:
    bool CloneModelBuffer(const char* buffer, size_t bufferSize) {
        modelBuffer_ = reinterpret_cast<char*>(malloc(bufferSize));
        if (modelBuffer_ == nullptr) {
            return false;
        }
        memcpy(modelBuffer_, buffer, bufferSize);
        return true;
    }

    size_t GetTensorSize(int index) const {
        TfLiteTensor* t = interpreter_->tensor(index);

        if (t == nullptr) {
            return -1;
        }

        size_t total_size = 1;

        for (auto i = 0; i < t->dims->size; ++i) {
            total_size *= t->dims->data[i];
        }

        return total_size;
    }

    float GetTensorScale(int index) {
        TfLiteTensor* t = interpreter_->tensor(index);

        if (t != nullptr) {
            return t->params.scale;
        }

        LOG(ERROR) << "Fail to get tensor: %d" << index;
        return -1;
    }

    int32_t GetTensorZeroPoint(int index) {
        TfLiteTensor* t = interpreter_->tensor(index);

        if (t != nullptr) {
            return t->params.zero_point;
        }

        LOG(ERROR) << "Fail to get tensor: " << index;
        return -1;
    }

    // Return a vector with the flattened contents of a tensor.
    template <typename T>
    std::vector<T> ExtractVector(int index) {
        T* v = interpreter_->typed_tensor<T>(index);

        if (v == nullptr) {
            LOG(ERROR) << "Fail to extract vector from tensor: " << index;
            return std::vector<T>();
        }

        return std::vector<T>(v, v + GetTensorSize(index));
    }

    template <typename T>
    std::vector<float> Dequantize(const std::vector<T>& data, float scale, int32_t zero_point) {
        std::vector<float> f;

        for (T q : data) {
            f.push_back(scale * (q - zero_point));
        }

        return f;
    }

    char* modelBuffer_ = nullptr;
};

int ANeuroPilotTFLiteOptions_create(ANeuralNetworksTFLiteOptions** options) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = new NpTFLiteOptions();
    s->allowFp16PrecisionForFp32 = true;
    s->execParallel = true;
    s->useNNAPI = true;

    *options = reinterpret_cast<ANeuralNetworksTFLiteOptions*>(s);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setExecParallel(ANeuralNetworksTFLiteOptions* options,
                                             bool enableParallel) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->execParallel = enableParallel;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setAllowFp16PrecisionForFp32(ANeuralNetworksTFLiteOptions* options,
                                                          bool allow) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->allowFp16PrecisionForFp32 = allow;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setNoSupportedOperationCheck(ANeuralNetworksTFLiteOptions* options,
                                                          bool no_supported_operation_check) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->no_supported_operation_check = no_supported_operation_check;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setUseAhwb(ANeuralNetworksTFLiteOptions* options,
                                                          bool use_ahwb) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->use_ahwb = use_ahwb;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setUseIon(ANeuralNetworksTFLiteOptions* options,
                                                          bool use_ion) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->use_ion = use_ion;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setAccelerationMode(ANeuralNetworksTFLiteOptions* options,
                                                 AccelerationMode mode) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    if (mode == NP_ACCELERATION_NNAPI) {
        s->useNNAPI = true;
    } else if (mode == NP_ACCELERATION_NEURON) {
        s->use_neuron_delegate = true;
    } else if (mode == NP_ACCELERATION_CPU) {
        s->useNNAPI = false;
    } else if (mode == NP_ACCELERATION_GPU) {
        s->useNNAPI = false;
        s->use_neuron_delegate = false;
        s->use_tflite_gpu_delegate = true;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setCacheDir(ANeuralNetworksTFLiteOptions* options,
                                         const char* cache_dir) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->cache_dir = cache_dir;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setPreference(ANeuralNetworksTFLiteOptions* options,
                                           int execution_preference) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->execution_preference = execution_preference;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setDisallowNnApiCpu(ANeuralNetworksTFLiteOptions* options,
                                                 bool disallow_nnapi_cpu) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->disallow_nnapi_cpu = disallow_nnapi_cpu;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setAcceleratorName(ANeuralNetworksTFLiteOptions* options,
                                                const char* accelerator_name) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->accelerator_name = accelerator_name;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setExecutionPriority(ANeuralNetworksTFLiteOptions* options,
                                                  int execution_priority) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->execution_priority = execution_priority;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setMaxCompilationTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_compilation_timeout_duration_ns) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->max_compilation_timeout_duration_ns = max_compilation_timeout_duration_ns;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setMaxExecutionTimeout(ANeuralNetworksTFLiteOptions* options,
                                                    uint64_t max_execution_timeout_duration_ns) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->max_execution_timeout_duration_ns = max_execution_timeout_duration_ns;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setMaxExecutionLoopTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_loop_timeout_duration_ns) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->max_execution_loop_timeout_duration_ns = max_execution_loop_timeout_duration_ns;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setMaxNumberDelegatedPartitions(
    ANeuralNetworksTFLiteOptions* options, uint32_t max_number_delegated_partitions) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->max_number_delegated_partitions = max_number_delegated_partitions;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setEncryptionLevel(ANeuralNetworksTFLiteOptions* options,
                                                int encryption_level) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->encryption_level = encryption_level;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_resizeInputTensor(ANeuralNetworksTFLiteOptions* options, int32_t index,
                                               const int* dims, int32_t dimsSize) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    std::vector<int> d{dims, dims + dimsSize};
    LOG(DEBUG) << "TFLiteOptions resize input " << index;
    for (auto x = d.begin(); x != d.end(); ++x) {
        LOG(DEBUG) << *x << " ";
    }
    std::pair<int32_t, std::vector<int>> p = std::make_pair(index, d);
    s->inputDimensions.insert(p);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setLowLatency(ANeuralNetworksTFLiteOptions* options,
                                           bool enableLowLatency) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    if (enableLowLatency) {
        s->optimization_hint |= OptimizationHint::kOptimizationLowLatency;
    } else {
        s->optimization_hint &= ~OptimizationHint::kOptimizationLowLatency;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setDeepFusion(ANeuralNetworksTFLiteOptions* options,
                                           bool enableDeepFusion) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    if (enableDeepFusion) {
        s->optimization_hint |= OptimizationHint::kOptimizationDeepFusion;
    } else {
        s->optimization_hint &= ~OptimizationHint::kOptimizationDeepFusion;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setBatchProcessing(ANeuralNetworksTFLiteOptions* options,
                                                bool enableBatchProcessing) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    if (enableBatchProcessing) {
        s->optimization_hint |= OptimizationHint::kOptimizationBatchProcessor;
    } else {
        s->optimization_hint &= ~OptimizationHint::kOptimizationBatchProcessor;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setWarmupRuns(ANeuralNetworksTFLiteOptions* options,
                                           uint32_t warmupRuns) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->warmup_runs = warmupRuns;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setBoostHint(ANeuralNetworksTFLiteOptions* options,
                                          uint8_t boostValue) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->boost_value = static_cast<int8_t>(boostValue);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setBoostDuration(ANeuralNetworksTFLiteOptions* options,
                                              uint32_t duration) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->boost_duration = duration;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setCompileOptionByString(ANeuralNetworksTFLiteOptions* options,
                                              const char* compileOptions) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->compile_options = compileOptions;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setGpuExecutionPreference(ANeuralNetworksTFLiteOptions* options,
                                                       int execution_preference) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    s->gpu_inference_usage = execution_preference;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLiteOptions_setGpuExecutionPriority(ANeuralNetworksTFLiteOptions* options,
                                                        int priority_index, int priority_setting) {
    CHECK_UNEXPECTED_NULL(options);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    if (priority_index == 1) {
        s->gpu_inference_priority1 = priority_setting;
    } else if (priority_index == 2) {
        s->gpu_inference_priority3 = priority_setting;
    } else if (priority_index == 3) {
        s->gpu_inference_priority3 = priority_setting;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

void ANeuroPilotTFLiteOptions_free(ANeuralNetworksTFLiteOptions* options) {
    if (options == nullptr) {
        return;
    }

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    delete s;
    s = nullptr;
}

int ANeuroPilotTFLite_create(ANeuralNetworksTFLite** tflite, const char* modelPath) {
    return ANeuroPilotTFLite_createAdv(tflite, modelPath, nullptr);
}

int ANeuroPilotTFLite_createCustom(ANeuralNetworksTFLite** tflite, const char* modelPath,
                                   const std::vector<TFLiteCustomOpExt>& customOperations) {
    return ANeuroPilotTFLite_createAdvCustom(tflite, modelPath, customOperations, nullptr);
}

int ANeuroPilotTFLite_createAdv(ANeuralNetworksTFLite** tflite, const char* modelPath,
                                ANeuralNetworksTFLiteOptions* options) {

    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(modelPath);

    std::ifstream m(modelPath);

    if (!m.good()) {
        LOG(ERROR) << "Fail to read model file: " << modelPath;
        return ANEURALNETWORKS_BAD_DATA;
    }

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    NpTFLite* tf = new NpTFLite(modelPath, s);
    CHECK_UNEXPECTED_NULL(tf);

    if (tf->BuildGraph() != kTfLiteOk) {
        LOG(ERROR) << "Fail to build graph";
        *tflite = nullptr;
        delete tf;
        tf = nullptr;
        return ANEURALNETWORKS_INCOMPLETE;
    }

    if (s != nullptr) {
        for (uint32_t i = 0; i < s->warmup_runs; i++) {
            tf->interpreter_->Invoke();
        }
    }

    *tflite = reinterpret_cast<ANeuralNetworksTFLite*>(tf);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_createAdvCustom(ANeuralNetworksTFLite** tflite, const char* modelPath,
                                      const std::vector<TFLiteCustomOpExt>& customOperations,
                                      ANeuralNetworksTFLiteOptions* options) {

    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(modelPath);

    std::ifstream m(modelPath);

    if (!m.good()) {
        LOG(ERROR) << "Fail to read model file: " << modelPath;
        return ANEURALNETWORKS_BAD_DATA;
    }

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    NpTFLite* tf = new NpTFLite(modelPath, customOperations, s);
    CHECK_UNEXPECTED_NULL(tf);

    if (tf->BuildGraph() != kTfLiteOk) {
        LOG(ERROR) << "Fail to build graph";
        *tflite = nullptr;
        delete tf;
        tf = nullptr;
        return ANEURALNETWORKS_INCOMPLETE;
    }

    if (s != nullptr) {
        for (uint32_t i = 0; i < s->warmup_runs; i++) {
            tf->interpreter_->Invoke();
        }
    }

    *tflite = reinterpret_cast<ANeuralNetworksTFLite*>(tf);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_createWithBuffer(ANeuralNetworksTFLite** tflite, const char* buffer,
                                       size_t bufferSize) {
    return ANeuroPilotTFLite_createAdvWithBuffer(tflite, buffer, bufferSize, nullptr);
}

int ANeuroPilotTFLite_createNeuronModelWithBuffer(NeuronModel** neuron_model,
                                                  const char* buffer,
                                                  const size_t bufferSize,
                                                  uint32_t* neuron_input_index,
                                                  uint32_t* neuron_output_index,
                                                  uint32_t* current_neuron_index) {

    CHECK_UNEXPECTED_NULL(neuron_model);
    CHECK_UNEXPECTED_NULL(buffer);

    NpTFLiteOptions* s = new NpTFLiteOptions();
    s->neuron_input_index = neuron_input_index;
    s->neuron_output_index = neuron_output_index;
    s->neuron_model = neuron_model;
    s->current_neuron_index = current_neuron_index;
    s->use_neuron_delegate = true;
    NpTFLite* tf = new NpTFLite(buffer, bufferSize, s);
    CHECK_UNEXPECTED_NULL(tf);

    if (tf->BuildGraph() == kTfLiteOk) {
        LOG(ERROR) << "Fail to build graph";
        delete tf;
        tf = nullptr;
        return ANEURALNETWORKS_INCOMPLETE;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_createCustomWithBuffer(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations) {
    return ANeuroPilotTFLite_createAdvCustomWithBuffer(tflite, buffer, bufferSize, customOperations,
                                                       nullptr);
}

int ANeuroPilotTFLite_createAdvWithBuffer(ANeuralNetworksTFLite** tflite, const char* buffer,
                                          size_t bufferSize,
                                          ANeuralNetworksTFLiteOptions* options) {

    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(buffer);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    NpTFLite* tf = new NpTFLite(buffer, bufferSize, s);
    CHECK_UNEXPECTED_NULL(tf);

    if (tf->BuildGraph() != kTfLiteOk) {
        LOG(ERROR) << "Fail to build graph";
        *tflite = nullptr;
        delete tf;
        tf = nullptr;
        return ANEURALNETWORKS_INCOMPLETE;
    }

    if (s != nullptr) {
        for (uint32_t i = 0; i < s->warmup_runs; i++) {
            tf->interpreter_->Invoke();
        }
    }

    *tflite = reinterpret_cast<ANeuralNetworksTFLite*>(tf);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getTensorCount(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                     int32_t* count) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(count);
    int ret = ANEURALNETWORKS_NO_ERROR;
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        *count = static_cast<int32_t>(tf->interpreter_->inputs().size());
    } else if (btype == TFLITE_BUFFER_TYPE_OUTPUT) {
        *count = static_cast<int32_t>(tf->interpreter_->outputs().size());
    } else {
        *count = 0;
        ret = ANEURALNETWORKS_BAD_DATA;
    }

    return ret;
}

int ANeuroPilotTFLite_createAdvCustomWithBuffer(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    const std::vector<TFLiteCustomOpExt>& customOperations, ANeuralNetworksTFLiteOptions* options) {

    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(buffer);

    NpTFLiteOptions* s = reinterpret_cast<NpTFLiteOptions*>(options);
    NpTFLite* tf = new NpTFLite(buffer, bufferSize, customOperations, s);
    CHECK_UNEXPECTED_NULL(tf);

    if (tf->BuildGraph() != kTfLiteOk) {
        LOG(ERROR) << "Fail to build graph";
        *tflite = nullptr;
        delete tf;
        tf = nullptr;
        return ANEURALNETWORKS_INCOMPLETE;
    }

    if (s != nullptr) {
        for (uint32_t i = 0; i < s->warmup_runs; i++) {
            tf->interpreter_->Invoke();
        }
    }

    *tflite = reinterpret_cast<ANeuralNetworksTFLite*>(tf);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getTensorRank(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                    int index, int* rank) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(rank);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);
    *rank = tensor->dims->size;
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getTensorDimensions(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                          int index, int* dimensions) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(dimensions);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);
    for (auto i = 0; i < tensor->dims->size; i++) {
        *dimensions = tensor->dims->data[i];
        dimensions++;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getTensorByteSize(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                        int index, size_t* size) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(size);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);
    *size = tensor->bytes;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getTensorType(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                    int index, TFLiteTensorType* ttype) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(ttype);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);

    switch (tensor->type) {
        case kTfLiteFloat32:
            *ttype = TFLITE_TENSOR_TYPE_FLOAT;
            break;
        case kTfLiteUInt8:
            *ttype = TFLITE_TENSOR_TYPE_UINT8;
            break;
        case kTfLiteInt32:
            *ttype = TFLITE_TENSOR_TYPE_INT32;
            break;
        case kTfLiteInt64:
            *ttype = TFLITE_TENSOR_TYPE_INT64;
            break;
        case kTfLiteString:
            *ttype = TFLITE_TENSOR_TYPE_STRING;
            break;
        case kTfLiteBool:
            *ttype = TFLITE_TENSOR_TYPE_BOOL;
            break;
        case kTfLiteInt16:
            *ttype = TFLITE_TENSOR_TYPE_INT16;
            break;
        case kTfLiteComplex64:
            *ttype = TFLITE_TENSOR_TYPE_COMPLEX64;
            break;
        case kTfLiteInt8:
            *ttype = TFLITE_TENSOR_TYPE_INT8;
            break;
        case kTfLiteFloat16:
            *ttype = TFLITE_TENSOR_TYPE_FLOAT16;
            break;
        default:
            *ttype = TFLITE_TENSOR_TYPE_NONE;
            break;
    }
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_setTensorBuffer(ANeuralNetworksTFLite* tflite, int tensorIndex, char* data) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(data);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensorIndex);
    tensor->data.raw = data;

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_setInputTensorData(ANeuralNetworksTFLite* tflite, int index, const void* data,
                                         size_t size) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(data);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;
    tensor_index = tf->interpreter_->inputs()[index];
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);
    CHECK_UNEXPECTED_SIZE(size == tensor->bytes);
    memcpy(tensor->data.raw, data, size);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getOutputTensorData(ANeuralNetworksTFLite* tflite, int index, void* data,
                                          size_t size) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(data);
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int tensor_index = 0;
    tensor_index = tf->interpreter_->outputs()[index];
    TfLiteTensor* tensor = tf->interpreter_->tensor(tensor_index);
    CHECK_UNEXPECTED_SIZE(size == tensor->bytes);
    memcpy(data, tensor->data.raw, size);

    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_getDequantizedOutputByIndex(ANeuralNetworksTFLite* tflite, void* buffer,
                                                  size_t bufferByteSize, int tensorIndex) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(buffer);

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);

    if (float* data = tf->interpreter_->typed_tensor<float>(0)) {
        LOG(ERROR) << "Can't get dequantized output with float model";
        return ANEURALNETWORKS_BAD_DATA;
    }

    std::vector<float> v = tf->GetDequantizedOutput(tensorIndex);

    if (v.empty()) {
        LOG(ERROR) << "Empty dequantized data";
        return ANEURALNETWORKS_BAD_DATA;
    }

    if (bufferByteSize != v.size() * sizeof(float)) {
        LOG(ERROR) << "Invalid buffer size: "
                                       << bufferByteSize
                                       << " != " << (size_t)(v.size() * sizeof(float));
        return ANEURALNETWORKS_BAD_DATA;
    }

    std::copy(v.begin(), v.end(), reinterpret_cast<float*>(buffer));
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilotTFLite_invoke(ANeuralNetworksTFLite* tflite) {
    CHECK_UNEXPECTED_NULL(tflite);

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    return tf->interpreter_->Invoke();
}

void ANeuroPilotTFLite_free(ANeuralNetworksTFLite* tflite) {
    if (tflite == nullptr) {
        return;
    }

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    delete tf;
    tf = nullptr;
}

int ANeuroPilotTFLite_setBufferHandle(ANeuralNetworksTFLite* tflite,
        void** memory_data, TFLiteBufferType btype, int index, bool cacheable, int buffer_size) {

    CHECK_UNEXPECTED_NULL(tflite);

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);

    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }

    return tf->SetBufferHandle(memory_data, tensor_index, cacheable, buffer_size);
}

int ANeuroPilotTFLite_setAhwb(ANeuralNetworksTFLite* tflite,
        AHardwareBuffer* buffer, TFLiteBufferType btype, int index) {

    CHECK_UNEXPECTED_NULL(tflite);

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);

    int tensor_index = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        tensor_index = tf->interpreter_->inputs()[index];
    } else {
        tensor_index = tf->interpreter_->outputs()[index];
    }

    return tf->SetAhwb(buffer, tensor_index);
}

/**
 * Deprecated
 */
int ANeuroPilotTFLite_bindToDeivce(ANeuralNetworksTFLite* tflite, uint32_t device) {
    UNUSED(tflite);
    UNUSED(device);
    return ANEURALNETWORKS_NO_ERROR;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLite_setExecParallel(ANeuralNetworksTFLite* tflite, bool enableParallel) {
    UNUSED(tflite);
    UNUSED(enableParallel);
    return ANEURALNETWORKS_NO_ERROR;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLite_setAllowFp16PrecisionForFp32(ANeuralNetworksTFLite* tflite, bool allow) {
    CHECK_UNEXPECTED_NULL(tflite);

    TfLiteStatus status = kTfLiteOk;
    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);

    int input = tf->interpreter_->inputs()[0];
    TfLiteType input_data_type = tf->interpreter_->tensor(input)->type;

    if (input_data_type != kTfLiteFloat32) {
        LOG(ERROR) << "Can't set allow FP16 precision with a non-float model";
        return ANEURALNETWORKS_BAD_STATE;
    }

    if (tf->options_.allowFp16PrecisionForFp32 == allow) {
        return ANEURALNETWORKS_NO_ERROR;
    }

    LOG(DEBUG) << "Set allow FP16 precision for FP32: " << allow;
    tf->options_.allowFp16PrecisionForFp32 = allow;
    // Set false to reset NNAPIDelegate
    //tf->interpreter_->UseNNAPI(false);
    // Must be called before UseNNAPI()
    tf->interpreter_->SetAllowFp16PrecisionForFp32(tf->options_.allowFp16PrecisionForFp32);
    // Set true to recreate NNAPIDelegate
    //tf->interpreter_->UseNNAPI(tf->options_.useNNAPI);
    status = tf->interpreter_->Invoke();
    return (status == kTfLiteOk ? ANEURALNETWORKS_NO_ERROR : ANEURALNETWORKS_BAD_STATE);
}

/**
 * Deprecated
 */
int ANeuroPilotTFLiteCustomOp_getInput(TfLiteContext* context, TfLiteNode* node, int index,
                                       TFLiteTensorExt* tfliteTensor) {
    TfLiteTensor* tensor = &context->tensors[node->inputs->data[index]];
    CHECK_UNEXPECTED_NULL(tensor);

    for (int i = 0; i < tensor->dims->size; i++) {
        if (i >= TFLITE_TENSOR_MAX_DIMENSTIONS) break;

        tfliteTensor->dims[i] = tensor->dims->data[i];
    }

    tfliteTensor->dimsSize = tensor->dims->size;
    tfliteTensor->dimsSize = (tfliteTensor->dimsSize >= TFLITE_TENSOR_MAX_DIMENSTIONS)
                                 ? TFLITE_TENSOR_MAX_DIMENSTIONS
                                 : tfliteTensor->dimsSize;

    if (tensor->type == kTfLiteFloat32) {
        tfliteTensor->buffer = reinterpret_cast<void*>(tensor->data.raw);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_FLOAT;
        tfliteTensor->bufferSize = tensor->bytes;
    } else if (tensor->type == kTfLiteUInt8) {
        tfliteTensor->buffer = reinterpret_cast<void*>(tensor->data.raw);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_UINT8;
        tfliteTensor->bufferSize = tensor->bytes;
    } else {
        LOG(ERROR) << "Input or Output is not float nor uint8 data";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLiteCustomOp_getOutput(TfLiteContext* context, TfLiteNode* node, int index,
                                        TFLiteTensorExt* tfliteTensor) {
    TfLiteTensor* tensor = &context->tensors[node->outputs->data[index]];
    CHECK_UNEXPECTED_NULL(tensor);

    for (int i = 0; i < tensor->dims->size; i++) {
        if (i >= TFLITE_TENSOR_MAX_DIMENSTIONS) break;

        tfliteTensor->dims[i] = tensor->dims->data[i];
    }

    tfliteTensor->dimsSize = tensor->dims->size;
    tfliteTensor->dimsSize = (tfliteTensor->dimsSize >= TFLITE_TENSOR_MAX_DIMENSTIONS)
                                 ? TFLITE_TENSOR_MAX_DIMENSTIONS
                                 : tfliteTensor->dimsSize;

    if (tensor->type == kTfLiteFloat32) {
        tfliteTensor->buffer = reinterpret_cast<void*>(tensor->data.raw);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_FLOAT;
        tfliteTensor->bufferSize = tensor->bytes;
    } else if (tensor->type == kTfLiteUInt8) {
        tfliteTensor->buffer = reinterpret_cast<void*>(tensor->data.raw);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_UINT8;
        tfliteTensor->bufferSize = tensor->bytes;
    } else {
        LOG(ERROR) << "Input or Output is not float nor uint8 data";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    return ANEURALNETWORKS_NO_ERROR;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLiteCustomOp_resizeOutput(TfLiteContext* context, TfLiteNode* node, int index,
                                           TfLiteIntArray* new_size) {
    TfLiteTensor* tensor = &context->tensors[node->outputs->data[index]];
    return (kTfLiteOk == context->ResizeTensor(context, tensor,
                                               reinterpret_cast<TfLiteIntArray*>(new_size))
                ? ANEURALNETWORKS_NO_ERROR
                : ANEURALNETWORKS_BAD_DATA);
}

/**
 * Deprecated
 */
void* ANeuroPilotTFLiteCustomOp_getUserData(TfLiteNode* node) {
    return node->user_data;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLiteCustomOp_getFloatAttribute(const char* buffer, size_t length, const char* attr,
                                                float* outValue) {

    if (attr == nullptr || outValue == nullptr) {
        return ANEURALNETWORKS_BAD_DATA;
    }

    flexbuffers::Map m = flexbuffers::GetRoot((unsigned char*)buffer, length).AsMap();
    const auto& keys = m.Keys();

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto key = keys[i].AsKey();

        if (std::strcmp(key, attr) == 0) {
            const auto& value = m[key];

            if (value.GetType() == flexbuffers::FBT_FLOAT) {
                *outValue = value.AsFloat();
                return ANEURALNETWORKS_NO_ERROR;
            }
        }
    }

    return ANEURALNETWORKS_BAD_DATA;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLiteCustomOp_getIntAttribute(const char* buffer, size_t length, const char* attr,
                                              int32_t* outValue) {

    if (attr == nullptr || outValue == nullptr) {
        return ANEURALNETWORKS_BAD_DATA;
    }

    flexbuffers::Map m = flexbuffers::GetRoot((unsigned char*)buffer, length).AsMap();
    const auto& keys = m.Keys();

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto key = keys[i].AsKey();

        if (std::strcmp(key, attr) == 0) {
            const auto& value = m[key];

            if (value.GetType() == flexbuffers::FBT_INT) {
                *outValue = value.AsInt32();
                return ANEURALNETWORKS_NO_ERROR;
            }
        }
    }

    return ANEURALNETWORKS_BAD_DATA;
}

/**
 * Deprecated
 */
TfLiteIntArray* ANeuroPilotTFLite_createIntArray(int size) { return TfLiteIntArrayCreate(size); }

/**
 * Deprecated
 */
int ANeuroPilotTFLite_freeIntArray(TfLiteIntArray* v) {
    free(v);
    return ANEURALNETWORKS_NO_ERROR;
}

int ANeuroPilot_getInferencePreference(void) {
    int ret = NP_INFERENCE_TYPE_NONE;
    static int32_t version = tflite::mtk::GetAndroidSdkVersionCached();
    static bool mtk_nn_quant_preferred =
        tflite::mtk::PropertyGetBool("ro.vendor.mtk_nn_quant_preferred", false);
    if (version <= 27) {
        ret = NP_INFERENCE_TYPE_QNAUT;
    } else {
        ret = (mtk_nn_quant_preferred ? NP_INFERENCE_TYPE_QNAUT : NP_INFERENCE_TYPE_FLOAT);
    }
    return ret;
}

/**
 * Deprecated
 */
int ANeuroPilotTFLite_getTensor(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                TFLiteTensorExt* tfliteTensor) {
    return ANeuroPilotTFLite_getTensorByIndex(tflite, btype, tfliteTensor, 0);
}

/**
 * Deprecated
 */
int ANeuroPilotTFLite_getTensorByIndex(ANeuralNetworksTFLite* tflite, TFLiteBufferType btype,
                                       TFLiteTensorExt* tfliteTensor, int tensorIndex) {
    CHECK_UNEXPECTED_NULL(tflite);
    CHECK_UNEXPECTED_NULL(tfliteTensor);

    NpTFLite* tf = reinterpret_cast<NpTFLite*>(tflite);
    int index = 0;

    tfliteTensor->type = TFLITE_TENSOR_TYPE_NONE;
    tfliteTensor->dimsSize = 0;
    memset(&tfliteTensor->dims[0], 0, sizeof(int) * TFLITE_TENSOR_MAX_DIMENSTIONS);
    tfliteTensor->buffer = nullptr;
    tfliteTensor->bufferSize = 0;

    if (btype == TFLITE_BUFFER_TYPE_INPUT) {
        index = tf->interpreter_->inputs()[tensorIndex];
    } else if (btype == TFLITE_BUFFER_TYPE_OUTPUT) {
        index = tf->interpreter_->outputs()[tensorIndex];
    }

    TfLiteTensor* tensor = tf->interpreter_->tensor(index);

    for (int i = 0; i < tensor->dims->size; i++) {
        if (i >= TFLITE_TENSOR_MAX_DIMENSTIONS) break;

        tfliteTensor->dims[i] = tensor->dims->data[i];
    }

    tfliteTensor->dimsSize = tensor->dims->size;
    tfliteTensor->dimsSize = (tfliteTensor->dimsSize >= TFLITE_TENSOR_MAX_DIMENSTIONS)
                                 ? TFLITE_TENSOR_MAX_DIMENSTIONS
                                 : tfliteTensor->dimsSize;

    if (float* data = tf->interpreter_->typed_tensor<float>(index)) {
        tfliteTensor->buffer = reinterpret_cast<void*>(data);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_FLOAT;
        tfliteTensor->bufferSize = tensor->bytes;
    } else if (unsigned char* data = tf->interpreter_->typed_tensor<unsigned char>(index)) {
        tfliteTensor->buffer = reinterpret_cast<void*>(data);
        tfliteTensor->type = TFLITE_TENSOR_TYPE_UINT8;
        tfliteTensor->bufferSize = tensor->bytes;
    } else {
        LOG(ERROR) << "Input or Output is not float nor uint8 data";
        return ANEURALNETWORKS_UNEXPECTED_NULL;
    }

    return ANEURALNETWORKS_NO_ERROR;
}
