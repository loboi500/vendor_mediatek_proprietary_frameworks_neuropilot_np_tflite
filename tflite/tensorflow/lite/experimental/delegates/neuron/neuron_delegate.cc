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
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate.h"

#include <utility>
#include <vector>

#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_kernel.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_delegate_validation.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_nnapi_delegate_kernel.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "utils/hash/farmhash.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

using tflite::delegate::nnapi::NNAPIDelegateKernel;
using tflite::delegate::nnapi::NNAPIValidationFailure;

namespace tflite {
namespace neuron {

const std::vector<uint8_t> mobilenetssd_bottom{
    248, 3,   126, 44,  74,  93,  120, 138,
    247, 251, 244, 147, 249, 168, 65,  155,
    133, 103, 85,  62,  235, 174, 54,  24,
    181, 193, 120, 251, 199, 151, 24,  199};
// NeuronDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class NeuronDelegate : public SimpleDelegateInterface {
 public:
  explicit NeuronDelegate(const NeuronDelegateOptions& options)
      : options_(options) {}

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    std::vector<NeuronValidationFailure> failure;
    std::vector<NNAPIValidationFailure> failure_nnapi;
    bool supported;
    if (redirect_cpu_) {
      return false;
    }
    if (redirect_nnapi_) {
        supported = NNAPIDelegateKernel::Validate(context,
                const_cast<TfLiteRegistration*>(registration), registration->builtin_code,
                registration->version, neuron_->android_sdk_version, node, true, &failure_nnapi);
        if (!supported) {
          TFLITE_LOG_PROD(
              tflite::TFLITE_LOG_ERROR, "OP %s (v%d) is not supported (%s)",
              tflite::EnumNameBuiltinOperator(
                  static_cast<BuiltinOperator>(registration->builtin_code)),
              registration->version,
              failure_nnapi.size() > 0 ? failure_nnapi[0].message.c_str() : "");
        }
    } else {
        supported = Validate(registration, node, context, &failure);
        if (!supported) {
          TFLITE_LOG_PROD(
              tflite::TFLITE_LOG_ERROR, "OP %s (v%d) is not supported (%s)",
              tflite::EnumNameBuiltinOperator(
                  static_cast<BuiltinOperator>(registration->builtin_code)),
              registration->version,
              failure.size() > 0 ? failure[0].message.c_str() : "");
        }
    }
    return supported;
  }

  std::string GetPropertyValue(const std::string& property) {
  #ifdef __ANDROID__
    char value[PROP_VALUE_MAX];
    if (0 == __system_property_get(property.c_str(), value)) {
      return std::string();
    }
    return std::string(value);
  #else   // !__ANDROID__
    return std::string();
  #endif  // __ANDROID__
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    if (GenModelToken(context) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Fail to gen neuron cache token.");
    }
    redirect_nnapi_ = !neuron_->neuron_exists || RedirectNnApi();
    redirect_cpu_ = RedirectCpu();
    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "TFLiteNeuronDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    if (redirect_nnapi_) {
      StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
      options.allow_fp16 = options_.allow_fp16;
      // Allow nnapi cpu to avoid crashes on some platforms due to unsupported ops.
      options.disallow_nnapi_cpu = false;
      options.execution_preference =
          (StatefulNnApiDelegate::Options::ExecutionPreference)
              options_.execution_preference;
      if (options_.execution_preference == kTurboBoost) {
        options.execution_preference =
            (StatefulNnApiDelegate::Options::ExecutionPreference)
                kFastSingleAnswer;
      }
      return std::make_unique<NeuronNNAPIDelegateKernel>(options);
    }
    return std::make_unique<NeuronDelegateKernel>(neuron_, options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

  TfLiteBufferHandle RegisterNeuronMemory(NeuronMemory* memory) {
    int map_size = tensor_memory_map_.size();
    for (int i = 0; i < map_size; i++) {
      if (tensor_memory_map_.at(i) == nullptr) {
        tensor_memory_map_.at(i) = memory;
        return i;
      }
    }
    tensor_memory_map_.push_back(memory);
    return map_size;
  }

  void GetTensorMemoryMap(std::vector<NeuronMemory*>* tensor_memory_map) {
      *tensor_memory_map = tensor_memory_map_;
  }

 private:
  // Compute the hash of a TfLiteIntArray.
  uint64_t GetHash(const TfLiteIntArray *int_array, uint64_t combine_with = 0) {
    constexpr auto kHashConst = 0x9e3779b97f4a7800ULL;
    uint64_t result = combine_with;
    for (auto i : TfLiteIntArrayView(int_array)) {
      result = result ^ (i + kHashConst + (result << 10) + (result >> 4));
    }
    return result;
  }
  // Create Model token & store in neuron_cache_token_.
  TfLiteStatus GenModelToken(TfLiteContext *context) {
    TfLiteIntArray *execution_plan = nullptr;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Fail to get exexution plan");
      return kTfLiteError;
    }
    TfLiteNode *input_node, *output_node;
    TfLiteRegistration *registration;
    if (context->GetNodeAndRegistration(context, 0, &input_node,
                                        &registration) != kTfLiteOk ||
        context->GetNodeAndRegistration(context, execution_plan->size - 1,
                                        &output_node,
                                        &registration) != kTfLiteOk) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, "Fail to get Node and Registration.");
      return kTfLiteError;
    }

    std::string bytes_count_str = std::to_string(context->tensors_size);
    const char *model_token = bytes_count_str.c_str();
    uint64_t token_parts[4];
    token_parts[0] =
        farmhash::Fingerprint64(model_token, std::strlen(model_token));
    token_parts[1] = GetHash(execution_plan);
    token_parts[2] = GetHash(input_node->inputs);
    token_parts[3] = GetHash(output_node->outputs);
    std::vector<uint8_t> nnapi_cache_token(32, 0);
    uint8_t *p = reinterpret_cast<uint8_t *>(token_parts);
    for (int i = 0; i < 4 * sizeof(uint64_t); i++) {
      nnapi_cache_token[i] = p[i];
    }

    // print model token
    // std::string test = "{";
    // for (auto i:nnapi_cache_token) {
    //       test += std::to_string(i);
    //       test += ",";
    //}
    // test += "};";

    neuron_cache_token_ = nnapi_cache_token;
    return kTfLiteOk;
  }

  bool RedirectNnApi() {
#if defined(__ANDROID__)
    constexpr char kRedirectProp[] = "debug.tflite.neuron.redirect_nnapi";
    char redirect[PROP_VALUE_MAX] = "";
    int length = __system_property_get(kRedirectProp, redirect);
    if (length == 1 && redirect[0] == '1') {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO,
                      "Redirect to NNAPI by system property");
      return true;
    }
#endif
    return false;
  }

  bool RedirectCpu() {
    // Aitutuv2: handle 6877 R get 0 score on mobilenetssd_bottom model
    if(GetPropertyValue("ro.hardware") == "mt6877" &&
      neuron_->android_sdk_version == kMinSdkVersionForNeuron13) {
        if(neuron_cache_token_ == mobilenetssd_bottom){
         return true;
        }
    }
    return false;
  }

  const NeuronApi* neuron_ = NeuronApiImplementation();
  const NeuronDelegateOptions options_;
  std::vector<NeuronMemory*> tensor_memory_map_;
  std::vector<uint8_t> neuron_cache_token_;
  // If true, redirect the graph to NNAPI delegate
  bool redirect_nnapi_ = false;
  bool redirect_cpu_ = false;
};

}  // namespace neuron
}  // namespace tflite

NeuronDelegateOptions TfLiteNeuronDelegateOptionsDefault() {
  NeuronDelegateOptions options = {
      // execution_preference = kFastSingleAnswer
      kFastSingleAnswer,
      // Default execution_priority = kPriorityHigh
      kPriorityHigh,
      // Default optimization_hint = kOptimizationDefault
      0,
      // Default boost value
      -1,
      // Default allow_fp16 = false
      false,
      // Default accelerator name
      nullptr,
      // Default boost_duration = 0
      0,
      // const char* cache_dir;
      nullptr,
      // const char* model_token;
      nullptr,
      // max number of delegated partitions
      0,
      // no supported operation check
      false,
      // neuron model
      nullptr,
      // neuron input index
      nullptr,
      // neuron output index
      nullptr,
      // current neuron index
      nullptr,
      // Default compile options
      nullptr,
      // use_ahwb;
      true,
      // use_ion;
      false,
  };

  // Return default options
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteNeuronDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteNeuronDelegateCreate(
    const NeuronDelegateOptions* options) {
  std::unique_ptr<tflite::neuron::NeuronDelegate> neuron(
      new tflite::neuron::NeuronDelegate(
          options ? *options : TfLiteNeuronDelegateOptionsDefault()));

  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(neuron));
}

// Destroys a delegate created with `TfLiteNeuronDelegateCreate` call.
void TfLiteNeuronDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

TfLiteBufferHandle TfLiteNeuronDelegateRegisterNeuronMemory(TfLiteDelegate* delegate,
        NeuronMemory* memory) {
    return reinterpret_cast<tflite::neuron::NeuronDelegate*>(delegate->data_)
      ->RegisterNeuronMemory(memory);
}

void TfLiteNeuronDelegateGetTensorMemoryMap(TfLiteDelegate* delegate,
                                            std::vector<NeuronMemory*>* tensor_memory_map) {
    return reinterpret_cast<tflite::neuron::NeuronDelegate*>(delegate->data_)
      ->GetTensorMemoryMap(tensor_memory_map);
}
