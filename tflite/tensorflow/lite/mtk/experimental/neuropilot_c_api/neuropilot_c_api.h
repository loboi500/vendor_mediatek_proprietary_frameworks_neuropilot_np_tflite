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

#ifndef __MTK_NEUROPILOT_C_API_H__
#define __MTK_NEUROPILOT_C_API_H__

#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"
#include <vector>
#include "tensorflow/lite/interpreter.h"

#define TFLITE_TENSOR_MAX_DIMENSTIONS 4

/************************************************************************************************/
// Must be synchronized with NeuroPilotTFLiteShim.h
typedef struct ANeuralNetworksTFLite ANeuralNetworksTFLite;
typedef struct NeuronModel NeuronModel;
typedef struct ANeuralNetworksTFLiteOptions ANeuralNetworksTFLiteOptions;
typedef struct ANeuralNetworksTFLiteTensor ANeuralNetworksTFLiteTensor;
typedef struct TfLiteContext TfLiteContext;

typedef enum {
  TFLITE_BUFFER_TYPE_INPUT = 0,
  TFLITE_BUFFER_TYPE_OUTPUT = 1,
} NpTFLiteBufferType;

typedef uint32_t TFLiteBufferType;

typedef enum {
  TFLITE_TENSOR_TYPE_NONE = 0,
  TFLITE_TENSOR_TYPE_FLOAT = 1,
  TFLITE_TENSOR_TYPE_UINT8 = 2,
  TFLITE_TENSOR_TYPE_INT32 = 3,
  TFLITE_TENSOR_TYPE_INT64 = 4,
  TFLITE_TENSOR_TYPE_STRING = 5,
  TFLITE_TENSOR_TYPE_BOOL = 6,
  TFLITE_TENSOR_TYPE_INT16 = 7,
  TFLITE_TENSOR_TYPE_COMPLEX64 = 8,
  TFLITE_TENSOR_TYPE_INT8 = 9,
  TFLITE_TENSOR_TYPE_FLOAT16 = 10,
} NpTFLiteTensorType;

typedef uint32_t TFLiteTensorType;

typedef enum {
  NP_INFERENCE_TYPE_NONE = 0,
  NP_INFERENCE_TYPE_QNAUT = 1,
  NP_INFERENCE_TYPE_FLOAT = 2,
} NpInferenceType;

typedef uint32_t InferenceType;

typedef enum {
  // Use CPU to inference the model
  NP_ACCELERATION_CPU = 0,
  // Turns on Android NNAPI for hardware acceleration when it is available.
  NP_ACCELERATION_NNAPI = 1,
  // Use Neuron Delegate
  NP_ACCELERATION_NEURON = 2,
  // Use TFLITE GPU delegate
  NP_ACCELERATION_GPU = 9999,
} NpAccelerationMode;

typedef uint32_t AccelerationMode;

typedef struct {
    // The data type specification for data stored in `data`. This affects
    // what member of `data` union should be used.
    TFLiteTensorType type;
    // Tensor shapes
    int dimsSize;
    int dims[TFLITE_TENSOR_MAX_DIMENSTIONS];
    // Data pointer. The appropriate type should be used for a typed
    // tensor based on `type`.
    // The memory pointed by this data pointer is managed by ANeuralNetworksTFLite instance.
    // Caller should not try to free this pointer.
    void* buffer;

    // Correct the error naming from TFLiteTensor, this is actual buffer size in byte.
    size_t bufferSize;
} TFLiteTensorExt;

typedef struct {
    const char* op_name;
    const char* target_name;
    const char* vendor_name;
    void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
    void (*free)(TfLiteContext* context, void* buffer);
    TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
    TfLiteStatus (*add_params)(void*, ANeuralNetworksModel*, std::vector<uint32_t>&, uint32_t&);
} TFLiteCustomOpExt;


/*************************************************************************************************/

__BEGIN_DECLS

/**
 * Create an {@link ANeuralNetworksTFLiteOptions} with the following default
 * setting.
 * - Compute with parallel execution by default.
 * - Compute float model with FP16 precision by default.
 *
 * <p>{@link ANeuroPilotTFLiteOptions_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} to be created.
 *               Set to NULL if unsuccessful.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLiteOptions_create(ANeuralNetworksTFLiteOptions** options);

/**
 * Specifies whether {@link ANeuralNetworksTFLiteOptions} is allowed to be
 * calculated with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format.
 * This function is only used with float model.
 * A float model is calculated with FP16 precision by default.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True to allow FP16 precision if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setAllowFp16PrecisionForFp32(
    ANeuralNetworksTFLiteOptions* options, bool allow);

/**
 * Set preferred acceleration mode.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param mode Refer to {@link NpAccelerationMode} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setAccelerationMode(
    ANeuralNetworksTFLiteOptions* options, AccelerationMode mode);

/**
 * Set compilation cache directory.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param user define cache directory.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setCacheDir(
    ANeuralNetworksTFLiteOptions* options, const char* cache_dir);

/**
 * Set Execution Preference.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution preference refer to {@link ExecutionPreference} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setPreference(
    ANeuralNetworksTFLiteOptions* options, int execution_preference);

/**
 * Set Disallow NnApi Cpu.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param disallow nnapi cpu.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setDisallowNnApiCpu(
    ANeuralNetworksTFLiteOptions* options, bool disallow_nnapi_cpu);

/**
 * Set Cacheable Ion buffer.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param cacheable Ion buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setCacheableIonBuffer(ANeuralNetworksTFLiteOptions* options,
                                                          bool cacheable_ion_buffer);

/**
 * Set Use Ion.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param Use Ion.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setUseIon(ANeuralNetworksTFLiteOptions* options,
                                                          bool use_ion);

/**
 * Set no supported operation check.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param no supported operation check.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setNoSupportedOperationCheck(ANeuralNetworksTFLiteOptions* options,
                                                          bool no_supported_operation_check);

/**
 * Set Accelerator Name.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param accelerator name.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setAcceleratorName(
    ANeuralNetworksTFLiteOptions* options, const char* accelerator_name);

/**
 * Set Execution Priority.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution prioriy refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setExecutionPriority(
    ANeuralNetworksTFLiteOptions* options, int execution_priority);

/**
 * Set Max Compilation Timeout.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max compilation timeout.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setMaxCompilationTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_compilation_timeout_duration_ns);

/**
 * Set Max Execution Timeout.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max execution timeout.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setMaxExecutionTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_timeout_duration_ns);

/**
 * Set Max Execution Loop Timeout.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max execution loop timeout.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setMaxExecutionLoopTimeout(
    ANeuralNetworksTFLiteOptions* options, uint64_t max_execution_loop_timeout_duration_ns);

/**
 * Set encryption level.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param encryption level refer to {@link NpEncryptionLevel} enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setEncryptionLevel(
    ANeuralNetworksTFLiteOptions* options,
    int encryption_level);

/**
 * Change the dimensionality of a given input tensor.
 *
 * @param options The instance.
 * @param inputIndex The index of the input tensor.
 * @param inputDims List of the dimensions.
 * @param inputDimsSize Number of the dimensions.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_resizeInputTensor(
    ANeuralNetworksTFLiteOptions* options, int32_t index, const int* dims,
    int32_t dimsSize);

/**
 * Allow to maximize the bandwidth utilization for low latency.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableLowLatency True to allow low latency if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setLowLatency(
    ANeuralNetworksTFLiteOptions* options,
    bool enableLowLatency);

/**
 * Allows deep fusion optimization. This may increase the model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableDeepFusion True to allow deep fusion if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setDeepFusion(
    ANeuralNetworksTFLiteOptions* options,
    bool enableDeepFusion);


/**
 * Allows set compile options by string.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param compileOptions set by string.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setCompileOptionByString(ANeuralNetworksTFLiteOptions* options,
                                              const char* compileOptions);
/**
 * Allows batch optimization of models with an N dimension greater than 1.
 * This may increase the model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param enableDeepFusion True to allow deep fusion if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setBatchProcessing(
    ANeuralNetworksTFLiteOptions* options,
    bool enableBatchProcessing);

/**
 * Set the number of warm up runs to do after the {@link ANeuroPilotTFLite} instance is created.
 * This may increase the model initialzation time.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param warmupRuns The number of warmup runs.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setWarmupRuns(
    ANeuralNetworksTFLiteOptions* options,
    uint32_t warmupRuns);

/**
 * Sets the model execution boost hint.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param boostValue The hint for the device frequency, ranged between 0 (lowest) to 100 (highest).
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setBoostHint(
    ANeuralNetworksTFLiteOptions* options,
    uint8_t boostValue);

/**
 * Specifies whether to allow extreme performance acceleration of model execution in Neuron
 * acceleration mode {@link NpAccelerationMode} by acquiring other system resources at the cost of
 * increased power consumption.
 * By default, NeuroPilot will apply extreme performance in Neuron acceleration mode {@link
 * NpAccelerationMode} + fast-single-answer {@link ExecutionPreference}.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True to apply extreme performance if possible.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setBoostDuration(
    ANeuralNetworksTFLiteOptions* options, uint32_t duration=2000);

/**
 * Specifies whether to use AhardwareBuffer
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True use AhardwareBuffer
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setUseAhwb(
    ANeuralNetworksTFLiteOptions* options, bool use_ahwb);

/**
 * Specifies whether to use Ion
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param allow True use Ion
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setUseIon(
    ANeuralNetworksTFLiteOptions* options, bool use_ion);

/** Set GPU Execution Preference.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param execution preference refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setGpuExecutionPreference(
    ANeuralNetworksTFLiteOptions* options, int execution_preference);

/**
 * Set GPU Execution Priority.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param priority index means set the setting to priority 1 or 2 or 3.
 * @param priority setting refer to enum definition.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
*/
int ANeuroPilotTFLiteOptions_setGpuExecutionPriority(
    ANeuralNetworksTFLiteOptions* options, int priority_index, int priority_setting);

/**
 * Delete a {@link ANeuralNetworksTFLiteOptions} object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
void ANeuroPilotTFLiteOptions_free(ANeuralNetworksTFLiteOptions* options);

/**
 * Create an {@link ANeuralNetworksTFLite} with the TFlite model stored in a
 * file.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *               Set to NULL if unsuccessful.
 * @param modelPath The full path of the tflite model file.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_create(ANeuralNetworksTFLite** tflite,
                             const char* modelPath);
int ANeuroPilotTFLite_createAdv(ANeuralNetworksTFLite** tflite,
                                const char* modelPath,
                                ANeuralNetworksTFLiteOptions* options);
/**
 * Create an {@link ANeuralNetworksTFLite} with the TFLite model stored in a
 * data buffer pointer. The data buffer will be duplicated in
 * ANeuralNetworksTFLite instance. Caller could free the input data buffer after
 * calling this API.
 *
 * <p>This only creates the object. Computation is performed once
 * {@link ANeuroPilotTFLite_invoke} is invoked.
 *
 * <p>{@link ANeuroPilotTFLite_free} should be called once the instance
 * is no longer needed.</p>
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to be created.
 *              Set to NULL if unsuccessful.
 * @param buffer The pointer to the tflite model buffer.
 * @param bufferSize The number of bytes of the tflite model buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_createWithBuffer(ANeuralNetworksTFLite** tflite,
                                       const char* buffer, size_t bufferSize);

int ANeuroPilotTFLite_createNeuronModelWithBuffer(NeuronModel** neuron_model,
                                                              const char* buffer,
                                                              const size_t bufferSize,
                                                              uint32_t* neuron_input_index,
                                                              uint32_t* neuron_output_index,
                                                              uint32_t* current_neuron_index);

int ANeuroPilotTFLite_createAdvWithBuffer(
    ANeuralNetworksTFLite** tflite, const char* buffer, size_t bufferSize,
    ANeuralNetworksTFLiteOptions* options);

int ANeuroPilotTFLite_createCustom(ANeuralNetworksTFLite** tflite,
                                   const char* modelPath,
                                   const std::vector<TFLiteCustomOpExt>& customOperations);
int ANeuroPilotTFLite_createAdvCustom(ANeuralNetworksTFLite** tflite,
                                   const char* modelPath,
                                   const std::vector<TFLiteCustomOpExt>& customOperations,
                                   ANeuralNetworksTFLiteOptions* options);

int ANeuroPilotTFLite_createCustomWithBuffer(ANeuralNetworksTFLite** tflite,
                                   const char* buffer,
                                   size_t bufferSize,
                                   const std::vector<TFLiteCustomOpExt>& customOperations);
int ANeuroPilotTFLite_createAdvCustomWithBuffer(ANeuralNetworksTFLite** tflite,
                                   const char* buffer,
                                   size_t bufferSize,
                                   const std::vector<TFLiteCustomOpExt>& customOperations,
                                   ANeuralNetworksTFLiteOptions* options);

/**
 * Get the number of input/output tensors associated with the model.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/out
 * tensor.
 * @param btype Input or output tensor.
 * @param count the number of input/output tensors.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getTensorCount(ANeuralNetworksTFLite* tflite,
                                     TFLiteBufferType btype, int32_t* count);

/**
 * Get the dimensional information of the input/output tensor with the given
 * index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/out
 * tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param rank The rank of the tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getTensorRank(ANeuralNetworksTFLite* tflite,
                                    TFLiteBufferType btype, int index,
                                    int* rank);

/**
 * Get the dimensional information of the input/output tensor with the given
 * index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/out
 * tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param dimensions The dimension array to be filled. The size of the array
 *                   must be exactly as large as the rank.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getTensorDimensions(ANeuralNetworksTFLite* tflite,
                                          TFLiteBufferType btype, int index,
                                          int* dimensions);

/**
 * Get the size of the underlying data in bytes.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output
 * tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param size The tensor's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getTensorByteSize(ANeuralNetworksTFLite* tflite,
                                        TFLiteBufferType btype, int index,
                                        size_t* size);

/**
 * Get the data type information of the input/output tensor with the given
 * index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output
 * tensor.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param ttpte The tensor's data type.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getTensorType(ANeuralNetworksTFLite* tflite,
                                    TFLiteBufferType btype, int index,
                                    TFLiteTensorType* ttype);

/**
 * Get the data type information of the input/output tensor with the given
 * index.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output
 * tensor.
 * @param btype Input or output tensor.
 * @param tensorIndex Zero-based index of tensor.
 * @param data The buffer.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_setTensorBuffer(ANeuralNetworksTFLite* tflite, int tensorIndex, char* data);

/**
 * Copies from the provided input buffer into the tensor's buffer.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/out
 * tensor.
 * @param index Zero-based index of the input tensor.
 * @param data The input buffer.
 * @param size The input buffer's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_setInputTensorData(ANeuralNetworksTFLite* tflite,
                                         int index, const void* data,
                                         size_t size);

/**
 * Copies to the provided output buffer from the output tensor's buffer.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the output
 * tensor.
 * @param index Zero-based index of the output tensor.
 * @param data The output buffer.
 * @param size The output buffer's size in bytes.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_getOutputTensorData(ANeuralNetworksTFLite* tflite,
                                          int index, void* data, size_t size);

/**
 * Store dequantized contents of the given output tensor to user-allocated
 * buffer. This function is only used with quantized model.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to get dequantized data from
 * a given output tensor.
 * @param buffer The pointer to the user-allocated buffer for storing
 * dequantized contents.
 * @param bufferByteSize Specifies the buffer size in bytes.
 * @param tensorIndex Zero-based index of the output tensor.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_getDequantizedOutputByIndex(ANeuralNetworksTFLite* tflite,
                                                  void* buffer,
                                                  size_t bufferByteSize,
                                                  int tensorIndex);

/**
 * Invoke inference. (run the whole graph in dependency order).
 *
 * @param tflite The {@link ANeuralNetworksTFLite} to invoke inference.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 *         ANEURALNETWORKS_OP_FAILED if the operation is failed.
 */
int ANeuroPilotTFLite_invoke(ANeuralNetworksTFLite* tflite);

/**
 * Delete a {@link ANeuralNetworksTFLite} object.
 *
 * Destroys the object used by the run time to keep track of the memory.
 * This will free the underlying actual memory if no other code has open
 * handles to this memory.
 *
 * @param memory The {@link ANeuralNetworksTFLite} object to be freed.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
void ANeuroPilotTFLite_free(ANeuralNetworksTFLite* tflite);

int ANeuroPilot_getInferencePreference(void);

/**
 * Deprecated
 * Get a tensor data structure. This function returns the first input or output
 * tensor.
 *
 * @param tflite The instance to get input/out tensor.
 * @param btype Input or output tensor.
 * @param tfliteTensor A pointer to store the tensor data structure.
 */
int ANeuroPilotTFLite_getTensor(ANeuralNetworksTFLite* tflite,
                                TFLiteBufferType btype,
                                TFLiteTensorExt* tfliteTensor);

/**
 * Deprecated
 * Get a tensor data structure. This function returns the input or output tensor
 * by the given index.
 *
 * @param tflite The instance to get input/out tensor.
 * @param btype Input or output tensor.
 * @param tfliteTensor A pointer to store the tensor data structure.
 * @param tensorIndex Zero-based index of tensor.
 */
int ANeuroPilotTFLite_getTensorByIndex(ANeuralNetworksTFLite* tflite,
                                       TFLiteBufferType btype,
                                       TFLiteTensorExt* tfliteTensor,
                                       int tensorIndex);

/**
 * Deprecated
 * Bind a {@link ANeuralNetworksTFLite} instance to the specified
 * device.(CPU/GPU/APU)
 *
 * @param tflite The instance.
 * @param device Device
 * ID.(ANEURALNETWORKS_CPU/ANEURALNETWORKS_GPU/ANEURALNETWORKS_APU)
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_bindToDeivce(ANeuralNetworksTFLite* tflite,
                                   uint32_t device);

/**
 * Deprecated
 * Set a {@link ANeuralNetworksTFLite} instance to use parallel execution or not
 *
 * @param tflite The instance.
 * @param enableParallel True to enable parallel execution.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_setExecParallel(ANeuralNetworksTFLite* tflite,
                                      bool enableParallel);

/**
 * Set input/ouput with Ahardwarebuffer and get buffer virtual address.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param memory_data Get Ahardwarebuffer virtual address.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 * @param cacheable Decide cacheable/non-cacheable buffer.
 * @param buffer_size Set buffer size.
 *
 * Available since API level 31.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_setBufferHandle(ANeuralNetworksTFLite* tflite,
        void** memory_data, TFLiteBufferType btype, int index, bool cacheable, int buffer_size);

/**
 * Set input/ouput with Ahardwarebuffer and get buffer virtual address.
 *
 * @param tflite The {@link ANeuralNetworksTFLite} which holds the input/output tensor.
 * @param buffer Set Ahardwarebuffer.
 * @param btype Input or output tensor.
 * @param index Zero-based index of tensor.
 *
 * Available since API level 31.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 *         ANEURALNETWORKS_BAD_STATE if NeuroPilot is not supported.
 */
int ANeuroPilotTFLite_setAhwb(ANeuralNetworksTFLite* tflite,
        AHardwareBuffer* buffer, TFLiteBufferType btype, int index);

/**
 * Set Max number delegated partition in NNAPI.
 *
 * @param options The {@link ANeuralNetworksTFLiteOptions} instance.
 * @param max number delegates partitions.
 *
 * Available only in NNAPI Delegate.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLiteOptions_setMaxNumberDelegatedPartitions(
    ANeuralNetworksTFLiteOptions* options, uint32_t max_number_delegated_partitions);
/**
 * Deprecated
 * Set a {@link ANeuralNetworksTFLite} instance to use parallel execution or not
 * Specifies whether {@link ANeuralNetworksTFLite} is allowed to be calculated
 * with range and/or precision as low as that of the IEEE 754 16-bit
 * floating-point format.
 * This function is only used with float model.
 * A float is calculated with FP16 by default.
 *
 * @param tflite The instance.
 * @param enableParallel True to enable parallel execution.
 *
 * @return ANEURALNETWORKS_NO_ERROR if successful.
 */
int ANeuroPilotTFLite_setAllowFp16PrecisionForFp32(
    ANeuralNetworksTFLite* tflite, bool allow);

int ANeuroPilotTFLiteCustomOp_getIntAttribute(const char* buffer, size_t length,
                                              const char* attr, int32_t* outValue);

int ANeuroPilotTFLiteCustomOp_getFloatAttribute(const char* buffer, size_t length,
                                                    const char* attr, float* outValue);

void* ANeuroPilotTFLiteCustomOp_getUserData(TfLiteNode* node);

int ANeuroPilotTFLiteCustomOp_resizeOutput(TfLiteContext* context,
                                       TfLiteNode* node,
                                       int index,
                                       TfLiteIntArray* new_size);

int ANeuroPilotTFLiteCustomOp_getInput(TfLiteContext* context, TfLiteNode* node,
                              int index, TFLiteTensorExt *tfliteTensor);

int ANeuroPilotTFLiteCustomOp_getOutput(TfLiteContext* context, TfLiteNode* node,
                               int index, TFLiteTensorExt *tfliteTensor);

TfLiteIntArray* ANeuroPilotTFLite_createIntArray(int size);

int ANeuroPilotTFLite_freeIntArray(TfLiteIntArray* v);

__END_DECLS

#endif  // __MTK_NEUROPILOT_C_API_H__
