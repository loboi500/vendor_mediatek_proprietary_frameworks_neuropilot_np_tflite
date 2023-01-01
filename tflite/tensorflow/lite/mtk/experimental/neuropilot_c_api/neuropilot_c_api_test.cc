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

#include "flatbuffers/flexbuffers.h"
#include "gtest/gtest.h"
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"
#include "tensorflow/lite/mtk/experimental/neuropilot_c_api/NeuroPilotTFLiteShim.h"
#include "tensorflow/lite/experimental/delegates/neuron/neuron_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

using namespace std;

#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tuple>

// Float model with single AvgPool OP
// Input [10, 784]
// Output [10, 14, 14, 1]
typedef struct {
    int width;
    int height;
} PaddingValues;

typedef struct {
    int stride_width;
    int stride_height;
    int filter_width;
    int filter_height;
    bool pad_same;  // true: SAME, false:VALID
    PaddingValues padding_values;
} CustomMaxPoolParams;

static constexpr int kInputRank = 2;
static constexpr int kInputDims[kInputRank] = {10, 784};
static constexpr size_t kInputSize = (10 * 784 * sizeof(float));
static constexpr int kOutputRank = 4;
static constexpr int kOutputDims[kOutputRank] = {10, 14, 14, 1};
static constexpr size_t kOutputSize = (10 * 14 * 14 * 1 * sizeof(float));
static constexpr char kAvgPoolModel[] = {
    0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x54, 0x01, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x54, 0x4f, 0x43, 0x4f,
    0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x3c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x8c, 0xfd, 0xff, 0xff, 0x90, 0xfd, 0xff, 0xff, 0x94, 0xfd, 0xff, 0xff,
    0x1e, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xb4, 0xfd, 0xff, 0xff,
    0x0c, 0x00, 0x14, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x0c, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xf4, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00,
    0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x05, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0e, 0x00, 0x16, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00,
    0x07, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x01, 0x00, 0x00, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x06, 0x00, 0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x06, 0x00, 0x06, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x06, 0x00,
    0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x01, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0xec, 0x00, 0x00, 0x00, 0xa0, 0x00, 0x00, 0x00,
    0x72, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x10, 0x03, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x50, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x72, 0x00,
    0x08, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x43, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xc6, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x52, 0x65, 0x73, 0x68, 0x61, 0x70, 0x65, 0x00, 0xbc, 0xff, 0xff, 0xff, 0x00, 0x00, 0x0e, 0x00,
    0x14, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x79, 0x73, 0x00, 0x00, 0x04, 0x00, 0x06, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x08, 0x00, 0x07, 0x00, 0x0c, 0x00,
    0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x10, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x52, 0x65, 0x73, 0x68, 0x61, 0x70, 0x65, 0x2f,
    0x73, 0x68, 0x61, 0x70, 0x65, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00};

struct OptimizationHint {
    bool lowLatency;
    bool deepFusion;
    bool batchProcessing;
};

class TFLiteCApiPreferenceTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    TFLiteCApiPreferenceTest() = default;
};

class TFLiteCApiPriorityTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    TFLiteCApiPriorityTest() = default;
};

class TFLiteCApiAllowFp16Test : public ::testing::TestWithParam<std::tuple<int, bool>> {
protected:
    TFLiteCApiAllowFp16Test() = default;
};

class TFLiteCApiOptimizationHintTest : public ::testing::TestWithParam<OptimizationHint> {
protected:
    TFLiteCApiOptimizationHintTest() = default;
};

class TFLiteCApiAccelerationModeTest : public ::testing::TestWithParam<int> {
protected:
    TFLiteCApiAccelerationModeTest() = default;
};

class TFLiteCApiBoostHintTest : public ::testing::TestWithParam<uint8_t> {
protected:
    TFLiteCApiBoostHintTest() = default;
};

class TFLiteCApiExtremePerformanceTest : public ::testing::TestWithParam<bool> {
protected:
    TFLiteCApiExtremePerformanceTest() = default;
};

TEST(TFLiteCApiTest, NullInstance) {
    const char* model_path = "/data/local/tmp/model_test.lite";
    int ret = ANeuroPilotTFLiteWrapper_makeTFLite(nullptr, model_path);
    ASSERT_EQ(ANEURALNETWORKS_UNEXPECTED_NULL, ret);
}

TEST(TFLiteCApiTest, NullModelPath) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLite(&tflite, nullptr);
    ASSERT_EQ(ANEURALNETWORKS_UNEXPECTED_NULL, ret);
}

TEST(TFLiteCApiTest, CreateNonExistentModel) {
    // Try to load a non-existent model.
    const char* model_path = "/data/local/tmp/model_test.lite";
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLite(&tflite, model_path);
    ANeuroPilotTFLiteWrapper_free(tflite);
    ASSERT_EQ(ANEURALNETWORKS_BAD_DATA, ret);
}

void file_to_buffer(const char* tflite_path, char** tflite_buffer, long* size) {
    char* buffer = 0;
    long length;
    FILE *f = fopen(tflite_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        length = ftell(f);
        fseek(f, 0, SEEK_SET);
        buffer = (char*)malloc(length);
        if (buffer) {
            fread(buffer, 1, length, f);
        }
        fclose(f);
        *tflite_buffer = buffer;
        *size = length;
    }
}

TEST(TFLiteCApiTest, NeuronModelTest) {
    NeuronModel* neuron_model = nullptr;
    const NeuronApi* neuronapi = NeuronApiImplementation();
    neuronapi->NeuronModel_create(&neuron_model);
    uint32_t operand_dim0[] = {1, 224, 224, 3};
    uint32_t operand_dim1[] = {1, 14, 14, 512};
    uint32_t operand_dim2[] = {1, 1001};
    NeuronOperandType operand_type0{.type=NEURON_TENSOR_QUANT8_ASYMM, .dimensionCount=4, .dimensions=operand_dim0, .scale=0.007843137718737125, .zeroPoint=128};
    NeuronOperandType operand_type1{.type=NEURON_TENSOR_QUANT8_ASYMM, .dimensionCount=4, .dimensions=operand_dim1, .scale=0.023528477177023888, .zeroPoint=0};
    NeuronOperandType operand_type2{.type=NEURON_TENSOR_QUANT8_ASYMM, .dimensionCount=2, .dimensions=operand_dim2, .scale=0.00390625, .zeroPoint=0};
    neuronapi->NeuronModel_addOperand(neuron_model, &operand_type0);
    neuronapi->NeuronModel_addOperand(neuron_model, &operand_type1);
    neuronapi->NeuronModel_addOperand(neuron_model, &operand_type2);
    uint32_t current_neuron_index = 3;

    const char* tflite_path1 = "/sdcard/top.tflite";
    char* tflite_buffer1 = nullptr;
    long tflite_size1;
    file_to_buffer(tflite_path1, &tflite_buffer1, &tflite_size1);
    uint32_t input_index1[] = {0};
    uint32_t output_index1[] = {1};
    ANeuroPilotTFLiteWrapper_makeNeuronModelWithBuffer(&neuron_model,
                                                       tflite_buffer1,
                                                       tflite_size1,
                                                       input_index1,
                                                       output_index1,
                                                       &current_neuron_index);

    cout << current_neuron_index << endl;
    cout << "first done" << endl;

    const char* tflite_path2 = "/sdcard/bottom.tflite";
    char* tflite_buffer2 = nullptr;
    long tflite_size2;
    file_to_buffer(tflite_path2, &tflite_buffer2, &tflite_size2);
    uint32_t input_index2[] = {1};
    uint32_t output_index2[] = {2};
    ANeuroPilotTFLiteWrapper_makeNeuronModelWithBuffer(&neuron_model,
                                                       tflite_buffer2,
                                                       tflite_size2,
                                                       input_index2,
                                                       output_index2,
                                                       &current_neuron_index);
    cout << "second done" << endl;

    neuronapi->NeuronModel_relaxComputationFloat32toFloat16(neuron_model, true);
    neuronapi->NeuronModel_identifyInputsAndOutputs(neuron_model, 1, input_index1, 1, output_index2);
    neuronapi->NeuronModel_finish(neuron_model);
    NeuronCompilation* compilation = nullptr;
    neuronapi->NeuronCompilation_create(neuron_model, &compilation);
    neuronapi->NeuronCompilation_finish(compilation);
    NeuronExecution* execution = nullptr;
    neuronapi->NeuronExecution_create(compilation, &execution);

    void* input = reinterpret_cast<void*>(calloc(1, 150528));
    uint8_t* p = reinterpret_cast<uint8_t*>(input);
    for (auto i = 0; i < 150528; i++) {
        *p++ = 7;
    }
    void* output = reinterpret_cast<void*>(malloc(1001));
    neuronapi->NeuronExecution_setInput(execution, 0, nullptr, input, 150528);
    neuronapi->NeuronExecution_setOutput(execution, 0, nullptr, output, 1001);
    neuronapi->NeuronExecution_compute(execution);

    cout << "compute done" << endl;

    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    ANeuralNetworksTFLiteOptions_create(&options);
    ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite, "/sdcard/model.tflite", options);
    ANeuroPilotTFLiteWrapper_setInputTensorData(tflite, 0, input, 150528);
    ANeuroPilotTFLiteWrapper_invoke(tflite);
    void* output_buffer = reinterpret_cast<void*>(malloc(1001));
    ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite, 0, output_buffer, 1001);
    if (!memcmp(output, output_buffer, 1001)) {
        cout << "compare success!" << endl;
    } else {
        cout << "compare fail!" << endl;
    }
}

// Sanity check for the state-ful NNAPI delegate using TfLiteBufferHandle.
TEST(TFLiteCApiTest, StatefulDelegateWithBufferHandles) {

  ANeuralNetworksTFLite* tflite = nullptr;

  ANeuralNetworksTFLiteOptions* options = nullptr;
  ANeuralNetworksTFLiteOptions_create(&options);

  ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);

  ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite, "/sdcard/mobilenet_v1_1.0_224_uint8.tflite", options);

  size_t in_buffer_size = 0;
  ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                   &in_buffer_size);

  void* input_memory_data = nullptr;

  ANeuroPilotTFLiteWrapper_setBufferHandle(tflite, &input_memory_data, TFLITE_BUFFER_TYPE_INPUT, 0,
                                           true, in_buffer_size);

  uint8_t* o = reinterpret_cast<uint8_t*>(input_memory_data);
  for (auto i = 0; i < in_buffer_size; i++) {
      *o++ = 7;
  }

  size_t out_buffer_size = 0;
  ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                   &out_buffer_size);

  void* output_memory_data= nullptr;

  ANeuroPilotTFLiteWrapper_setBufferHandle(tflite, &output_memory_data, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                           true, out_buffer_size);

  ANeuroPilotTFLiteWrapper_invoke(tflite);

  ANeuralNetworksTFLite* tflite2 = nullptr;
  ANeuralNetworksTFLiteOptions* options2 = nullptr;
  ANeuralNetworksTFLiteOptions_create(&options2);

  ANeuralNetworksTFLiteOptions_setAccelerationMode(options2, NP_ACCELERATION_NEURON);

  ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite2, "/sdcard/mobilenet_v1_1.0_224_uint8.tflite", options2);

  void* input2 = reinterpret_cast<void*>(calloc(1, in_buffer_size));
  uint8_t* p = reinterpret_cast<uint8_t*>(input2);
  for (auto i = 0; i < in_buffer_size; i++) {
      *p++ = 7;
  }
  ANeuroPilotTFLiteWrapper_setInputTensorData(tflite2, 0, input2, in_buffer_size);
  ANeuroPilotTFLiteWrapper_invoke(tflite2);
  void* output_buffer2 = reinterpret_cast<void*>(malloc(out_buffer_size));
  ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite2, 0, output_buffer2, out_buffer_size);

  //std::ofstream myFileA ("/data/local/tmp/myFileA_ori.bin", ios::out | ios::binary);
  //myFileA.write((char*)output_memory_data, out_buffer_size);

  //std::ofstream myFileB ("/data/local/tmp/myFileB_ori.bin", ios::out | ios::binary);
  //myFileB.write((char*)output_buffer2, out_buffer_size);

  if (!memcmp(output_memory_data, output_buffer2, out_buffer_size)) {
      cout << "compare success!" << endl;
  } else {
      cout << "compare fail!" << endl;
  }
}

static size_t getNumPaddingBytes(size_t byte_size) {
  size_t num_padding_bytes = 0;
  if (byte_size % 128) {
    num_padding_bytes = 128 -
                        (byte_size % 128);
  }
  return num_padding_bytes;
}

// Sanity check for the state-ful NNAPI delegate using TfLiteBufferHandle.
TEST(TFLiteCApiTest, SetAhwbTest) {

  // init
  ANeuralNetworksTFLite* tflite = nullptr;

  ANeuralNetworksTFLiteOptions* options = nullptr;
  ANeuralNetworksTFLiteOptions_create(&options);

  ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);

  ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite, "/sdcard/mobilenet_v1_1.0_224_uint8.tflite", options);
  //end init

  //set input
  size_t in_buffer_size = 0;
  ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                   &in_buffer_size);

  uint64_t usage=AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

  in_buffer_size += getNumPaddingBytes(in_buffer_size);

  AHardwareBuffer_Desc desc {
      .width=static_cast<uint32_t>(in_buffer_size),
      .height=1,
      .layers=1,
      .format=AHARDWAREBUFFER_FORMAT_BLOB,
      .usage=usage,
      .stride=static_cast<uint32_t>(in_buffer_size),
  };

  AHardwareBuffer* buffer = nullptr;

  AHardwareBuffer_allocate(&desc, &buffer);

  void* input_memory_data = nullptr;

  AHardwareBuffer_lock(buffer, usage, -1, NULL, reinterpret_cast<void**>(&input_memory_data));

  uint8_t* o = reinterpret_cast<uint8_t*>(input_memory_data);
  for (auto i = 0; i < in_buffer_size; i++) {
      *o++ = 7;
  }
  AHardwareBuffer_unlock(buffer, nullptr);

  ANeuroPilotTFLiteWrapper_setAhwb(tflite, buffer, TFLITE_BUFFER_TYPE_INPUT, 0);
  //end set input

  //set output
  size_t out_buffer_size = 0;

  ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                   &out_buffer_size);
  size_t out_buffer_size_original = out_buffer_size;

  out_buffer_size += getNumPaddingBytes(out_buffer_size);

  AHardwareBuffer_Desc out_desc {
      .width=static_cast<uint32_t>(out_buffer_size),
      .height=1,
      .layers=1,
      .format=AHARDWAREBUFFER_FORMAT_BLOB,
      .usage=usage,
      .stride=static_cast<uint32_t>(out_buffer_size),
  };

  AHardwareBuffer* out_buffer = nullptr;

  AHardwareBuffer_allocate(&out_desc, &out_buffer);

  ANeuroPilotTFLiteWrapper_setAhwb(tflite, out_buffer, TFLITE_BUFFER_TYPE_OUTPUT, 0);
  //end set output

  // other inf
  ANeuroPilotTFLiteWrapper_invoke(tflite);

  ANeuralNetworksTFLite* tflite2 = nullptr;
  ANeuralNetworksTFLiteOptions* options2 = nullptr;
  ANeuralNetworksTFLiteOptions_create(&options2);

  ANeuralNetworksTFLiteOptions_setAccelerationMode(options2, NP_ACCELERATION_NEURON);

  ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite2, "/sdcard/mobilenet_v1_1.0_224_uint8.tflite", options2);

  void* input2 = reinterpret_cast<void*>(calloc(1, in_buffer_size));
  ASSERT_TRUE(input2 != nullptr);
  uint8_t* p = reinterpret_cast<uint8_t*>(input2);
  for (auto i = 0; i < in_buffer_size; i++) {
      *p++ = 7;
  }
  ANeuroPilotTFLiteWrapper_setInputTensorData(tflite2, 0, input2, in_buffer_size);
  ANeuroPilotTFLiteWrapper_invoke(tflite2);
  // end other inf

  //compare
  void* output_buffer2 = reinterpret_cast<void*>(malloc(out_buffer_size_original));
  ASSERT_TRUE(output_buffer2 != nullptr);
  ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite2, 0, output_buffer2, out_buffer_size_original);

  void* output_memory_data = nullptr;

  AHardwareBuffer_lock(out_buffer, usage, -1, NULL, reinterpret_cast<void**>(&output_memory_data));

  //write file

  //std::ofstream myFileA ("/data/local/tmp/myFileA.bin", ios::out | ios::binary);
  //myFileA.write((char*)output_memory_data, out_buffer_size_original);

  //std::ofstream myFileB ("/data/local/tmp/myFileB.bin", ios::out | ios::binary);
  //myFileB.write((char*)output_buffer2, out_buffer_size_original);
  if (output_buffer2 != nullptr && !memcmp(output_memory_data, output_buffer2, out_buffer_size_original)) {
      cout << "compare success!" << endl;
  } else {
      cout << "compare fail!" << endl;
  }
  AHardwareBuffer_unlock(out_buffer, nullptr);
  AHardwareBuffer_release(out_buffer);
  AHardwareBuffer_release(buffer);

  if (output_buffer2 != nullptr) {
        free(output_buffer2);
  }
  if (input2 != nullptr) {
        free(input2);
  }
}

TEST(TFLiteCApiTest, CreateWithBuffer) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                            sizeof(kAvgPoolModel));
    ANeuroPilotTFLiteWrapper_free(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
}

TEST(TFLiteCApiTest, Create) {
    remove("/data/local/tmp/model.lite");
    std::ofstream tflite_file("/data/local/tmp/model.lite",
                              std::ios::out | std::ios::app | std::ios::binary);
    ASSERT_TRUE(tflite_file.is_open());

    tflite_file.write(kAvgPoolModel, sizeof(kAvgPoolModel));
    tflite_file.close();

    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLite(&tflite, "/data/local/tmp/model.lite");
    ANeuroPilotTFLiteWrapper_free(tflite);

    remove("/data/local/tmp/model.lite");
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
}

TEST(TFLiteCApiTest, GetTensorCount) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                            sizeof(kAvgPoolModel));
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    int32_t count = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorCount(tflite, TFLITE_BUFFER_TYPE_INPUT, &count);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(count, 1);
    ret = ANeuroPilotTFLiteWrapper_getTensorCount(tflite, TFLITE_BUFFER_TYPE_OUTPUT, &count);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(count, 1);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, GetInputTensor) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                            sizeof(kAvgPoolModel));
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    int rank = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorRank(tflite, TFLITE_BUFFER_TYPE_INPUT, 0, &rank);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(rank, kInputRank);
    int diemssions[rank];
    ret = ANeuroPilotTFLiteWrapper_getTensorDimensions(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                       diemssions);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    for (auto i = 0; i < rank; i++) {
        ASSERT_EQ(kInputDims[i], diemssions[i]);
    }
    size_t buffer_size = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                     &buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(kInputSize, buffer_size);
    TFLiteTensorType tensr_type = TFLITE_TENSOR_TYPE_NONE;
    ret = ANeuroPilotTFLiteWrapper_getTensorType(tflite, TFLITE_BUFFER_TYPE_INPUT, 0, &tensr_type);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(TFLITE_TENSOR_TYPE_FLOAT, tensr_type);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, GetOutputTensor) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                            sizeof(kAvgPoolModel));
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    int rank = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorRank(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &rank);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(rank, kOutputRank);
    int diemssions[rank];
    ret = ANeuroPilotTFLiteWrapper_getTensorDimensions(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                       diemssions);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    for (auto i = 0; i < rank; i++) {
        ASSERT_EQ(kOutputDims[i], diemssions[i]);
    }
    size_t buffer_size = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                     &buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(kOutputSize, buffer_size);
    TFLiteTensorType tensr_type = TFLITE_TENSOR_TYPE_NONE;
    ret = ANeuroPilotTFLiteWrapper_getTensorType(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &tensr_type);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(TFLITE_TENSOR_TYPE_FLOAT, tensr_type);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, Invoke) {
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                            sizeof(kAvgPoolModel));
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    size_t in_buffer_size = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                     &in_buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    void* input_buffer = reinterpret_cast<void*>(calloc(1, in_buffer_size));
    float* p = reinterpret_cast<float*>(input_buffer);
    for (auto i = 0; i < in_buffer_size / sizeof(float); i++) {
        *p = rand() % 256;
        p++;
    }
    ret = ANeuroPilotTFLiteWrapper_setInputTensorData(tflite, 0, input_buffer, in_buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    int rank = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorRank(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &rank);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(rank, kOutputRank);

    int diemssions[rank];
    ret = ANeuroPilotTFLiteWrapper_getTensorDimensions(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                       diemssions);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    size_t out_buffer_size = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                     &out_buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    void* output_buffer = reinterpret_cast<void*>(calloc(1, out_buffer_size));
    ASSERT_NE(output_buffer, nullptr);
    ret = ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite, 0, output_buffer, out_buffer_size);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    if (input_buffer != nullptr) {
        free(input_buffer);
    }
    if (output_buffer != nullptr) {
        free(output_buffer);
    }

    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, ResizeInputTensor) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    static constexpr int dims[kInputRank] = {8, 784};
    ret = ANeuralNetworksTFLiteOptions_resizeInputTensor(options, 0, dims, kInputRank);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    int rank = 0;
    ret = ANeuroPilotTFLiteWrapper_getTensorRank(tflite, TFLITE_BUFFER_TYPE_INPUT, 0, &rank);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ASSERT_EQ(rank, kInputRank);
    int diemssions[rank];
    ret = ANeuroPilotTFLiteWrapper_getTensorDimensions(tflite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                                       diemssions);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    for (auto i = 0; i < rank; i++) {
        ASSERT_EQ(dims[i], diemssions[i]);
    }
    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

/*static const char custom_max_pool_quant_model[] = {
    0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x20, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x54, 0x4f, 0x43, 0x4f,
    0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xf8, 0xff, 0xff, 0xff,
    0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x18, 0xff, 0xff, 0xff,
    0x20, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1c, 0x01, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x86, 0xff, 0xff, 0xff,
    0x00, 0x00, 0x00, 0x03, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x50, 0x6c, 0x61, 0x63,
    0x65, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x72, 0x00, 0x84, 0xff, 0xff, 0xff, 0x30, 0x00, 0x00, 0x00,
    0x24, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x80, 0x3f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x43, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x08, 0x00, 0x07, 0x00, 0x0c, 0x00,
    0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x10, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0d, 0x00, 0x00, 0x00, 0x43, 0x75, 0x73, 0x74, 0x6f, 0x6d, 0x4d, 0x61, 0x78, 0x50, 0x6f, 0x6f,
    0x6c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x14, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x43,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x14, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00,
    0x00, 0x00, 0x10, 0x00, 0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
    0x6b, 0x73, 0x69, 0x7a, 0x65, 0x00, 0x04, 0x01, 0x02, 0x02, 0x01, 0x70, 0x61, 0x64, 0x64, 0x69,
    0x6e, 0x67, 0x00, 0x04, 0x53, 0x41, 0x4d, 0x45, 0x00, 0x5f, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74,
    0x5f, 0x71, 0x75, 0x61, 0x6e, 0x74, 0x69, 0x7a, 0x65, 0x64, 0x00, 0x73, 0x74, 0x72, 0x69, 0x64,
    0x65, 0x73, 0x00, 0x04, 0x01, 0x02, 0x02, 0x01, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x66, 0x6f, 0x72,
    0x6d, 0x61, 0x74, 0x00, 0x04, 0x4e, 0x48, 0x57, 0x43, 0x00, 0x05, 0x32, 0x14, 0x4d, 0x43, 0x24,
    0x05, 0x01, 0x05, 0x01, 0x0f, 0x4e, 0x42, 0x23, 0x68, 0x14, 0x2c, 0x14, 0x2c, 0x0a, 0x24, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x08, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x04, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x43, 0x75, 0x73, 0x74, 0x6f, 0x6d, 0x4d, 0x61, 0x78, 0x50, 0x6f, 0x6f, 0x6c, 0x00, 0x00, 0x00};

static const char custom_max_pool_model[] = {
    0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00, 0x18, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0xd0, 0x01, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x54, 0x4f, 0x43, 0x4f,
    0x20, 0x43, 0x6f, 0x6e, 0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x14, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x20, 0xff, 0xff, 0xff,
    0x24, 0xff, 0xff, 0xff, 0x28, 0xff, 0xff, 0xff, 0x0c, 0x00, 0x14, 0x00, 0x04, 0x00, 0x08, 0x00,
    0x0c, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x08, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0xa2, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x1c, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x50, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f, 0x6c, 0x64, 0x65, 0x72, 0x00, 0x08, 0x00, 0x0c, 0x00,
    0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x43, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x04, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x43, 0x75, 0x73, 0x74,
    0x6f, 0x6d, 0x4d, 0x61, 0x78, 0x50, 0x6f, 0x6f, 0x6c, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x14, 0x00, 0x14, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x14, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
    0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61, 0x00, 0x00, 0x00, 0x6b, 0x73, 0x69, 0x7a,
    0x65, 0x00, 0x04, 0x01, 0x02, 0x02, 0x01, 0x70, 0x61, 0x64, 0x64, 0x69, 0x6e, 0x67, 0x00, 0x05,
    0x56, 0x41, 0x4c, 0x49, 0x44, 0x00, 0x5f, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x71, 0x75,
    0x61, 0x6e, 0x74, 0x69, 0x7a, 0x65, 0x64, 0x00, 0x73, 0x74, 0x72, 0x69, 0x64, 0x65, 0x73, 0x00,
    0x04, 0x01, 0x02, 0x02, 0x01, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x66, 0x6f, 0x72, 0x6d, 0x61, 0x74,
    0x00, 0x04, 0x4e, 0x48, 0x57, 0x43, 0x00, 0x05, 0x32, 0x14, 0x4e, 0x44, 0x24, 0x05, 0x01, 0x05,
    0x01, 0x0f, 0x4f, 0x43, 0x23, 0x68, 0x14, 0x2c, 0x14, 0x2c, 0x0a, 0x24, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x08, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x04, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x43, 0x75, 0x73, 0x74, 0x6f, 0x6d, 0x4d, 0x61, 0x78, 0x50, 0x6f, 0x6f, 0x6c, 0x00, 0x00, 0x00};

static int computePadding(int stride, int dilation_rate, int in_size, int filter_size,
                          int out_size) {
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
    return padding > 0 ? padding : 0;
}

static void* initCustomMaxPool(TfLiteContext* context, const char* buffer, size_t length) {
    (void)context;
    CustomMaxPoolParams* user_data = new CustomMaxPoolParams;
    auto map = flexbuffers::GetRoot((const uint8_t*)buffer, length).AsMap();
    cout << "kernel type: " << map["ksize"].GetType() << endl;
    cout << "strides type: " << map["strides"].GetType() << endl;
    cout << "padding type: " << map["padding"].GetType() << endl;
    auto kernel_vec = map["ksize"].AsTypedVector();
    auto strides_vec = map["strides"].AsTypedVector();
    cout << "kernel size: " << kernel_vec.size() << endl;
    cout << "stride size: " << strides_vec.size() << endl;

    user_data->filter_width = kernel_vec[2].AsInt8();
    user_data->filter_height = kernel_vec[1].AsInt8();
    user_data->stride_width = strides_vec[2].AsInt8();
    user_data->stride_height = strides_vec[1].AsInt8();
    string string_pad_same("SAME");
    user_data->pad_same = map["padding"].ToString().compare(string_pad_same) == 0 ? true : false;
    cout << "pad_same: " << user_data->pad_same << endl;
    cout << "filter_width: " << user_data->filter_width << endl;
    cout << "filter_height: " << user_data->filter_height << endl;
    cout << "stride_width: " << user_data->stride_width << endl;
    cout << "stride_height: " << user_data->stride_height << endl;

    return user_data;
}

static void freeCustomMaxPool(TfLiteContext* context, void* buffer) {
    (void)context;
    // Free the CustomMaxPoolParams buffer allocated in initCustomMaxPool()
    delete reinterpret_cast<CustomMaxPoolParams*>(buffer);
}

static TfLiteStatus prepareCustomMaxPool(TfLiteContext* context, TfLiteNode* node) {
    CustomMaxPoolParams* user_data =
        reinterpret_cast<CustomMaxPoolParams*>(ANeuroPilotTFLiteWrapper_getCustomOpUserData(node));

    TFLiteTensorExt input;

    if (ANeuroPilotTFLiteWrapper_getCustomOpInput(context, node, 0, &input) !=
        ANEURALNETWORKS_NO_ERROR) {
        cout << "Can not get input tensor" << endl;
        return kTfLiteError;
    }

    if (input.dimsSize != 4) {
        cout << "Wrong input dimension:" << input.dimsSize << "!= 4" << endl;
        return kTfLiteError;
    }

    int batches = input.dims[0];
    int height = input.dims[1];
    int width = input.dims[2];
    int channels_out = input.dims[3];

    auto pad_same = user_data->pad_same;
    auto computeOutSize = [pad_same](int imageSize, int filterSize, int stride) -> int {
        return pad_same == true
                   ? (imageSize + stride - 1) / stride
                   : pad_same == false ? (imageSize - filterSize + stride) / stride : 0;
    };

    int outWidth = computeOutSize(width, user_data->filter_width, user_data->stride_width);
    int outHeight = computeOutSize(height, user_data->filter_height, user_data->stride_height);

    user_data->padding_values.height =
        computePadding(user_data->stride_height, 1, height, user_data->filter_height, outHeight);
    user_data->padding_values.width =
        computePadding(user_data->stride_width, 1, width, user_data->filter_width, outWidth);

    // Prepare the output dimension according to the input and max_disp attribute.
    TfLiteIntArray* outputSize = ANeuroPilotTFLiteWrapper_createIntArray(4);
    outputSize->data[0] = batches;
    outputSize->data[1] = outHeight;
    outputSize->data[2] = outWidth;
    outputSize->data[3] = channels_out;

    // Resize the output dimension
    int ret = ANeuroPilotTFLiteWrapper_resizeCustomOpOutput(context, node, 0, outputSize);
    return (ret == ANEURALNETWORKS_NO_ERROR ? kTfLiteOk : kTfLiteError);
}

static TfLiteStatus addCustomMaxPoolParams(void* data, ANeuralNetworksModel* nn_model,
                                           vector<uint32_t>& augmented_inputs, uint32_t& next_id) {
    CustomMaxPoolParams* user_data = reinterpret_cast<CustomMaxPoolParams*>(data);

    auto add_scalar_int32 = [&nn_model, &augmented_inputs, &next_id](int value) {
        ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};

        if (ANeuralNetworksModel_addOperand(nn_model, &operand_type) != ANEURALNETWORKS_NO_ERROR) {
            cout << "Fail to add operand to NN" << endl;
        }

        if (ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value, sizeof(int32_t)) !=
            ANEURALNETWORKS_NO_ERROR) {
            cout << "Fail to set operand value to NN" << endl;
        }

        augmented_inputs.push_back(next_id++);
    };
    cout << "addCustomMaxPoolParams()" << endl;
    add_scalar_int32(user_data->pad_same == true ? ANEURALNETWORKS_PADDING_SAME
                                                 : ANEURALNETWORKS_PADDING_VALID);
    add_scalar_int32(user_data->stride_width);
    add_scalar_int32(user_data->stride_height);
    add_scalar_int32(user_data->filter_width);
    add_scalar_int32(user_data->filter_height);
    add_scalar_int32(ANEURALNETWORKS_FUSED_NONE);
    return kTfLiteOk;
}

TEST(TFLiteCApiTest, DISABLED_CreateCustomWithBuffer) {
    const vector<TFLiteCustomOpExt> customOperations = {
        {
            .op_name = "CustomMaxPool",
            .target_name = "gpu",
            .vendor_name = "mtk",
            .init = initCustomMaxPool,
            .free = freeCustomMaxPool,
            .prepare = prepareCustomMaxPool,
            .add_params = addCustomMaxPoolParams,
        },
    };
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeCustomTFLiteWithBuffer(
        &tflite, custom_max_pool_model, sizeof(custom_max_pool_model), customOperations);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    if (ret == ANEURALNETWORKS_NO_ERROR) {
        ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    }
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, DISABLED_CreateCustomWithQuantBuffer) {
    const vector<TFLiteCustomOpExt> customOperations = {
        {
            .op_name = "CustomMaxPool",
            .target_name = "vpu",
            .vendor_name = "mtk",
            .init = initCustomMaxPool,
            .free = freeCustomMaxPool,
            .prepare = prepareCustomMaxPool,
            .add_params = addCustomMaxPoolParams,
        },
    };
    ANeuralNetworksTFLite* tflite = nullptr;
    int ret = ANeuroPilotTFLiteWrapper_makeCustomTFLiteWithBuffer(
        &tflite, custom_max_pool_quant_model, sizeof(custom_max_pool_quant_model),
        customOperations);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    if (ret == ANEURALNETWORKS_NO_ERROR) {
        ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    }
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);
    ANeuroPilotTFLiteWrapper_free(tflite);
}*/

TEST(TFLiteCApiTest, SetPreference) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setPreference(options, ExecutionPreference::kLowPower);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetCacheableIonBuffer) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ret = ANeuralNetworksTFLiteOptions_setCacheableIonBuffer(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetUseIon) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setUseIon(options, true);
    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ret = ANeuralNetworksTFLiteOptions_setCacheableIonBuffer(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetNoSupportedOperationCheck) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setNoSupportedOperationCheck(options, true);
    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ret = ANeuralNetworksTFLiteOptions_setCacheableIonBuffer(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiAccelerationModeTest, ParameterizedTest) {
    int mode = GetParam();
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, mode);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiPreferenceTest, ParameterizedTest) {
    int mode = std::get<0>(GetParam());
    int preference = std::get<1>(GetParam());
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, mode);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setPreference(options, preference);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiPriorityTest, ParameterizedTest) {
    int mode = std::get<0>(GetParam());
    int priority = std::get<1>(GetParam());
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, mode);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setExecutionPriority(options, priority);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiAllowFp16Test, ParameterizedTest) {
    int mode = std::get<0>(GetParam());
    bool allowFp16 = std::get<1>(GetParam());
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, mode);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(options, allowFp16);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiOptimizationHintTest, ParameterizedTest) {
    bool lowLatency = GetParam().lowLatency;
    bool deepFusion = GetParam().deepFusion;
    bool batchProcessing = GetParam().batchProcessing;

    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setLowLatency(options, lowLatency);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setDeepFusion(options, deepFusion);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setBatchProcessing(options, batchProcessing);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiBoostHintTest, ParameterizedTest) {
    uint8_t boostValue = GetParam();

    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setPreference(options, kSustainedSpeed);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setBoostHint(options, boostValue);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST_P(TFLiteCApiExtremePerformanceTest, ParameterizedTest) {
    bool allow = GetParam();

    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAllowExtremePerformance(options, allow, 1000);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

INSTANTIATE_TEST_CASE_P(ParameterizedTest, TFLiteCApiAccelerationModeTest,
                        ::testing::Values(NP_ACCELERATION_CPU, NP_ACCELERATION_NNAPI,
                                          NP_ACCELERATION_NEURON));

INSTANTIATE_TEST_CASE_P(
    ParameterizedTest, TFLiteCApiPreferenceTest,
    ::testing::Values(std::make_tuple(NP_ACCELERATION_NNAPI, kUndefined),
                      std::make_tuple(NP_ACCELERATION_NNAPI, kLowPower),
                      std::make_tuple(NP_ACCELERATION_NNAPI, kFastSingleAnswer),
                      std::make_tuple(NP_ACCELERATION_NNAPI, kSustainedSpeed),
                      std::make_tuple(NP_ACCELERATION_NEURON, kUndefined),
                      std::make_tuple(NP_ACCELERATION_NEURON, kLowPower),
                      std::make_tuple(NP_ACCELERATION_NEURON, kFastSingleAnswer),
                      std::make_tuple(NP_ACCELERATION_NEURON, kSustainedSpeed)));

INSTANTIATE_TEST_CASE_P(
    ParameterizedTest, TFLiteCApiPriorityTest,
    ::testing::Values(std::make_tuple(NP_ACCELERATION_NNAPI, ANEURALNETWORKS_PRIORITY_LOW),
                      std::make_tuple(NP_ACCELERATION_NNAPI, ANEURALNETWORKS_PRIORITY_MEDIUM),
                      std::make_tuple(NP_ACCELERATION_NNAPI, ANEURALNETWORKS_PRIORITY_HIGH),
                      std::make_tuple(NP_ACCELERATION_NEURON, ANEURALNETWORKS_PRIORITY_LOW),
                      std::make_tuple(NP_ACCELERATION_NEURON, ANEURALNETWORKS_PRIORITY_MEDIUM),
                      std::make_tuple(NP_ACCELERATION_NEURON, ANEURALNETWORKS_PRIORITY_HIGH)));

INSTANTIATE_TEST_CASE_P(ParameterizedTest, TFLiteCApiAllowFp16Test,
                        ::testing::Values(std::make_tuple(NP_ACCELERATION_NNAPI, true),
                                          std::make_tuple(NP_ACCELERATION_NNAPI, false),
                                          std::make_tuple(NP_ACCELERATION_NEURON, true),
                                          std::make_tuple(NP_ACCELERATION_NEURON, false)));

INSTANTIATE_TEST_CASE_P(
    ParameterizedTest, TFLiteCApiOptimizationHintTest,
    ::testing::Values(OptimizationHint{true, true, true}, OptimizationHint{true, true, false},
                      OptimizationHint{true, false, false}, OptimizationHint{false, false, false},
                      OptimizationHint{true, false, true}, OptimizationHint{false, false, true},
                      OptimizationHint{false, true, true}));

INSTANTIATE_TEST_CASE_P(ParameterizedTest, TFLiteCApiBoostHintTest,
                        ::testing::Values(100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0));

INSTANTIATE_TEST_CASE_P(ParameterizedTest, TFLiteCApiExtremePerformanceTest,
                        ::testing::Values(true, false));

TEST(TFLiteCApiTest, SetOptions) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setPreference(options, ExecutionPreference::kLowPower);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    const char* cache_dir = "/data/local/tmp/";
    ret = ANeuralNetworksTFLiteOptions_setCacheDir(options, cache_dir);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setDisallowNnApiCpu(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    const char* accelerator_name = "mtk-mdla";
    ret = ANeuralNetworksTFLiteOptions_setAcceleratorName(options, accelerator_name);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setExecutionPriority(options, ANEURALNETWORKS_PRIORITY_LOW);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setMaxCompilationTimeout(options, 1000000000);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setMaxExecutionTimeout(options, 1000000001);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setMaxExecutionLoopTimeout(options, 1000000002);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetCacheDir) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    const char* cache_dir = "/data/local/tmp/";
    ret = ANeuralNetworksTFLiteOptions_setCacheDir(options, cache_dir);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetWarmupRuns) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setWarmupRuns(options, 10);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

TEST(TFLiteCApiTest, SetAccelerator) {
    ANeuralNetworksTFLite* tflite = nullptr;
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setUseIon(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuralNetworksTFLiteOptions_setUseAhwb(options, true);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    const char* accelerator_name = "mtk-gpu";
    ret = ANeuralNetworksTFLiteOptions_setAcceleratorName(options, accelerator_name);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&tflite, kAvgPoolModel,
                                                           sizeof(kAvgPoolModel), options);
    ASSERT_EQ(ANEURALNETWORKS_NO_ERROR, ret);

    ANeuralNetworksTFLiteOptions_free(options);
    ANeuroPilotTFLiteWrapper_free(tflite);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
