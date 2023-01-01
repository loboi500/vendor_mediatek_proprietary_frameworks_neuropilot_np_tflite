/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/mtk/kernels/internal/reference/mtk_reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "flatbuffers/flexbuffers.h"

namespace tflite {
namespace ops {
namespace mtk {
namespace custom_op {


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    return const_cast<char*>(buffer);
}


}  // namespace custom_op

TfLiteRegistration* Register_MTK_CUSTOM_OP_REF() {
    static TfLiteRegistration r = {custom_op::Init,
                                   nullptr,
                                   nullptr,
                                   nullptr};
    return &r;
}

TfLiteRegistration* Register_MTK_CUSTOM_OP() {
    return Register_MTK_CUSTOM_OP_REF();
}


}  // namespace mtk
}  // namespace ops
}  // namespace tflite

