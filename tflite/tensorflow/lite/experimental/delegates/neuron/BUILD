load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")
load("//tensorflow/lite/experimental/delegates/neuron:build_def.bzl", "android_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "neuronapi_lib",
    hdrs = [
        "neuron_types.h",
    ],
)

cc_library(
    name = "neuron_delegate_utils",
    hdrs = [
        "neuron_delegate_utils.h",
    ],
)

cc_library(
    name = "neuron_implementation",
    srcs = ["neuron_implementation.cc"],
    hdrs = [
        "neuron_implementation.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    linkopts = android_linkopts(),
    deps = [
        ":neuronapi_lib",
    ],
)

cc_library(
    name = "neuron_delegate_kernel",
    srcs = [
        "neuron_delegate.h",
        "neuron_delegate_builder.h",
        "neuron_delegate_kernel.cc",
        "neuron_nnapi_delegate_kernel.h",
    ],
    hdrs = [
        "neuron_delegate_kernel.h",
        "APUWareUtilsApi.h",
    ],
    linkopts = android_linkopts(),
    deps = [
        "neuron_delegate_utils",
        "neuron_delegate_validation",
        "neuron_implementation",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/delegates/utils:simple_delegate",
	"//tensorflow/lite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_library(
    name = "neuron_delegate_validation",
    srcs = [
        "neuron_delegate_validation.cc",
    ],
    hdrs = ["neuron_delegate_validation.h"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "neuronapi_lib",
        "neuron_delegate_utils",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "neuron_delegate",
    srcs = [
        "neuron_delegate.cc",
    ],
    hdrs = [
        "neuron_delegate.h",
        "APUWareUtilsApi.h",
    ],
    deps = [
        "neuron_delegate_kernel",
        "neuron_implementation",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
    ],
)

#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "neuron_delegate_provider",
    srcs = ["neuron_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":neuron_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_neuron_delegate",
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)

cc_binary(
    name = "label_image_plus_neuron_delegate",
    srcs = ["//tensorflow/lite/examples/label_image:label_image.cc"],
    copts = tflite_copts(),
    linkopts = android_linkopts(),
    deps = [
        ":neuron_delegate_provider",
        "//tensorflow/lite/examples/label_image",
        "//tensorflow/lite/examples/label_image:bitmap_helpers",
        "//tensorflow/lite/tools/delegates:tflite_execution_providers",
    ],
)

exports_files([
    "neuron_delegate.h",
    "version_script.lds",
])
