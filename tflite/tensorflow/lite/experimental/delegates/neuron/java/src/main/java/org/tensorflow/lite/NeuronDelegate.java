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
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED ""AS IS"", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

package com.mediatek.neuropilot_S.neuron;

import com.mediatek.neuropilot_S.Delegate;
import com.mediatek.neuropilot_S.TensorFlowLite;

/** {@link Delegate} for Neuron inference. */
public class NeuronDelegate implements Delegate, AutoCloseable {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private long delegateHandle;

  /** Delegate options. */
  public static final class Options {
    public Options() {}

    /**
     * undefined, specifies default behavior. so far, the default setting of NEURON is
     * EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
     */
    public static final int EXECUTION_PREFERENCE_UNDEFINED = -1;

    /**
     * Prefer executing in a way that minimizes battery drain. This is desirable for compilations
     * that will be executed often.
     */
    public static final int EXECUTION_PREFERENCE_LOW_POWER = 0;

    /**
     * Prefer returning a single answer as fast as possible, even if this causes more power
     * consumption.
     */
    public static final int EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER = 1;

    /**
     * Prefer maximizing the throughput of successive frames, for example when processing successive
     * frames coming from the camera.
     */
    public static final int EXECUTION_PREFERENCE_SUSTAINED_SPEED = 2;

    public static final int EXECUTION_PRIORITY_LOW = 90;
    public static final int EXECUTION_PRIORITY_MEDIUM = 100;
    public static final int EXECUTION_PRIORITY_HIGH = 110;

    /**
     * Sets the inference preference for precision/compilation/runtime tradeoffs.
     *
     * @param preference One of EXECUTION_PREFERENCE_LOW_POWER,
     *     EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER, and EXECUTION_PREFERENCE_SUSTAINED_SPEED.
     */
    public Options setExecutionPreference(int preference) {
      this.executionPreference = preference;
      return this;
    }

    /**
     * Sets execution priority.
     *
     * @param preference One of EXECUTION_PRIORITY_LOW,
     *     EXECUTION_PRIORITY_MEDIUM, and EXECUTION_PRIORITY_HIGH.
     */
    public Options setExecutionPrioriy(int executionPriority) {
      this.executionPriority = executionPriority;
      return this;
    }

    /**
     * Sets enable low latency.
     *
     */
    public Options setEnableLowLatency(boolean enableLowLatency) {
      this.enableLowLatency = enableLowLatency;
      return this;
    }

    /**
     * Sets enable deep fusion.
     *
     */
    public Options setEnableDeepFusion(boolean enableDeepFusion) {
      this.enableDeepFusion = enableDeepFusion;
      return this;
    }

    /**
     * Sets enable batch processing.
     *
     */
    public Options setEnableBatchProcessing(boolean enableBatchProcessing) {
      this.enableBatchProcessing = enableBatchProcessing;
      return this;
    }

    /**
     * Sets boost value.
     *
     */
    public Options setBoostValue(int boostValue) {
      this.boostValue = boostValue;
      return this;
    }

    /**
     * Sets boost duration.
     *
     */
    public Options setBoostDuration(int boostDuration) {
      this.boostDuration = boostDuration;
      return this;
    }

    /**
     * Set compile options for neuron delegate.
     *
     * <p>Only effective on Android 12 (API level 31) and above.
     */
    public Options setCompileOptions(String compileOptions) {
      this.compileOptions = compileOptions;
      return this;
    }

    /**
     * Specifies the name of the target accelerator to be used by NNAPI. If this parameter is
     * specified the {@link #setUseNnapiCpu(boolean)} method won't have any effect.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setAcceleratorName(String name) {
      this.acceleratorName = name;
      return this;
    }

    /**
     * Configure the location to be used to store model compilation cache entries. If either {@code
     * cacheDir} or {@code modelToken} parameters are unset NNAPI caching will be disabled.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setCacheDir(String cacheDir) {
      this.cacheDir = cacheDir;
      return this;
    }

    /**
     * Sets the token to be used to identify this model in the model compilation cache. If either
     * {@code cacheDir} or {@code modelToken} parameters are unset NNAPI caching will be disabled.
     *
     * <p>Only effective on Android 10 (API level 29) and above.
     */
    public Options setModelToken(String modelToken) {
      this.modelToken = modelToken;
      return this;
    }

    /**
     * Sets the maximum number of graph partitions that the delegate will try to delegate. If more
     * partitions could be delegated than the limit, the ones with the larger number of nodes will
     * be chosen. If unset it will use the NNAPI default limit.
     */
    public Options setMaxNumberOfDelegatedPartitions(int limit) {
      this.maxDelegatedPartitions = limit;
      return this;
    }

    /**
     * Enable or disable to allow fp32 computation to be run in fp16 in NNAPI. See
     * https://source.android.com/devices/neural-networks#android-9
     *
     * <p>Only effective on Android 9 (API level 28) and above.
     */
    public Options setAllowFp16(boolean enable) {
      this.allowFp16 = enable;
      return this;
    }

    /**
     * Enable or disable to use Ion
     *
     * <p>Only effective on Android 11 (API level 30)
     */
    public Options setUseIon(boolean useIon) {
      this.useIon = useIon;
      return this;
    }

    /**
     * Enable or disable to use Ahwb
     *
     * <p>Only effective on Android 12 (API level 31)
     */
    public Options setUseAhwb(boolean useAhwb) {
      this.useAhwb = useAhwb;
      return this;
    }

    private int executionPreference = EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER;
    private String acceleratorName = null;
    private String cacheDir = null;
    private String modelToken = null;
    private Integer maxDelegatedPartitions = null;
    private Boolean allowFp16 = null;
    private int executionPriority = EXECUTION_PRIORITY_MEDIUM;
    private boolean enableLowLatency;
    private boolean enableDeepFusion;
    private boolean enableBatchProcessing;
    private int boostValue = -1;
    private int boostDuration;
    private String compileOptions = null;
    private boolean useAhwb = true;
    private boolean useIon;
  }

  public NeuronDelegate(Options options) {
    // Ensure the native TensorFlow Lite libraries are available.
    TensorFlowLite.init();
    delegateHandle =
        createDelegate(
            options.executionPreference,
            options.acceleratorName,
            options.cacheDir,
            options.modelToken,
            options.maxDelegatedPartitions != null ? options.maxDelegatedPartitions : -1,
            options.allowFp16 != null ? options.allowFp16 : false,
            options.executionPriority,
            options.enableLowLatency,
            options.enableDeepFusion,
            options.enableBatchProcessing,
            options.boostValue,
            options.boostDuration,
            options.compileOptions,
            options.useAhwb,
            options.useIon);
  }

  public NeuronDelegate() {
    this(new Options());
  }

  @Override
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      deleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  private static native long createDelegate(
      int preference,
      String deviceName,
      String cacheDir,
      String modelToken,
      int maxDelegatedPartitions,
      boolean allowFp16,
      int executionPriority,
      boolean enableLowLatency,
      boolean enableDeepFusion,
      boolean enableBatchProcessing,
      int boostValue,
      int boostDuration,
      String compileOptions,
      boolean useAhwb,
      boolean useIon);

  private static native void deleteDelegate(long delegateHandle);

}
