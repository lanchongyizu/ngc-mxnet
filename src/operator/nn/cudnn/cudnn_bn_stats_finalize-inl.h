/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file cudnn_bn_stats_finalize-inl.h
 * \brief
 * \author Dick Carter
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_BN_STATS_FINALIZE_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_BN_STATS_FINALIZE_INL_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "../bn_stats_finalize-inl.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4

#if defined(__CUDACC__)
template<typename DType>
class CuDNNBNStatsFinalizeOp {
 public:
  CuDNNBNStatsFinalizeOp() {
    using namespace mshadow;
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));

#if CUDNN_VERSION >= 7600
    // New fused BNForwardFinalizeStatisticsTraining-op descriptor creation
    CUDNN_CALL(cudnnCreateFusedOpsPlan(&train_op_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING));
    CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(&train_op_const_params_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING));
    CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(&train_op_variant_params_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING));

    // New fused BNForwardFinalizeStatisticsInference-op descriptor creation
    CUDNN_CALL(cudnnCreateFusedOpsPlan(&inference_op_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));
    CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(&inference_op_const_params_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));
    CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(&inference_op_variant_params_,
      CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE));
#endif
  }

  void Init(const BNStatsFinalizeParam &param, const TShape shape,
            const OpContext &ctx) {
    CHECK_GE(param.eps, CUDNN_BN_MIN_EPSILON)
     << "CuDNN requires eps to be no less than " << CUDNN_BN_MIN_EPSILON;
    this->param_ = param;
    InitDescriptors(shape);

#if CUDNN_VERSION >= 7600
    // Set up the 'Const Param Pack' for the BNForwardFinalizeStatisticsTraining op
    // Describe pointer alignments
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_YSUM_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_YSQSUM_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_BIAS_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(train_op_const_params_,
      CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    // Set the I/O descriptors
    // sum and sum_squares input descriptor (typically fp32)
    SetOpParamDesc(train_op_const_params_,
                   CUDNN_PARAM_YSTATS_DESC, in_desc_);
    // scale, bias, running_mean and running_var input descriptors, as well as the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    SetOpParamDesc(train_op_const_params_,
                   CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, in_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    SetOpParamDesc(train_op_const_params_,
                   CUDNN_PARAM_BN_EQSCALEBIAS_DESC, out_desc_);

    // Set up the 'Const Param Pack' for the BNForwardFinalizeStatisticsInference op
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_BIAS_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    SetOpParamVal(inference_op_const_params_,
      CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, CUDNN_PTR_ELEM_ALIGNED);
    // Set the I/O descriptors
    // scale, bias, running_mean and running_var input descriptors, as well as the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    SetOpParamDesc(inference_op_const_params_,
                   CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, in_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    SetOpParamDesc(inference_op_const_params_,
                   CUDNN_PARAM_BN_EQSCALEBIAS_DESC, out_desc_);

    // Set the mode parameter in the ops
    // The ops just want to make sure you don't want CUDNN_BATCHNORM_PER_ACTIVATION.
    SetOpParamVal(train_op_const_params_, CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
    SetOpParamVal(inference_op_const_params_, CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);

    // Now make the training- and inference-op plans and check that the workspace size is 0
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t workspace_size_bytes = 1;
    // Make op plan for both forward training and forward inference ops
    CUDNN_CALL(cudnnMakeFusedOpsPlan(s->dnn_handle_, train_op_,
                                     train_op_const_params_, &workspace_size_bytes));
    CHECK_EQ(workspace_size_bytes, 0U)
      << "Unexpected non-zero workspace size for CuDNNBNStatsFinalize fwd training.";
    // Make op plan for backward wgrad op and set backward wgrad workspace size
    workspace_size_bytes = 1;
    CUDNN_CALL(cudnnMakeFusedOpsPlan(s->dnn_handle_, inference_op_,
                                     inference_op_const_params_, &workspace_size_bytes));
    CHECK_EQ(workspace_size_bytes, 0U)
      << "Unexpected non-zero workspace size for CuDNNBNStatsFinalize fwd inference.";

    // Having verified a 0 workspace, set this in the ops
    cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                              CUDNN_PTR_WORKSPACE, nullptr);
    cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                              CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                                              &workspace_size_bytes);
    cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                              CUDNN_PTR_WORKSPACE, nullptr);
    cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                              CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                                              &workspace_size_bytes);
#endif
  }

  ~CuDNNBNStatsFinalizeOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));

#if CUDNN_VERSION >= 7600
    // New fused BNForwardFinalizeStatisticsTraining-op descriptor destruction
    CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(train_op_variant_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(train_op_const_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsPlan(train_op_));

    // New fused BNForwardFinalizeStatisticsInference-op descriptor destruction
    CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(inference_op_variant_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(inference_op_const_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsPlan(inference_op_));
#endif
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data,
               const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), static_cast<size_t>(bn_stats_finalize::kNumNonAuxInputs));
    CHECK_EQ(aux_states.size(), static_cast<size_t>(bn_stats_finalize::kNumAuxStates));
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs));
      CHECK_EQ(req.size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs));
    } else {
      // Only equiv_scale and equiv_bias
      CHECK_GE(out_data.size(), 2U);
      CHECK_GE(req.size(), 2U);
    }
    CHECK_EQ(req[bn_stats_finalize::kEquivScale], kWriteTo);
    CHECK_EQ(req[bn_stats_finalize::kEquivBias], kWriteTo);
    // All inputs, outputs and aux states should have a shape equal to the one used to init the op
    for (auto &input : in_data)
      CHECK_EQ(input.shape_, init_shape_);
    for (auto &output : out_data)
      CHECK_EQ(output.shape_, init_shape_);
    for (auto &aux_state : aux_states)
      CHECK_EQ(aux_state.shape_, init_shape_);

    int features = in_data[bn_stats_finalize::kSum].shape_[0];

    Stream<gpu> *s = ctx.get_stream<gpu>();
    DType *equiv_scale = Get1dPtr(out_data[bn_stats_finalize::kEquivScale], features, s);
    DType *equiv_bias = Get1dPtr(out_data[bn_stats_finalize::kEquivBias], features, s);

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> sum = in_data[bn_stats_finalize::kSum]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> sum_squares = in_data[bn_stats_finalize::kSumOfSquares]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> gamma = in_data[bn_stats_finalize::kGamma]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> beta = in_data[bn_stats_finalize::kBeta]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_mean = aux_states[bn_stats_finalize::kMovingMean]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_inv_var = aux_states[bn_stats_finalize::kMovingVar]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      if (param_.fix_gamma) gamma = 1.f;

      // Enforce argument types for safety.
      double eps = static_cast<double>(param_.eps);
      double avg_factor = 1.0 - param_.momentum;
      int64_t elem_count = static_cast<int64_t>(param_.elem_count);

      if (ctx.is_train) {
        Tensor<gpu, 1, DTypeParam> save_mean = out_data[bn_stats_finalize::kMean]
            .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        Tensor<gpu, 1, DTypeParam> save_inv_var = out_data[bn_stats_finalize::kInvStdDev]
            .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        // Set data pointers and other light-weight parameters in the 'variant param pack'
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_YSUM, sum.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_YSQSUM, sum_squares.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_SCALE, gamma.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_BIAS, beta.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_SAVED_MEAN, save_mean.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_SAVED_INVSTD, save_inv_var.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_RUNNING_MEAN, moving_mean.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_RUNNING_VAR, moving_inv_var.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_EQSCALE, equiv_scale);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_PTR_BN_EQBIAS, equiv_bias);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT,
                                                  &elem_count);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR,
                                                  &avg_factor);
        cudnnSetFusedOpsVariantParamPackAttribute(train_op_variant_params_,
                                                  CUDNN_SCALAR_DOUBLE_BN_EPSILON, &eps);
        // Launch forward finalize training op
        CUDNN_CALL(cudnnFusedOpsExecute(s->dnn_handle_, train_op_, train_op_variant_params_));
      } else {
        // Set data pointers and other light-weight parameters in the 'variant param pack'
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_SCALE, gamma.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_BIAS, beta.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_RUNNING_MEAN, moving_mean.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_RUNNING_VAR, moving_inv_var.dptr_);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_EQSCALE, equiv_scale);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_PTR_BN_EQBIAS, equiv_bias);
        cudnnSetFusedOpsVariantParamPackAttribute(inference_op_variant_params_,
                                                  CUDNN_SCALAR_DOUBLE_BN_EPSILON, &param_.eps);
        // Launch forward finalize inference op
        CUDNN_CALL(cudnnFusedOpsExecute(s->dnn_handle_, inference_op_,
                                        inference_op_variant_params_));
      }

      if (req.size() > static_cast<size_t>(bn_stats_finalize::kGammaOut)) {
        switch (req[bn_stats_finalize::kGammaOut]) {
          case kWriteInplace:
          case kNullOp:
            break;  // Nothing to do
          case kWriteTo:
            // Copy is only needed when the framework cannot do kWriteInPlace, e.g. where the I/O's
            // of the op correspond to the I/O's of the symbol (as it might in testing).
            CopyDTypeParamTensor(out_data[bn_stats_finalize::kGammaOut],
                                 in_data[bn_stats_finalize::kGamma], ctx);
            break;
          default:
            LOG(FATAL) << "BNStatsFinalize::Forward(): Unexpected req[] for pass-thru gamma: "
                       << req[bn_stats_finalize::kGammaOut];
        }
      }

      if (req.size() > static_cast<size_t>(bn_stats_finalize::kBetaOut)) {
        switch (req[bn_stats_finalize::kBetaOut]) {
          case kWriteInplace:
          case kNullOp:
            break;  // Nothing to do
          case kWriteTo:
            // Copy is only needed when the framework cannot do kWriteInPlace, e.g. where the I/O's
            // of the op correspond to the I/O's of the symbol (as it might in testing).
            CopyDTypeParamTensor(out_data[bn_stats_finalize::kBetaOut],
                                 in_data[bn_stats_finalize::kBeta], ctx);
            break;
          default:
            LOG(FATAL) << "BNStatsFinalize::Forward(): Unexpected req[] for pass-thru beta: "
                       << req[bn_stats_finalize::kBetaOut];
        }
      }
    })
#endif  // CUDNN_VERSION >= 7600
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    // Only incoming gradients (== number of fwd outputs) are inputs of the backward node.
    CHECK_EQ(inputs.size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs));
    // Only outgoing gradients (== number of non-aux fwd inputs) are outputs of the backward node.
    CHECK_EQ(outputs.size(), static_cast<size_t>(bn_stats_finalize::kNumNonAuxInputs));
    CHECK_EQ(req.size(), static_cast<size_t>(bn_stats_finalize::kNumNonAuxInputs));
    // We normally expect to see kWriteInplace here, so we will do nothing.
    switch (req[bn_stats_finalize::kGamma]) {
      case kWriteInplace:
      case kNullOp:
        break;  // Nothing to do
      case kWriteTo:
        // Copy is only needed when the framework cannot do kWriteInPlace, e.g. where the I/O's
        // of the op correspond to the I/O's of the symbol (as it might in testing).
        CopyDTypeParamTensor(outputs[bn_stats_finalize::kGamma],
                             inputs[bn_stats_finalize::kGammaOut], ctx);
        break;
      default:
        LOG(FATAL) << "BNStatsFinalize::Backward(): Unexpected req[] for gamma gradient: "
                   << req[bn_stats_finalize::kGamma];
    }
    switch (req[bn_stats_finalize::kBeta]) {
      case kWriteInplace:
      case kNullOp:
        break;  // Nothing to do
      case kWriteTo:
        // Copy is only needed when the framework cannot do kWriteInPlace, e.g. where the I/O's
        // of the op correspond to the I/O's of the symbol (as it might in testing).
        CopyDTypeParamTensor(outputs[bn_stats_finalize::kBeta],
                             inputs[bn_stats_finalize::kBetaOut], ctx);
        break;
      default:
        LOG(FATAL) << "BNStatsFinalize::Backward(): Unexpected req[] for beta gradient: "
                   << req[bn_stats_finalize::kBeta];
    }
  }

/*!
 * \brief Returns whether the cuDNN library version supports the batchnorm
 * operation described by `param`.
 */
  static bool Supports(const BNStatsFinalizeParam &param, int dtype, const TShape &shape) {
    // TODO(dcarter): update this
    return !param.use_global_stats;
  }

 private:
  // Converts a TBlob to a 1D dptr of the specifed size, checking for that it's contiguous.
  DType *Get1dPtr(const TBlob &tb, int size, mshadow::Stream<gpu> *s) {
    mshadow::Tensor<gpu, 1, DType> data = tb.get<gpu, 1, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    return data.dptr_;
  }

  // Copy a tensor of 'DTypeParam' (typically float).  Used as a fallback if no kWriteInplace.
  void CopyDTypeParamTensor(const TBlob &to_data, const TBlob &from_data, const OpContext &ctx) {
    using namespace mshadow;
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK_EQ(from_data.Size(), to_data.Size()) << "Copy requested of unequal-sized Tensors";
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> from = from_data
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(from_data.Size()), s);
      Tensor<gpu, 1, DTypeParam> to = to_data
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(to_data.Size()), s);
      CUDA_CALL(cudaMemcpyAsync(to.dptr_,
                                from.dptr_,
                                from_data.Size() * sizeof(DTypeParam),
                                cudaMemcpyDeviceToDevice,
                                mshadow::Stream<gpu>::GetStream(s)));
    })
  }

  // Zero a tensor of 'DTypeParam' (typically float).
  void ZeroDTypeParamTensor(const TBlob &to_data, const OpContext &ctx) {
    using namespace mshadow;
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> to = to_data
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(to_data.Size()), s);
      CUDA_CALL(cudaMemsetAsync(to.dptr_,
                                0,
                                to_data.Size() * sizeof(DTypeParam),
                                mshadow::Stream<gpu>::GetStream(s)));
    })
  }

#if CUDNN_VERSION >= 7600
  // Set a fused op parameter by value of type 'T'.  Avoids ptr requirement of underlying call.
  template <typename T>
  void SetOpParamVal(cudnnFusedOpsConstParamPack_t param_pack,
                     cudnnFusedOpsConstParamLabel_t param_label, T param_val) {
    T param_val_copy = param_val;
    CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(param_pack, param_label, &param_val_copy));
  }

  // Set a fused op parameter descriptor given a pointer to the descriptor 'T'.
  template <typename T>
  void SetOpParamDesc(cudnnFusedOpsConstParamPack_t param_pack,
                      cudnnFusedOpsConstParamLabel_t param_label, T *param_ptr) {
    CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(param_pack, param_label, param_ptr));
  }
#endif

  void InitDescriptors(const TShape &shape) {
    using namespace mshadow;
    init_shape_ = shape;
    dim_ = init_shape_.ndim();
    CHECK_EQ(dim_, 1) << "Expecting 1D 'sum' input.";
    int c = init_shape_[0];
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc_,
                                          format,
                                          dtype_,
                                          1, c, 1, 1));
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc_,
                               format,
                               DataType<DTypeParam>::kCudnnFlag,
                               1, c, 1, 1));
    })
  }

  int dim_;
  cudnnDataType_t dtype_;
  int dtype_param_;
  cudnnTensorDescriptor_t out_desc_, in_desc_;
  BNStatsFinalizeParam param_;
  // The shape used to init the descriptors of the op
  TShape init_shape_;

#if CUDNN_VERSION >= 7600
  // New forward BNForwardFinalizeStatisticsTraining op
  cudnnFusedOpsPlan_t train_op_;
  // Op parameters are held in a 'const parameter pack' of descriptor info and data ptr alignments.
  cudnnFusedOpsConstParamPack_t train_op_const_params_;
  // Op I/O data ptrs and 'light-weight' parameters are held in a 'variant param pack'
  cudnnFusedOpsVariantParamPack_t train_op_variant_params_;

  // New fused BNForwardFinalizeStatisticsInference op
  cudnnFusedOpsPlan_t inference_op_;
  // Op parameters are held in a 'const parameter pack' of descriptor info and data ptr alignments.
  cudnnFusedOpsConstParamPack_t inference_op_const_params_;
  // Op I/O data ptrs and 'light-weight' parameters are held in a 'variant param pack'
  cudnnFusedOpsVariantParamPack_t inference_op_variant_params_;
#endif
};
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 4
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_BN_STATS_FINALIZE_INL_H_
