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
 * Copyright (c) 2018 by Contributors
 * \file tensorrt-inl.h
 * \brief
 * \author Marek Kolodziej
*/
#ifndef MXNET_OPERATOR_TENSORRT_INL_H_
#define MXNET_OPERATOR_TENSORRT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "./elemwise_op_common.h"
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"
#include "./channel_op_common.h"
#include "./tensor/broadcast_reduce_op.h"
#include "../../nnvm/src/top/op_common.h"

namespace mxnet {
namespace op {

struct TensorRTParam : public dmlc::Parameter<TensorRTParam> {
  int num_inputs;
  int num_outputs;
  DMLC_DECLARE_PARAMETER(TensorRTParam) {
    DMLC_DECLARE_FIELD(num_inputs).set_lower_bound(1)
    .describe("Number of inputs to the TensoRT subgraph operator");
    DMLC_DECLARE_FIELD(num_outputs).set_default(1)
    .describe("Number of outputs out of the TensorRT subgraph operator");
  }
};  // struct TensorRTParam


template<typename xpu, typename DType>
class TensorRTOp : public Operator {
 public:
  explicit TensorRTOp(TensorRTParam param)
    : num_inputs_(param.num_inputs), num_outputs_(param.num_outputs) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    using nnvm::TShape;
    using std::cout;
    using std::endl;

    CHECK_EQ(static_cast<int>(in_data.size()), num_inputs_);
    CHECK_EQ(static_cast<int>(out_data.size()), num_outputs_);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    for (int i = 0; i < num_outputs_; i++) {
        Tensor<xpu, 2, DType> tensor_in = in_data[i].get<xpu, 2, DType>(s);
        Tensor<xpu, 2, DType> tensor_out = out_data[i].get<xpu, 2, DType>(s);
        Assign(tensor_out, req[i], F<mshadow_op::identity>(tensor_in));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    throw dmlc::Error("TensorRT is for forward pass only");
  }

 private:
  int num_inputs_;
  int num_outputs_;
};  // class TensorRTOp

template<typename xpu>
Operator *CreateOp(TensorRTParam param, int dtype, std::vector<TShape> *in_shape);

#if DMLC_USE_CXX11
class TensorRTProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_inputs; ++i) {
      ret.push_back(std::string("arg") + std::to_string(i));
    }
    return ret;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    out_type->clear();
    out_type->reserve(param_.num_outputs);
    for (int i = 0; i < param_.num_outputs; ++i) {
      out_type->push_back(dtype);
    }
    aux_type->clear();
    return true;
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    out_shape->clear();
    for (TShape t : *in_shape) {
        out_shape->push_back(t);
    }
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new TensorRTProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "TensorRT";
  }

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> ret;
    for (int i = 0; i < param_.num_outputs; ++i) {
      std::ostringstream os;
      os << "output" << i;
      ret.push_back(os.str());
    }
    return ret;
  }

  int NumOutputs() const override {
    return param_.num_outputs;
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  TensorRTParam param_;
};  // class TensorRTProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSORRT_INL_H_
