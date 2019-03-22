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
 * \file tensorrt.cc
 * \brief
 * \author Marek Kolodziej
*/

#include "./tensorrt-inl.h"

#include <stdexcept>

namespace mxnet {
namespace op {

using nnvm::FInferShape;

template<>
Operator* CreateOp<cpu>(TensorRTParam param, int dtype, std::vector<TShape> *in_shape) {
    throw dmlc::Error("TensorRT only works on GPU");
    return nullptr;
}

Operator* TensorRTProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0), in_shape);
}

DMLC_REGISTER_PARAMETER(TensorRTParam);

MXNET_REGISTER_OP_PROPERTY(TensorRT, TensorRTProp)
.describe(R"code(TensorRT operator for fast inference
)code" ADD_FILELINE)
.set_return_type("NDArray-or-Symbol[]")
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to feed into TensorRT")
.add_arguments(TensorRTParam::__FIELDS__());

NNVM_REGISTER_OP(TensorRT).add_alias("tensorrt");

}  // namespace op
}  // namespace mxnet
