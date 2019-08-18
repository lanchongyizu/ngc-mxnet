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
 *  Copyright (c) 2019 by Contributors
 * \file multi_sum_sq.cc
 * \brief vectorized sum or squared over multiple arrays operators
 * \author Clement Fuji Tsang
 */

#include "./multi_sum_sq-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MultiSumSqParam);

NNVM_REGISTER_OP(multi_sum_sq)
.describe(R"code(Compute the sums of squares of multiple arrays
)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    return static_cast<uint32_t>(dmlc::get<MultiSumSqParam>(attrs.parsed).num_arrays);
  })
.set_num_outputs(1)
.set_attr_parser(ParamParser<MultiSumSqParam>)
.set_attr<nnvm::FInferShape>("FInferShape", MultiSumSqShape)
.set_attr<nnvm::FInferType>("FInferType", MultiSumSqType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiSumSqParam>(attrs.parsed).num_arrays;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("array_") + std::to_string(i));
    }
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol[]", "Arrays")
.add_arguments(MultiSumSqParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
