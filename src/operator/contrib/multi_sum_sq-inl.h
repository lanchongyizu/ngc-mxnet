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
 * \file multi_l2_norm-inl.h
 * \brief vectorized L2 norm over multiple arrays operators
 * \author Clement Fuji Tsang
 */


#ifndef MXNET_OPERATOR_CONTRIB_MULTI_SUM_SQ_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTI_SUM_SQ_INL_H_

#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct MultiSumSqParam : public dmlc::Parameter<MultiSumSqParam> {
  int num_arrays;
  DMLC_DECLARE_PARAMETER(MultiSumSqParam) {
    DMLC_DECLARE_FIELD(num_arrays)
    .describe("number of input arrays.");
  }
};

inline bool MultiSumSqShape(const NodeAttrs& attrs,
                            std::vector<TShape>* in_shape,
                            std::vector<TShape>* out_shape) {
  const auto& p = dmlc::get<MultiSumSqParam>(attrs.parsed);
  out_shape->resize(1);
  SHAPE_ASSIGN_CHECK(*out_shape, 0,  TShape{p.num_arrays});
  CHECK_EQ(in_shape->size(), p.num_arrays);
  for (auto s : *in_shape) {
    if (s.ndim() == 0)
      return false;
  }
  return true;
}

inline bool MultiSumSqType(const NodeAttrs& attrs,
                           std::vector<int>* in_type,
                           std::vector<int>* out_type) {
  const auto& param_ = dmlc::get<MultiSumSqParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param_.num_arrays);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, "array_" + std::to_string(i));
    }
  }
  out_type->clear();
  out_type->push_back(mshadow::kFloat32);
  return true;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_MULTI_SUM_SQ_INL_H_
