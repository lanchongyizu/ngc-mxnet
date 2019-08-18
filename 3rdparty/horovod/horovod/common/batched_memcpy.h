// Copyright (C) 2019 NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef BATCHED_MEMCPY_H
#define BATCHED_MEMCPY_H

#define BATCHED_D2D_CAPACITY 160
struct BatchedD2DParams {
  void* out[BATCHED_D2D_CAPACITY];
  void* in[BATCHED_D2D_CAPACITY];
  size_t sizes[BATCHED_D2D_CAPACITY];
};

// Performs a batched d2d memcopy
void batched_d2d_memcpy(BatchedD2DParams& params, int num_copies, cudaStream_t stream = 0);

#endif // BATCHED_MEMCPY_H
