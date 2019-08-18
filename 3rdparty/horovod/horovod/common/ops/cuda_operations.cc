// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#include "cuda_operations.h"
#include "../batched_memcpy.h"

#include <thread>

namespace horovod {
namespace common {

cudaError_t CUDAContext::GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                         cudaEventDisableTiming);
}

cudaError_t CUDAContext::ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    queue.push(event);
  }

  return cudaSuccess;
}

void CUDAContext::ErrorCheck(std::string op_name, cudaError_t cuda_result) {
  if (cuda_result != cudaSuccess) {
    throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
  }
}

void CUDAContext::RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                              std::string name, cudaStream_t& stream) {
  cudaEvent_t event;
  ErrorCheck("GetCudaEvent", GetCudaEvent(&event));
  ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
  event_queue.emplace(name, event);
}

void CUDAContext::WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                                const std::vector<TensorTableEntry>& entries, Timeline& timeline) {
  while (!event_queue.empty()) {
    std::string name;
    cudaEvent_t event;
    std::tie(name, event) = event_queue.front();
    event_queue.pop();
    if (name != "") {
      timeline.ActivityStartAll(entries, name);
    }
    ErrorCheck("cudaEventSynchronize", cudaEventSynchronize(event));
    if (name != "") {
      timeline.ActivityEndAll(entries);
    }
    ErrorCheck("ReleaseCudaEvent", ReleaseCudaEvent(event));
  }
}

CUDAAllreduce::CUDAAllreduce(CUDAContext* context,
                             HorovodGlobalState* global_state)
    : AllreduceOp(global_state), cuda_context_(context) {}

bool CUDAAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void CUDAAllreduce::MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
                                       void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto& buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*) e.tensor->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset += e.tensor->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) entries.size()) {
        // Perform batched d2d memcpy
        batched_d2d_memcpy(d2d_params, count, cuda_context_->streams[global_state_->current_stream][first_entry.device]);
        count = 0;
      }
    }
    buffer_len = (size_t)offset;

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }

    buffer_len = (size_t) offset;
  }

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void CUDAAllreduce::MemcpyOutFusionBuffer(const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)(e.output->data());
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset += e.tensor->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int) entries.size()) {
        // Perform batched d2d memcpy
        batched_d2d_memcpy(d2d_params, count, cuda_context_->streams[global_state_->current_stream][first_entry.device]);
        count = 0;
      }
    }

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
      offset += e.tensor->size();
    }
  }
}

void CUDAAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[global_state_->current_stream][first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void CUDAAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                               const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync((void*) e.output->data(), buffer_data_at_offset,
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[global_state_->current_stream][first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void CUDAAllreduce::InitCUDA(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(first_entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[global_state_->current_stream][first_entry.device];
  if (stream == nullptr) {
    int greatest_priority;
    cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                              cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
  }
}

void CUDAAllreduce::InitCUDAQueue(const std::vector<TensorTableEntry>& entries, const Response& response) {
  event_queue_ = std::queue<std::pair<std::string, cudaEvent_t>>();
  stream_ = &cuda_context_->streams[global_state_->current_stream][entries[0].device];
  host_buffer_ = nullptr;

  if (global_state_->timeline.Initialized()) {
    cuda_context_->RecordEvent(event_queue_, QUEUE, *stream_);
  }
}

Status CUDAAllreduce::FinalizeCUDAQueue(const std::vector<TensorTableEntry>& entries) {
  // Use completion marker via event because it's faster than
  // blocking cudaStreamSynchronize() in this thread.
  cuda_context_->RecordEvent(event_queue_, "", *stream_);

  auto& first_entry = entries[0];
  void* host_buffer = host_buffer_;
  auto& event_queue = event_queue_;
  auto& timeline = global_state_->timeline;
  auto& cuda_context = cuda_context_;

  cuda_context_->finalizer_thread_pool.execute([entries, first_entry, host_buffer,
                                event_queue, &timeline, &cuda_context]() mutable {
    auto cuda_result = cudaSetDevice(first_entry.device);
    cuda_context->ErrorCheck("cudaSetDevice", cuda_result);

    cuda_context->WaitForEvents(event_queue, entries, timeline);
    if (host_buffer != nullptr) {
      free(host_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  });

  // Update current stream
  global_state_->current_stream = (global_state_->current_stream + 1) %
                                  global_state_->num_streams;

  return Status::InProgress();
}

} // namespace common
} // namespace horovod