// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A tool to optimize a flatbuffer model by deduping buffers and stripping
// debug strings.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "third_party/gloop/base/init_google.h"
#include "third_party/gloop/strings/human_readable.h"
#include "third_party/mediapipe/framework/port/file_helpers.h"
#include "tools/dedupe_flatbuffer.h"
#include "runtime/util/memory_mapped_file.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"  // from @org_tensorflow
#include "tensorflow/compiler/mlir/lite/schema/schema_reflection_data.h"  // from @org_tensorflow
#include "tflite/model_builder.h"  // from @litert

ABSL_FLAG(std::string, model_path, "",
          "Path to the input TFLite flatbuffer model.");
ABSL_FLAG(std::string, output_path, "",
          "Path to the output TFLite flatbuffer model.");

ABSL_FLAG(bool, dedupe_buffers, true, "Removes duplicate buffers.");
ABSL_FLAG(bool, dedupe_model, true, "Removes duplicate model stuff.");
ABSL_FLAG(bool, strip_debug_strings, true, "Removes debug strings.");

namespace {

constexpr size_t kDefaultOffset = 128;
constexpr size_t kAlignmentSize = 128;

struct OriginalBuffer {
  int idx;
  const char* data;
  size_t size;
};

template <typename T>
std::vector<T> ToStdVector(const flatbuffers::Vector<T>* v) {
  std::vector<T> result;
  result.reserve(v->size());
  for (int i = 0; i < v->size(); ++i) {
    result.push_back(v->Get(i));
  }
  return result;
}

void PadToAlignment(std::string& str) {
  if (str.size() % kAlignmentSize != 0) {
    str += std::string(kAlignmentSize - (str.size() % kAlignmentSize), '\0');
  }
}

bool IsModelSupported(const tflite::ModelT& model) {
  for (const auto& buffer : model.buffers) {
    if (!buffer->data.empty()) {
      ABSL_LOG(INFO) << "Buffers must be in extended mode.";
      return false;
    }
  }
  for (const auto& subgraph : model.subgraphs) {
    for (const auto& oper : subgraph->operators) {
      if (oper->large_custom_options_offset > 0) {
        ABSL_LOG(INFO) << "Large custom options offsets are not supported.";
        return false;
      }
    }
  }
  return true;
}

std::string GetBufferContent(const char* model_data,
                             const tflite::ModelT& model) {
  size_t min_offset = std::numeric_limits<size_t>::max();
  size_t max_offset = 0;
  for (const auto& buffer : model.buffers) {
    if (buffer->size == 0) {
      continue;
    }
    min_offset = std::min(min_offset, buffer->offset);
    max_offset = std::max(max_offset, buffer->offset + buffer->size);
  }
  return std::string(model_data + min_offset, max_offset - min_offset);
}

std::string DedupeBuffers(const char* model_data, tflite::ModelT& model) {
  // Sort the buffer declarations by size, descending, to increase the chance
  // of finding duplicate buffers.
  ABSL_LOG(INFO) << "Deduping buffers";
  std::vector<OriginalBuffer> old_buffers;
  old_buffers.reserve(model.buffers.size());
  for (int i = 0; i < model.buffers.size(); i++) {
    const auto& buffer = model.buffers[i];
    if (buffer->offset == 0) {
      // Non-embedded buffers shouldn't be here in the first place.
      old_buffers.push_back({.idx = i, .data = model_data, .size = 0});
    } else {
      old_buffers.push_back({.idx = i,
                             .data = model_data + buffer->offset,
                             .size = buffer->size});
    }
  }
  std::sort(old_buffers.begin(), old_buffers.end(),
            [](const OriginalBuffer& a, const OriginalBuffer& b) {
              return a.size > b.size;
            });

  // Clear the buffers and add the empty sentinel buffer.
  model.buffers.clear();
  model.buffers.push_back(std::make_unique<tflite::BufferT>(
      tflite::BufferT{.offset = 0, .size = 0}));

  std::string new_buffer_content;
  std::vector<size_t> buffer_index_mapping;
  buffer_index_mapping.resize(old_buffers.size());
  for (const auto& old_buffer : old_buffers) {
    // See if the buffer is already in the new buffer content.
    // This is *extremely* inefficient: it scans all previous buffers to check
    // for a copy of the current buffer. This may need to be optimized if used
    // with larger models.
    off_t new_offset = -1;
    for (size_t i = 0; i + old_buffer.size < new_buffer_content.size();
         i += kAlignmentSize) {
      if (memcmp(new_buffer_content.data() + i, old_buffer.data,
                 old_buffer.size) == 0) {
        new_offset = i;
        break;
      }
    }
    // If not, append the buffer to the new buffer content.
    if (new_offset == -1) {
      PadToAlignment(new_buffer_content);
      new_offset = new_buffer_content.size();
      new_buffer_content.append(std::string(old_buffer.data, old_buffer.size));
    }

    // Add the buffer declaration to the model if it doesn't already exist.
    const auto existing_buffer_it = std::find_if(
        model.buffers.begin(), model.buffers.end(), [&](const auto& buffer) {
          return buffer->offset == new_offset &&
                 buffer->size == old_buffer.size;
        });
    if (existing_buffer_it != model.buffers.end()) {
      buffer_index_mapping[old_buffer.idx] =
          existing_buffer_it - model.buffers.begin();
    } else {
      buffer_index_mapping[old_buffer.idx] = model.buffers.size();
      model.buffers.push_back(std::make_unique<tflite::BufferT>(
          tflite::BufferT{.offset = static_cast<uint64_t>(new_offset),
                          .size = old_buffer.size}));
    }
  }
  // This is a hack to make sure no buffers have an offset of 0, which would
  // result in the offset not being encoded in the flatbuffer because 0 is the
  // default value. This is removed later when we update the offets.
  for (const auto& buffer : model.buffers) {
    buffer->offset += kDefaultOffset;
  }

  // Update all buffer references in the model.
  for (auto& subgraph : model.subgraphs) {
    for (auto& tensor : subgraph->tensors) {
      tensor->buffer = buffer_index_mapping[tensor->buffer];
    }
  }
  std::map<std::string, uint32_t> buffer_names;
  for (const auto& metadata : model.metadata) {
    if (metadata->buffer >= buffer_index_mapping.size()) {
      ABSL_LOG(INFO) << "Invalid buffer index: " << metadata->name << ", "
                     << metadata->buffer
                     << " >= " << buffer_index_mapping.size();
      buffer_names[metadata->name] = metadata->buffer;
    } else {
      if (buffer_names.contains(metadata->name)) {
        ABSL_LOG(INFO) << "Duplicate metadata name: " << metadata->name;
      }
      buffer_names[metadata->name] = buffer_index_mapping[metadata->buffer];
    }
  }
  model.metadata.clear();
  for (const auto& buffer_name : buffer_names) {
    model.metadata.push_back(
        std::make_unique<tflite::MetadataT>(tflite::MetadataT{
            .name = buffer_name.first, .buffer = buffer_name.second}));
  }
  model.metadata_buffer.clear();

  ABSL_LOG(INFO) << "Buffer count: " << old_buffers.size() << " -> "
                 << model.buffers.size();
  return new_buffer_content;
}

void AdjustBufferOffsets(std::string& serialized_model) {
  size_t offset = serialized_model.size() - kDefaultOffset;
  ABSL_CHECK(offset % kAlignmentSize == 0);
  ABSL_LOG(INFO) << "Adjusting buffer offsets";
  auto* mutable_buffers =
      tflite::GetMutableModel(serialized_model.data())->mutable_buffers();
  for (int i = 0; i < mutable_buffers->size(); ++i) {
    auto* buffer = mutable_buffers->GetMutableObject(i);
    ABSL_CHECK(buffer->mutate_offset(buffer->offset() + offset));
  }
}

void StripDebugStrings(tflite::ModelT& model) {
  ABSL_LOG(INFO) << "Stripping debug strings";
  for (auto& subgraph : model.subgraphs) {
    subgraph->name.clear();
    for (auto& tensor : subgraph->tensors) {
      tensor->name.clear();
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, /*remove_flags=*/true);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  auto file = ::litert::lm::MemoryMappedFile::Create(model_path);
  ABSL_CHECK_OK(file);

  const char* model_data = reinterpret_cast<const char*>(file.value()->data());
  std::unique_ptr<tflite::FlatBufferModel> flat_buffer_model =
      tflite::FlatBufferModel::VerifyAndBuildFromBuffer(model_data,
                                                        file.value()->length());
  if (!flat_buffer_model) {
    ABSL_LOG(ERROR) << "Failed to build flatbuffer model";
    return 1;
  }
  auto model = std::make_unique<tflite::ModelT>();
  flat_buffer_model->GetModel()->UnPackTo(model.get());

  ABSL_CHECK(IsModelSupported(*model));

  std::string new_buffer_content = absl::GetFlag(FLAGS_dedupe_buffers)
                                       ? DedupeBuffers(model_data, *model)
                                       : GetBufferContent(model_data, *model);
  if (absl::GetFlag(FLAGS_strip_debug_strings)) {
    StripDebugStrings(*model);
  }

  flatbuffers::FlatBufferBuilder builder;
  auto new_model = tflite::CreateModel(builder, model.get());
  tflite::FinishModelBuffer(builder, new_model);
  std::string serialized_model(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());

  if (absl::GetFlag(FLAGS_dedupe_model)) {
    ABSL_CHECK_EQ(tflite::schema_reflection_data_size(), 1);
    const ::reflection::Schema* schema =
        ::reflection::GetSchema(tflite::schema_reflection_data_create()->data);

    absl::StatusOr<flatbuffers::DetachedBuffer> deduped_buffer =
        litert::lm::DedupeFlatBuffer(
            *schema, reinterpret_cast<const uint8_t*>(serialized_model.data()));
    ABSL_CHECK_OK(deduped_buffer);
    serialized_model =
        std::string(reinterpret_cast<const char*>(deduped_buffer->data()),
                    deduped_buffer->size());
  }

  // The size of the metadata flatbuffer, which is the first part of the
  // serialized model. Reducing this size is important since it directly
  // translates to reduced peak memory usage when loading an obfuscated model.
  const size_t flatbuffer_size = serialized_model.size();

  PadToAlignment(serialized_model);
  if (absl::GetFlag(FLAGS_dedupe_buffers)) {
    AdjustBufferOffsets(serialized_model);
  }
  serialized_model += new_buffer_content;

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  if (output_path.empty()) {
    output_path = model_path + ".optimized";
  }
  ABSL_CHECK_OK(mediapipe::file::SetContents(output_path, serialized_model));
  ABSL_LOG(INFO) << "Wrote extended model to: " << output_path;
  ABSL_LOG(INFO) << "Model size (metadata flatbuffer + buffers): "
                 << strings::HumanReadableNumBytes::ToString(
                        serialized_model.size());
  ABSL_LOG(INFO) << "Metadata size: "
                 << strings::HumanReadableNumBytes::ToString(flatbuffer_size);
  ABSL_LOG(INFO) << "Buffer size: "
                 << strings::HumanReadableNumBytes::ToString(
                        new_buffer_content.size());
  return 0;
}
