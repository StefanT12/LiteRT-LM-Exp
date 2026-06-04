// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_TOOLS_DEDUPE_FLATBUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_TOOLS_DEDUPE_FLATBUFFER_H_

#include <cstdint>

#include "absl/log/absl_check.h"  // from @com_google_absl  // IWYU pragma: keep
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers

namespace litert::lm {

// Deduplicates a serialized flatbuffer.
//
// This function uses the flatbuffer's schema to traverse the flatbuffer and
// update all duplicate data to point to the same location in the buffer.
//
// Note that this function supports most flatbuffer types, but not all. The
// following types are NOT supported, and will result in an error:
//   - Structs
//   - Arrays (not to be confused with vectors)
//   - Vectors of Unions
//   - Vector64
//   - RPC
//   - nested_flatbuffer and flexbuffer attributes
absl::StatusOr<flatbuffers::DetachedBuffer> DedupeFlatBuffer(
    const reflection::Schema& schema, const uint8_t* buffer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_TOOLS_DEDUPE_FLATBUFFER_H_
