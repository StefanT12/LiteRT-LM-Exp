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

#include "tools/dedupe_flatbuffer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/table.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

class FlatBufferDeduper {
 public:
  static absl::StatusOr<flatbuffers::DetachedBuffer> Dedupe(
      const reflection::Schema& schema, const uint8_t* buffer) {
    FlatBufferDeduper deduper(schema);
    ASSIGN_OR_RETURN(uoffset_t root_offset,
                     deduper.DedupeTable(schema.root_table(),
                                         flatbuffers::GetAnyRoot(buffer)));
    deduper.Finish(root_offset);
    return deduper.builder_.Release();
  }

 private:
  using uoffset_t = flatbuffers::uoffset_t;
  using voffset_t = flatbuffers::voffset_t;

  explicit FlatBufferDeduper(const reflection::Schema& schema)
      : schema_(schema) {}

  absl::StatusOr<uoffset_t> DedupeTable(const reflection::Object* object,
                                        const flatbuffers::Table* table) {
    if (object->is_struct()) {
      return absl::InvalidArgumentError("Flatbuffer structs are not supported");
    }

    // Snapshot the original offset so we can revert the builder if a
    // duplicate is found.
    uoffset_t original_offset = builder_.GetSize();

    // Tables cannot be nested in flatbuffers, so we need to dedupe nested
    // fields first, then build the table once we know the offsets of the nested
    // fields.
    absl::flat_hash_map<voffset_t, uoffset_t> nested_fields;
    for (voffset_t i = object->fields()->size(); i-- > 0;) {
      const reflection::Field* field = object->fields()->Get(i);
      if (!table->CheckField(field->offset())) {
        continue;
      }
      const reflection::Type* type = field->type();
      if (type->base_type() == reflection::Obj) {
        ASSIGN_OR_RETURN(nested_fields[field->offset()],
                         DedupeTable(schema_.objects()->Get(type->index()),
                                     flatbuffers::GetFieldT(*table, *field)));

      } else if (type->base_type() == reflection::Vector) {
        ASSIGN_OR_RETURN(
            nested_fields[field->offset()],
            DedupeVector(field, flatbuffers::GetFieldAnyV(*table, *field)));

      } else if (type->base_type() == reflection::String) {
        ASSIGN_OR_RETURN(nested_fields[field->offset()],
                         DedupeString(flatbuffers::GetFieldS(*table, *field)));

      } else if (type->base_type() == reflection::Union) {
        ASSIGN_OR_RETURN(nested_fields[field->offset()],
                         DedupeTable(&flatbuffers::GetUnionType(
                                         schema_, *object, *field, *table),
                                     flatbuffers::GetFieldT(*table, *field)));
      }
    }

    uoffset_t table_start = builder_.StartTable();
    // Iterate in reverse order because flatbuffers are built backwards.
    // In theory iterating from largest to smallest element would lead to
    // tighter packing, but this works well enough for now.
    for (voffset_t i = object->fields()->size(); i-- > 0;) {
      const reflection::Field* field = object->fields()->Get(i);
      if (!table->CheckField(field->offset())) {
        continue;
      }

      // clang-format off
      const reflection::Type* type = field->type();
      switch (type->base_type()) {
        #define ADD_SCALAR(SUF, T, DEF)                       \
          builder_.AddElement<T>(field->offset(),             \
              flatbuffers::GetField##SUF<T>(*table, *field),  \
              static_cast<T>(field->DEF()));                  \
          break
        case reflection::UType:   ADD_SCALAR(I, int8_t, default_integer);
        case reflection::Byte:    ADD_SCALAR(I, int8_t, default_integer);
        case reflection::Bool:    ADD_SCALAR(I, uint8_t, default_integer);
        case reflection::UByte:   ADD_SCALAR(I, uint8_t, default_integer);
        case reflection::Short:   ADD_SCALAR(I, int16_t, default_integer);
        case reflection::UShort:  ADD_SCALAR(I, uint16_t, default_integer);
        case reflection::Int:     ADD_SCALAR(I, int32_t, default_integer);
        case reflection::UInt:    ADD_SCALAR(I, uint32_t, default_integer);
        case reflection::Long:    ADD_SCALAR(I, int64_t, default_integer);
        case reflection::ULong:   ADD_SCALAR(I, uint64_t, default_integer);
        case reflection::Float:   ADD_SCALAR(F, float, default_real);
        case reflection::Double:  ADD_SCALAR(F, double, default_real);
        case reflection::Union:
        case reflection::String:
        case reflection::Vector:
        case reflection::Obj:
          builder_.AddOffset(
              field->offset(),
              flatbuffers::Offset<>(nested_fields.at(field->offset())));
          break;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Field '", field->name()->c_str(),
                           "' has unsupported type: ", type->base_type()));
          break;
        #undef ADD_SCALAR
      }
      // clang-format on
    }
    uoffset_t table_end = builder_.EndTable(table_start);

    const flatbuffers::Table* new_table = flatbuffers::GetTemporaryPointer(
        builder_, flatbuffers::Offset<flatbuffers::Table>(table_end));
    for (uoffset_t other_offset : table_cache_[object]) {
      const flatbuffers::Table* other_table = flatbuffers::GetTemporaryPointer(
          builder_, flatbuffers::Offset<flatbuffers::Table>(other_offset));
      if (AreTablesEqual(object, new_table, other_table)) {
        // PopBytes will remove all bytes added to the builder in this function,
        // which aren't needed because we found a duplicate.
        builder_.PopBytes(builder_.GetSize() - original_offset);
        return other_offset;
      }
    }
    table_cache_[object].push_back(table_end);
    return table_end;
  }

  bool AreTablesEqual(const reflection::Object* object,
                      const flatbuffers::Table* table1,
                      const flatbuffers::Table* table2) {
    for (voffset_t i = object->fields()->size(); i-- > 0;) {
      const reflection::Field* field = object->fields()->Get(i);
      // Check that the field is set in both tables.
      if (table1->CheckField(field->offset()) !=
          table2->CheckField(field->offset())) {
        return false;
      }
      if (!table1->CheckField(field->offset())) {
        continue;
      }
      const reflection::Type* type = field->type();
      // clang-format off
      switch (type->base_type()) {
        #define COMPARE_SCALAR(T)                           \
          if (flatbuffers::GetFieldI<T>(*table1, *field) != \
              flatbuffers::GetFieldI<T>(*table2, *field)) { \
            return false;                                   \
          }                                                 \
          break
        case reflection::UType:   COMPARE_SCALAR(int8_t);
        case reflection::Byte:    COMPARE_SCALAR(int8_t);
        case reflection::Bool:    COMPARE_SCALAR(uint8_t);
        case reflection::UByte:   COMPARE_SCALAR(uint8_t);
        case reflection::Short:   COMPARE_SCALAR(int16_t);
        case reflection::UShort:  COMPARE_SCALAR(uint16_t);
        case reflection::Int:     COMPARE_SCALAR(int32_t);
        case reflection::UInt:    COMPARE_SCALAR(uint32_t);
        case reflection::Long:    COMPARE_SCALAR(int64_t);
        case reflection::ULong:   COMPARE_SCALAR(uint64_t);
        case reflection::Float: {
          float value1 = flatbuffers::GetFieldF<float>(*table1, *field);
          float value2 = flatbuffers::GetFieldF<float>(*table2, *field);
          if (std::abs(value1 - value2) >
              std::numeric_limits<float>::epsilon()) {
            return false;
          }
          break;
        }
        case reflection::Double: {
          double value1 = flatbuffers::GetFieldF<double>(*table1, *field);
          double value2 = flatbuffers::GetFieldF<double>(*table2, *field);
          if (std::abs(value1 - value2) >
              std::numeric_limits<double>::epsilon()) {
            return false;
          }
          break;
        }
        case reflection::Union:
        case reflection::String:
        case reflection::Vector:
        case reflection::Obj:
          // Non-scalar types can be compared by pointer because they they
          // will have been deduplicated already.
          if (table1->GetPointer<const void*>(field->offset()) !=
              table2->GetPointer<const void*>(field->offset())) {
            return false;
          }
          break;
        default:
          // Treat unsupported types as non-equal. DedupeTable will fail on it.
          return false;
        #undef COMPARE_SCALAR
      }
      // clang-format on
    }
    return true;
  }

  absl::StatusOr<uoffset_t> DedupeVector(
      const reflection::Field* field, const flatbuffers::VectorOfAny* vector) {
    reflection::BaseType element_type = field->type()->element();
    size_t elem_size = flatbuffers::GetTypeSize(element_type);
    size_t alignment = elem_size;
    if (field->attributes() != nullptr) {
      if (auto* force_align = field->attributes()->LookupByKey("force_align")) {
        int align;
        if (absl::SimpleAtoi(force_align->value()->c_str(), &align) &&
            align > 1) {
          alignment = align;
        }
      }
    }

    // Snapshot the original offset so we can revert the builder if a
    // duplicate is found.
    uoffset_t original_offset = builder_.GetSize();

    // If this is a vector of non-scalars, dedupe the contents first so the
    // offsets can be added to the vector.
    std::vector<uoffset_t> offsets(vector->size());
    for (size_t i = vector->size(); i-- > 0;) {
      if (element_type == reflection::String) {
        ASSIGN_OR_RETURN(
            offsets[i],
            DedupeString(
                flatbuffers::GetAnyVectorElemPointer<const flatbuffers::String>(
                    vector, i)));

      } else if (element_type == reflection::Obj) {
        ASSIGN_OR_RETURN(
            offsets[i],
            DedupeTable(
                schema_.objects()->Get(field->type()->index()),
                flatbuffers::GetAnyVectorElemPointer<const flatbuffers::Table>(
                    vector, i)));
      }
    }

    builder_.StartVector(vector->size(), elem_size, alignment);
    for (size_t i = vector->size(); i-- > 0;) {
      if (flatbuffers::IsInteger(element_type)) {
        int64_t n = flatbuffers::GetAnyVectorElemI(vector, element_type, i);
        builder_.Align(elem_size);
        // Note: this only works on little-endian systems.
        builder_.PushBytes(reinterpret_cast<const uint8_t*>(&n), elem_size);

      } else if (element_type == reflection::Float) {
        builder_.PushElement<float>(
            flatbuffers::GetAnyVectorElemF(vector, element_type, i));

      } else if (element_type == reflection::Double) {
        builder_.PushElement<double>(
            flatbuffers::GetAnyVectorElemF(vector, element_type, i));

      } else if (element_type == reflection::String ||
                 element_type == reflection::Obj) {
        builder_.PushElement<uoffset_t>(builder_.ReferTo(offsets[i]));

      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Vector '", field->name()->c_str(),
                         "' has an unsupported element type: ", element_type));
      }
    }
    uoffset_t new_offset = builder_.EndVector(vector->size());

    auto* new_vector = flatbuffers::GetTemporaryPointer(
        builder_, flatbuffers::Offset<flatbuffers::VectorOfAny>(new_offset));
    for (uoffset_t other_offset : vector_cache_[field->type()->element()]) {
      auto* other = flatbuffers::GetTemporaryPointer(
          builder_,
          flatbuffers::Offset<flatbuffers::VectorOfAny>(other_offset));
      if (AreVectorsEqual(field->type()->element(), new_vector, other)) {
        // PopBytes will remove all bytes added to the builder in this function,
        // which aren't needed because we found a duplicate.
        builder_.PopBytes(new_offset - original_offset);
        return other_offset;
      }
    }
    vector_cache_[field->type()->element()].push_back(new_offset);
    return new_offset;
  }

  bool AreVectorsEqual(const reflection::BaseType element_type,
                       const flatbuffers::VectorOfAny* vec1,
                       const flatbuffers::VectorOfAny* vec2) {
    if (vec1->size() != vec2->size()) {
      return false;
    }
    for (uoffset_t i = 0; i < vec1->size(); i++) {
      if (flatbuffers::IsInteger(element_type)) {
        if (flatbuffers::GetAnyVectorElemI(vec1, element_type, i) !=
            flatbuffers::GetAnyVectorElemI(vec2, element_type, i)) {
          return false;
        }

      } else if (flatbuffers::IsFloat(element_type)) {
        double diff =
            std::abs(flatbuffers::GetAnyVectorElemF(vec1, element_type, i) -
                     flatbuffers::GetAnyVectorElemF(vec2, element_type, i));
        if (element_type == reflection::Float &&
            diff > std::numeric_limits<float>::epsilon()) {
          return false;
        } else if (element_type == reflection::Double &&
                   diff > std::numeric_limits<double>::epsilon()) {
          return false;
        }

      } else if (element_type == reflection::String ||
                 element_type == reflection::Obj) {
        // Non-scalar types can be compared by pointer because they they
        // will have been deduplicated already.
        if (flatbuffers::GetAnyVectorElemPointer<const void>(vec1, i) !=
            flatbuffers::GetAnyVectorElemPointer<const void>(vec2, i)) {
          return false;
        }

      } else {
        // Treat unsupported types as non-equal. DedupeVector will fail on it.
        return false;
      }
    }
    return true;
  }

  absl::StatusOr<uoffset_t> DedupeString(const flatbuffers::String* str) {
    auto str_view = str->string_view();
    auto [it, inserted] = string_cache_.try_emplace(str_view);
    if (inserted) {
      it->second = builder_.CreateString(str).o;
    }
    return it->second;
  }

  void Finish(uoffset_t root_offset) {
    const char* file_ident = nullptr;
    if (schema_.file_ident() && strlen(schema_.file_ident()->c_str()) ==
                                    flatbuffers::kFileIdentifierLength) {
      file_ident = schema_.file_ident()->c_str();
    }
    builder_.Finish(flatbuffers::Offset<>(root_offset), file_ident);
  }

  flatbuffers::FlatBufferBuilder builder_;
  const reflection::Schema& schema_;

  absl::flat_hash_map<const reflection::Object*, std::vector<uoffset_t>>
      table_cache_;
  absl::flat_hash_map<reflection::BaseType, std::vector<uoffset_t>>
      vector_cache_;
  absl::flat_hash_map<std::string_view, uoffset_t> string_cache_;
};

}  // namespace

absl::StatusOr<flatbuffers::DetachedBuffer> DedupeFlatBuffer(
    const reflection::Schema& schema, const uint8_t* buffer) {
  return FlatBufferDeduper::Dedupe(schema, buffer);
}

}  // namespace litert::lm
