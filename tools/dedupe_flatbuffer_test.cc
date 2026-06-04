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

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/detached_buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/minireflect.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tools/test_flatbuffer_generated.h"
#include "tools/test_flatbuffer_reflection_data.h"

namespace litert::lm {
namespace {

using ::flatbuffers::Offset;
using ::testing::Eq;
using ::testing::Lt;

flatbuffers::DetachedBuffer Dedupe(const flatbuffers::DetachedBuffer& buffer) {
  const reflection::Schema* schema =
      reflection::GetSchema(test_flatbuffer_reflection_data_create()->data);
  absl::StatusOr<flatbuffers::DetachedBuffer> deduped_buffer =
      DedupeFlatBuffer(*schema, buffer.data());
  ABSL_CHECK_OK(deduped_buffer);
  return *std::move(deduped_buffer);
}

std::string ToString(const flatbuffers::DetachedBuffer& buffer) {
  return flatbuffers::FlatBufferToString(buffer.data(), TestTableTypeTable());
}

TEST(DedupeFlatbufferTest, Empty) {
  flatbuffers::FlatBufferBuilder fbb;
  auto table = CreateTestTable(fbb);
  fbb.Finish(table);
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, Scalars) {
  flatbuffers::FlatBufferBuilder fbb;
  TestTableBuilder table_builder(fbb);
  table_builder.add_i1(true);
  table_builder.add_i8(2);
  table_builder.add_u8(3);
  table_builder.add_i32(4);
  table_builder.add_u32(5);
  table_builder.add_i64(6);
  table_builder.add_u64(7);
  table_builder.add_f32(8.0);
  table_builder.add_f64(9.0);
  table_builder.add_enum_b(EnumI8_B);
  table_builder.add_enum_i32(EnumI32_D);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, String) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::String> str = fbb.CreateString("abc");
  TestTableBuilder table_builder(fbb);
  table_builder.add_str(str);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, Table) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<TestTable> nested_table = CreateTestTable(fbb, /*i1=*/true);
  TestTableBuilder table_builder(fbb);
  table_builder.add_tab(nested_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, Union) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<flatbuffers::String>>> vec_str =
      fbb.CreateVectorOfStrings({"abc"});
  SimpleTable1Builder simple_table_builder(fbb);
  simple_table_builder.add_vec_str(vec_str);
  Offset<SimpleTable1> simple_table = simple_table_builder.Finish();

  TestTableBuilder table_builder(fbb);
  table_builder.add_unn_type(TableUnion_SimpleTable1);
  table_builder.add_unn(simple_table.Union());
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, ScalarVector) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<int>> vec_i32 = fbb.CreateVector<int>({1, 2, 3});
  Offset<flatbuffers::Vector<float>> vec_f32 =
      fbb.CreateVector<float>({1.0, 2.0, 3.0});
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_i32(vec_i32);
  table_builder.add_vec_f32(vec_f32);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, StringVector) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<flatbuffers::String>>> vec_str =
      fbb.CreateVectorOfStrings({"abc", "def"});
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_str(vec_str);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, TableVector) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<TestTable>>> vec_table = fbb.CreateVector({
      CreateTestTable(fbb, /*i1=*/false, /*i8=*/3),
      CreateTestTable(fbb, /*i1=*/false, /*i8=*/2, /*u8=*/3),
  });
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_table(vec_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  // EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, AlignedVector) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<unsigned char>> vec_aligned =
      fbb.CreateVector<unsigned char>({1, 2, 3});
  TestTableBuilder table_builder(fbb);
  table_builder.add_i64(1);
  table_builder.add_vec_aligned(vec_aligned);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Eq(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, DedupeStringsInVector) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<flatbuffers::String>>> vec_str =
      fbb.CreateVectorOfStrings({"abc", "def", "abc"});
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_str(vec_str);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Lt(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, DedupeStringsInTable) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::String> str1 = fbb.CreateString("abc");
  TestTableBuilder nested_table_builder(fbb);
  nested_table_builder.add_str(str1);
  Offset<TestTable> nested_table = nested_table_builder.Finish();

  Offset<flatbuffers::String> str2 = fbb.CreateString("abc");
  TestTableBuilder table_builder(fbb);
  table_builder.add_str(str2);
  table_builder.add_tab(nested_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Lt(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, DedupeTables) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<TestTable>>> vec_table = fbb.CreateVector({
      CreateTestTable(fbb, /*i1=*/false, /*i8=*/2),
      CreateTestTable(fbb, /*i1=*/false, /*i8=*/2),
  });
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_table(vec_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Lt(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, DedupeVectors) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<Offset<flatbuffers::String>>> vec_str1 =
      fbb.CreateVectorOfStrings({"abc", "def"});
  TestTableBuilder nested_table_builder1(fbb);
  nested_table_builder1.add_i8(8);
  nested_table_builder1.add_vec_str(vec_str1);
  Offset<TestTable> nested_table1 = nested_table_builder1.Finish();

  Offset<flatbuffers::Vector<Offset<flatbuffers::String>>> vec_str2 =
      fbb.CreateVectorOfStrings({"abc", "def"});
  TestTableBuilder nested_table_builder2(fbb);
  nested_table_builder2.add_vec_str(vec_str2);
  Offset<TestTable> nested_table2 = nested_table_builder2.Finish();

  Offset<flatbuffers::Vector<Offset<TestTable>>> vec_table =
      fbb.CreateVector({nested_table1, nested_table2});
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_table(vec_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Lt(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

TEST(DedupeFlatbufferTest, FuzzyDedupeFloatVectors) {
  flatbuffers::FlatBufferBuilder fbb;
  Offset<flatbuffers::Vector<float>> vec1 = fbb.CreateVector({1.0f, 2.0f});
  TestTableBuilder nested_table_builder1(fbb);
  nested_table_builder1.add_vec_f32(vec1);
  Offset<TestTable> nested_table1 = nested_table_builder1.Finish();

  Offset<flatbuffers::Vector<float>> vec2 =
      fbb.CreateVector({1.00000001f, 2.00000001f});
  TestTableBuilder nested_table_builder2(fbb);
  nested_table_builder2.add_vec_f32(vec2);
  Offset<TestTable> nested_table2 = nested_table_builder2.Finish();

  Offset<flatbuffers::Vector<Offset<TestTable>>> vec_table =
      fbb.CreateVector({nested_table1, nested_table2});
  TestTableBuilder table_builder(fbb);
  table_builder.add_vec_table(vec_table);
  fbb.Finish(table_builder.Finish());
  flatbuffers::DetachedBuffer buffer = fbb.Release();

  flatbuffers::DetachedBuffer deduped_buffer = Dedupe(buffer);
  EXPECT_THAT(deduped_buffer.size(), Lt(buffer.size()));
  EXPECT_THAT(ToString(deduped_buffer), Eq(ToString(buffer)));
}

}  // namespace
}  // namespace litert::lm
