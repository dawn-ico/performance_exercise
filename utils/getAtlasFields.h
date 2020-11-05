#pragma once

#include "atlas/field.h"

#include <string.h>
#include <tuple>

std::tuple<atlas::Field, atlasInterface::Field<dawn::float_type>>
MakeAtlasField(const std::string &name, int size, int ksize) {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real32(),
                       atlas::array::make_shape(size, ksize)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real64(),
                       atlas::array::make_shape(size, ksize)};
#else
#error DAWN_PRECISION is invalid
#endif
  return {field_F, atlas::array::make_view<dawn::float_type, 2>(field_F)};
};

std::tuple<atlas::Field, atlasInterface::VerticalField<dawn::float_type>>
MakeVerticalAtlasField(const std::string &name, int ksize) {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real32(),
                       atlas::array::make_shape(ksize)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real64(),
                       atlas::array::make_shape(ksize)};
#else
#error DAWN_PRECISION is invalid
#endif
  return {field_F, atlas::array::make_view<dawn::float_type, 1>(field_F)};
};

std::tuple<atlas::Field, atlasInterface::SparseDimension<dawn::float_type>>
MakeAtlasSparseField(const std::string &name, int size, int sparseSize,
                     int ksize) {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real32(),
                       atlas::array::make_shape(size, ksize, sparseSize)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
  atlas::Field field_F{name, atlas::array::DataType::real64(),
                       atlas::array::make_shape(size, ksize, sparseSize)};
#else
#error DAWN_PRECISION is invalid
#endif
  return {field_F, atlas::array::make_view<dawn::float_type, 3>(field_F)};
};