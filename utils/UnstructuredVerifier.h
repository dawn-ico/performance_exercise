#pragma once

#include "atlas/mesh.h"
#include "atlas/mesh/HybridElements.h"
#include "interface/atlas_interface.hpp"

// Verifiyer class for atlas' ArrayViews, very close in behaviour to
// gridtools::dawn::verifier
class UnstructuredVerifier {
private:
  bool use_default_precision_ = false;
  double precision_;

  template <typename Value>
  void setDefaultPrecision() {
    if(std::is_same<Value, double>::value) {
      precision_ = 1e-10;
    } else if(std::is_same<Value, float>::value) {
      precision_ = 1e-6;
    } else if(std::is_same<Value, int>::value) {
      precision_ = 0;
    } else {
      assert(false);
    }
  }

  template <typename value_type>
  std::tuple<bool, value_type> compare_below_threshold(value_type expected, value_type actual,
                                                       value_type precision) const {
    bool outcome = false;
    value_type error = 0.;
    if(precision == 0) {
      outcome = expected == actual;
    } else if(std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3) {
      if(std::fabs(expected - actual) < precision) {
        outcome = true;
        error = std::fabs(expected - actual);
      }
    } else {
      if(std::fabs((expected - actual) / (precision * expected)) < 1.0) {
        outcome = true;
        error = std::fabs((expected - actual) / expected);
      }
    }
    return {outcome, error};
  }

public:
  UnstructuredVerifier() : use_default_precision_(true) {}
  UnstructuredVerifier(double precision) : use_default_precision_(false), precision_(precision) {}

  template <typename Value, int RANK>
  bool compareArrayView(const atlas::array::ArrayView<Value, RANK>& lhs,
                        const atlas::array::ArrayView<Value, RANK>& rhs, int max_erros = 10) {
    // it's far from trivial to compare two _general_ array views, since they
    // can have arbitrary rank, with dimensions unknown at compile time since we
    // currently only consider scalar fields in 3 spatial dimensions I decided
    // to force this to two
    //      NOTE: f(i,k) is rank 2, with i being the element index on a level,
    //      and k the given level NOTE: even atlas seems to limit the used
    //      dimensions to rank 4, which would be a 2D tensor on each element at
    //      each level
    //            (2 additional ranks to address the tensor)
    static_assert(RANK == 2, "");

    if(use_default_precision_) {
      setDefaultPrecision<Value>();
    }

    // first compare geometry
    if((lhs.shape(0) != rhs.shape(0)) || (lhs.shape(1) != rhs.shape(1))) {
      return false;
    }

    // then the values
    bool verified = true;
    for(int i = 0; i < lhs.shape(0); i++) {
      for(int k = 0; k < lhs.shape(1); k++) {
        Value valueLhs = lhs(i, k);
        Value valueRhs = rhs(i, k);
        auto [result, error] = compare_below_threshold(valueLhs, valueRhs, Value(precision_));
        if(!result) {
          if(--max_erros >= 0) {
            std::cerr << "( idx: " << i << " lvl: " << k << " ) : "
                      << "  error: " << error << " lhs " << valueLhs << " rhs " << valueRhs
                      << std::endl;
          }
          verified = false;
        }
      }
    }

    return verified;
  }

  template <typename Value, int RANK>
  bool compareArrayView(const std::vector<int>& indices, int k_size,
                        const atlas::array::ArrayView<Value, RANK>& lhs,
                        const atlas::array::ArrayView<Value, RANK>& rhs, int max_erros = 10) {
    // it's far from trivial to compare two _general_ array views, since they
    // can have arbitrary rank, with dimensions unknown at compile time since we
    // currently only consider scalar fields in 3 spatial dimensions I decided
    // to force this to two
    //      NOTE: f(i,k) is rank 2, with i being the element index on a level,
    //      and k the given level NOTE: even atlas seems to limit the used
    //      dimensions to rank 4, which would be a 2D tensor on each element at
    //      each level
    //            (2 additional ranks to address the tensor)
    static_assert(RANK == 2, "");

    if(use_default_precision_) {
      setDefaultPrecision<Value>();
    }

    // first compare geometry
    if((lhs.shape(0) != rhs.shape(0)) || (lhs.shape(1) != rhs.shape(1))) {
      return false;
    }

    // then the values
    bool verified = true;
    for(int level = 0; level < k_size; level++) {
      for(int idx : indices) {
        Value valueLhs = lhs(idx, level);
        Value valueRhs = rhs(idx, level);
        auto [result, error] = compare_below_threshold(valueLhs, valueRhs, Value(precision_));
        if(!result) {
          if(--max_erros >= 0) {
            std::cerr << "( idx: " << idx << " lvl: " << level << " ) : "
                      << "  error: " << error << " lhs " << valueLhs << " rhs " << valueRhs
                      << std::endl;
          }
          verified = false;
        }
      }
    }

    return verified;
  }

  template <typename Value, int RANK>
  static std::tuple<double, double, double>
  getErrorNorms(const std::vector<int>& indices, int k_size,
                const atlas::array::ArrayView<Value, RANK>& in,
                const atlas::array::ArrayView<Value, RANK>& ref) {

    // it's far from trivial to compare two _general_ array views, since they
    // can have arbitrary rank, with dimensions unknown at compile time since we
    // currently only consider scalar fields in 3 spatial dimensions I decided
    // to force this to two
    //      NOTE: f(i,k) is rank 2, with i being the element index on a level,
    //      and k the given level NOTE: even atlas seems to limit the used
    //      dimensions to rank 4, which would be a 2D tensor on each element at
    //      each level
    //            (2 additional ranks to address the tensor)
    static_assert(RANK == 2, "");

    // first compare geometry
    assert(in.shape(0) == ref.shape(0) && in.shape(1) == ref.shape(1));

    double Linf = 0.;
    double L1 = 0.;
    double L2 = 0.;
    int N = k_size * indices.size();
    for(int level = 0; level < k_size; level++) {
      for(int idx : indices) {
        double dif = in(idx, level) - ref(idx, level);
        Linf = fmax(fabs(dif), Linf);
        L1 += fabs(dif);
        L2 += dif * dif;
      }
    }
    L1 /= N;
    L2 = sqrt(L2) / sqrt(N);
    return {Linf, L1, L2};
  }
};
