#include <eigen3/Eigen/Dense>
#include <cmath>
#include "linear_fn.h"
#include "quadratic_fn.h"
#include "autodiff_fn.h"
#include "explicit_fn.h"
#include "finite_diff_fn.h"

namespace test {
    template <typename scalar_t=double>
    std::vector<std::function<scalar_t(Eigen::VectorX<scalar_t>)>> functions {
            [](const Eigen::VectorX<scalar_t>& x) { return x.sum(); },                      // 1: sum
            [](const Eigen::VectorX<scalar_t>& x) { return x.dot(x); },                     // 2: dot product
            [](const Eigen::VectorX<scalar_t>& x) { return x.sum() * x.sum() * x.sum(); },  // 3: cube of sum
            [](const Eigen::VectorX<scalar_t>& x) { return x.prod(); },                     // 4: product of elements
            [](const Eigen::VectorX<scalar_t>& x) { return exp(3 * x.sum()); },             // 5: exp of sum
            [](const Eigen::VectorX<scalar_t>& x) { return sin(3 * x.sum()); },             // 6: sin of sum
            [](const Eigen::VectorX<scalar_t>& x) { return exp(sin(x.sum())); },            // 7: exp(sin(x^T x))
    };

    template <typename scalar_t=double>
    std::vector<std::function<Eigen::VectorX<scalar_t>(Eigen::VectorX<scalar_t>)>> gradients {
            [](const Eigen::VectorX<scalar_t>& x) { return Eigen::VectorX<scalar_t>::Ones(x.size()); }, // 1
            [](const Eigen::VectorX<scalar_t>& x) { return 2 * x; }, // 2
            [](const Eigen::VectorX<scalar_t>& x) { return 3 * (x.sum() * x.sum()) * Eigen::VectorX<scalar_t>::Ones(x.size()); }, // 3
            [](const Eigen::VectorX<scalar_t>& x) {
                Eigen::VectorX<scalar_t> result = Eigen::VectorX<scalar_t>::Zero(x.size());
                for (int i = 0; i < x.size(); ++i) {
                    if (x(i) != 0) {
                        result(i) = x.prod() / x(i);
                    }
                }
                return result;
            },  // 4
            [](const Eigen::VectorX<scalar_t>& x) { return 3 * exp(3 * x.sum()) * Eigen::VectorX<scalar_t>::Ones(x.size()); }, // 5
            [](const Eigen::VectorX<scalar_t>& x) { return 3 * cos(3 * x.sum()) * Eigen::VectorX<scalar_t>::Ones(x.size()); }, // 6
            [](const Eigen::VectorX<scalar_t>& x) { return exp(sin(x.sum())) * cos(x.sum()) * Eigen::VectorX<scalar_t>::Ones(x.size()); }  // 7
    };

    template <typename scalar_t=double>
    std::vector<std::function<Eigen::MatrixX<scalar_t>(Eigen::VectorX<scalar_t>)>> hessians {
            [](const Eigen::VectorX<scalar_t>& x) { return Eigen::MatrixX<scalar_t>::Zero(x.size(), x.size()); }, // 1
            [](const Eigen::VectorX<scalar_t>& x) { return Eigen::MatrixX<scalar_t>::Identity(x.size(), x.size()) * 2; }, // 2
            [](const Eigen::VectorX<scalar_t>& x) { return 6 * Eigen::MatrixX<scalar_t>::Ones(x.size(), x.size()) * x.sum(); }, // 3
            [](const Eigen::VectorX<scalar_t>& x) {
                Eigen::MatrixX<scalar_t> result = Eigen::MatrixX<scalar_t>::Zero(x.size(), x.size());
                scalar_t product = x.prod();
                for (int i = 0; i < x.size(); ++i) {
                    for (int j = 0; j < x.size(); ++j) {
                        if (i != j && x(i)*x(j) != 0) {
                            result(i, j) = product / (x(i) * x(j));
                        }
                    }
                }
                return result;
            }, // 4
            [](const Eigen::VectorX<scalar_t>& x) { return 9 * exp(3 * x.sum()) * Eigen::MatrixX<scalar_t>::Ones(x.size(), x.size()); }, // 5
            [](const Eigen::VectorX<scalar_t>& x) { return -9 * sin(3 * x.sum()) * Eigen::MatrixX<scalar_t>::Ones(x.size(), x.size()); }, // 6
            [](const Eigen::VectorX<scalar_t>& x) { return exp(sin(x.sum())) * (cos(x.sum()) * cos(x.sum()) - sin(x.sum())) * Eigen::MatrixX<scalar_t>::Ones(x.size(), x.size()); } // 7
    };
}