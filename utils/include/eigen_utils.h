#ifndef TORC_EIGEN_UTILS_H
#define TORC_EIGEN_UTILS_H

#include "eigen3/Eigen/Dense"

namespace torc::utils {
    /**
     * @brief Converts a std vector to an Eigen column vector.
     * @tparam scalar_t the data type in the vectors
     * @param vec the std vector
     * @return the Eigen column vector
     */
    template <class scalar_t>
    Eigen::VectorX<scalar_t> StdToEigenVector(const std::vector<scalar_t>& vec) {
        Eigen::VectorX<scalar_t> eig_vec(vec.size());
        for (int i = 0; i < eig_vec.size(); i++) {
            eig_vec(i) = vec[i];
        }
        return eig_vec;
    }

    /**
     * @brief Converts an Eigen column vector to a std vector
     * @tparam scalar_t the data type in the vectors
     * @param vec the Eigen column vector
     * @return the std vector
     */
    template <class scalar_t>
    std::vector<scalar_t> EigenToStdVector(const Eigen::VectorX<scalar_t>& vec) {
        return std::vector<scalar_t>(vec.data(), vec.data() + vec.rows());
    }

    /**
     * @brief Converts an Eigen row vector to a std vector
     * @tparam scalar_t the data type in the vectors
     * @param vec the Eigen row vector
     * @return the std vector
     */
    // template <class scalar_t>
    // std::vector<scalar_t> EigenToStdVector(const Eigen::RowVectorX<scalar_t>& vec) {
    //     return std::vector<scalar_t>(vec.data(), vec.data() + vec.cols());
    // }
}

#endif
