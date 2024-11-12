//
// Created by zolkin on 7/26/24.
//
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "full_order_rigid_body.h"

#ifndef TORC_FULL_ORDER_TEST_FCNS_H
#define TORC_FULL_ORDER_TEST_FCNS_H

bool VectorEqualWithMargin(const torc::models::vectorx_t& v1, const torc::models::vectorx_t& v2, const double MARGIN) {
    using namespace torc::models;
    if (v1.size() != v2.size()) {
        return false;
    }

    for (int i = 0; i < v1.size(); i++) {
        if (std::abs(v1(i) - v2(i)) > MARGIN) {
            return false;
        }
    }

    return true;
}

void CheckDerivatives(torc::models::FullOrderRigidBody& model) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 9e-4;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        vectorx_t x_rand = model.GetRandomState();

        // Get random input
        vectorx_t input_rand;
        input_rand.setRandom(model.GetNumInputs());

        // Hold analytic derivatives
        matrixx_t A, B;
        A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
        B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetNumInputs());

        // Calculate analytic derivatives
        model.DynamicsDerivative(x_rand, input_rand, A, B);

        // Check wrt Configs
        vectorx_t xdot_test = model.GetDynamics(x_rand, input_rand);
        vectorx_t q_rand, v_rand;
        model.ParseState(x_rand, q_rand, v_rand);

        for (int i = 0; i < v_rand.size(); i++) {
            vectorx_t q_xd, v_xd;
            model.ParseState(x_rand, q_xd, v_xd);
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d dquat = Eigen::Vector3d::Zero();
                dquat(i - 3) += DELTA;

                q_xd.array().segment<4>(3) = (model.GetBaseOrientation(q_xd) * pinocchio::quaternion::exp3(dquat)).coeffs();
            } else {
                if (i >= 6) {
                    q_xd(i + 1) += DELTA;
                } else {
                    q_xd(i) += DELTA;
                }
            }
            vectorx_t x_d = model.BuildState(q_xd, v_xd);
            vectorx_t deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < deriv_d.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < v_rand.size(); i++) {
            vectorx_t x_d = x_rand;
            x_d(i + q_rand.size()) += DELTA;

            vectorx_t deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < deriv_d.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i + v_rand.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            vectorx_t deriv_d = model.GetDynamics(x_rand, input_d);

            for (int j = 0; j < deriv_d.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }
    }
}

void CheckContactDerivatives(torc::models::FullOrderRigidBody& model, const torc::models::RobotContactInfo& contact_info) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 1e-2;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        vectorx_t x_rand = model.GetRandomState();
        vectorx_t q, v;
        model.ParseState(x_rand, q, v);


        // Get random input
        vectorx_t input_rand;
        input_rand.setRandom(model.GetNumInputs());

        // Hold analytic derivatives
        matrixx_t A, B;
        A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
        B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetNumInputs());

        // Calculate analytic derivatives
        model.DynamicsDerivative(x_rand, input_rand, contact_info, A, B);

        // Check wrt Configs
        vectorx_t xdot_test = model.GetDynamics(x_rand, input_rand, contact_info);

        for (int i = 0; i < v.size(); i++) {
            vectorx_t q_rand, v_rand;
            model.ParseState(x_rand, q_rand, v_rand);
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                q_rand.array().segment<4>(3) =(model.GetBaseOrientation(q_rand) * pinocchio::quaternion::exp3(v)).coeffs();
            } else {
                if (i >= 6) {
                    q_rand(i + 1) += DELTA;
                } else {
                    q_rand(i) += DELTA;
                }
            }

            vectorx_t x_d = FullOrderRigidBody::BuildState(q_rand, v_rand);
            vectorx_t deriv_d = model.GetDynamics(x_d, input_rand, contact_info);

            for (int j = 0; j < deriv_d.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < v.size(); i++) {
            vectorx_t q_rand, v_rand;
            model.ParseState(x_rand, q_rand, v_rand);
            v_rand(i) += DELTA;

            vectorx_t x_d = FullOrderRigidBody::BuildState(q_rand, v_rand);

            vectorx_t deriv_d = model.GetDynamics(x_d, input_rand, contact_info);

            for (int j = 0; j < deriv_d.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i + v_rand.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t q_rand, v_rand;
            model.ParseState(x_rand, q_rand, v_rand);

            vectorx_t input_d = input_rand;
            input_d(i) += DELTA;
            vectorx_t deriv_d = model.GetDynamics(x_rand, input_d, contact_info);

            // for (int j = 0; j < deriv_d.size(); j++) {
            //     double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
            //     REQUIRE_THAT(fd - A(j, i + v_rand.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            // }

            for (int j = 0; j < v_rand.size(); j++) {
                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }
    }
}

void CheckInverseDynamicsDerivatives(torc::models::FullOrderRigidBody& model, const std::vector<std::string>& contact_names) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 1e-4;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
//    srand(Catch::getSeed());        // Set the srand seed manually
    srand(1);

    constexpr int NUM_CONFIGS = 1; // TODO: put back to 10
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get random state
        vectorx_t random_state = model.GetRandomState();
        // set normal config
//        random_state << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        random_state.head(7) << 0, 0, 0, 0.9988, 0, 0, 0.0500;

        // Get random acceleration
        vectorx_t random_acc(model.GetVelDim());
        random_acc.setRandom();
        random_acc = 10 * random_acc;

        // Get random external forces and a random number of contacts
        int num_contacts = 1; // rand() % contact_names.size();
        std::vector<ExternalForce> f_ext;
        for (int i = 0; i < num_contacts; i++) {
            vector3_t force = vector3_t::Zero();
            force.setRandom();
            force = force * 10;
            std::cout << "force: " << force.transpose() << std::endl;
            f_ext.emplace_back(contact_names.at(i), force);
        }

//        pinocchio::container::aligned_vector<pinocchio::Force> forces = model.ConvertExternalForcesToPin(f_ext);

        std::cout << "num contacts: " << num_contacts << std::endl;

        // Hold analytic derivatives
        matrixx_t dtau_dq, dtau_dv, dtau_da;
        dtau_dq.setZero(model.GetVelDim(), model.GetVelDim());
        dtau_dv.setZero(model.GetVelDim(), model.GetVelDim());
        dtau_da.setZero(model.GetVelDim(), model.GetVelDim());

        vectorx_t q_rand, v_rand;
        model.ParseState(random_state, q_rand, v_rand);

        // Calculate analytic derivatives
//        model.InverseDynamicsDerivative(q_rand, v_rand, random_acc, f_ext, dtau_dq, dtau_dv, dtau_da);

        // Check wrt Configs
//        vectorx_t tau_1 = model.InverseDynamics(random_state, random_acc, f_ext);

        std::cout << "configuration: " << q_rand << std::endl;

        matrixx_t force_deriv;
        force_deriv = model.ExternalForcesDerivativeWrtConfiguration(q_rand, f_ext);
        for (int i = 0; i < v_rand.size(); i++) {
            vectorx_t q_xd, v_xd;
            model.ParseState(random_state, q_xd, v_xd);
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d dquat = Eigen::Vector3d::Zero();
                dquat(i - 3) += DELTA;

                q_xd.array().segment<4>(3) = (model.GetBaseOrientation(q_xd) * pinocchio::quaternion::exp3(dquat)).coeffs();
            } else {
                if (i >= 6) {
                    q_xd(i + 1) += DELTA;
                } else {
                    q_xd(i) += DELTA;
                }
            }
//            vectorx_t x_d = torc::models::FullOrderRigidBody::BuildState(q_xd, v_xd);       // State w/ disturbance
//            vectorx_t tau_d = model.InverseDynamics(x_d, random_acc, f_ext);

            pinocchio::container::aligned_vector<pinocchio::Force> original_force = model.ConvertExternalForcesToPin(q_rand, f_ext);
            pinocchio::container::aligned_vector<pinocchio::Force> force_d = model.ConvertExternalForcesToPin(q_xd, f_ext);

//            std::cout << "original force: " << original_force.at(model.GetParentJointIdx(f_ext.at(0).frame_name)) << std::endl;
//            std::cout << "disturbed force: " << force_d.at(model.GetParentJointIdx(f_ext.at(0).frame_name)) << std::endl;

//            for (int j = 0; j < tau_d.size(); j++) {
//                // Check entire function
//                double fd = (tau_d(j) - tau_1(j)) / DELTA;
//                if (std::abs(fd - dtau_dq(j, i)) > FD_MARGIN) {
//                    std::cout << "fd: " << fd << ", analytic: " << dtau_dq(j, i) << " row: " << j << ", col: " << i
//                              << std::endl;
//                }
////                CHECK_THAT(fd - dtau_dq(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
//            }

            std::cout << "linear" << std::endl;
            // linear
            for (int j = 0; j < 3; j++) {
                // Check just force derivative
//                std::cout << "fd force linear" << (original_force.at(0).linear() - force_d.at(0).linear())/DELTA << std::endl;
                double fd = (original_force.at(model.GetParentJointIdx(f_ext.at(0).frame_name)).linear()(j)
                        - force_d.at(model.GetParentJointIdx(f_ext.at(0).frame_name)).linear()(j)) / DELTA;
                if (std::abs(fd - force_deriv(j, i)) > FD_MARGIN) {
                    std::cout << "fd: " << fd << ", analytic: " << force_deriv(j, i) << " row: " << j << ", col: " << i
                              << std::endl;
                }
            }

            std::cout << "angular" << std::endl;
            // angular
            for (int j = 0; j < 3; j++) {
                // Check just force derivative
//                std::cout << "fd force linear" << (original_force.at(0).linear() - force_d.at(0).linear())/DELTA << std::endl;
                double fd = (original_force.at(model.GetParentJointIdx(f_ext.at(0).frame_name)).angular()(j)
                        - force_d.at(model.GetParentJointIdx(f_ext.at(0).frame_name)).angular()(j)) / DELTA;
                if (std::abs(fd - force_deriv(j + 3, i)) > FD_MARGIN) {
                    std::cout << "fd: " << fd << ", analytic: " << force_deriv(j + 3, i) << " row: " << j + 3 << ", col: " << i
                              << std::endl;
                }
            }
        }

        std::cout << "analytic force deriv: " << std::endl << force_deriv << std::endl;

        // Now check wrt velocities
//        for (int i = 0; i < v_rand.size(); i++) {
//            vectorx_t x_d = x_rand;
//            x_d(i + q_rand.size()) += DELTA;
//
//            vectorx_t deriv_d = model.GetDynamics(x_d, input_rand);
//
//            for (int j = 0; j < deriv_d.size(); j++) {
//                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
//                REQUIRE_THAT(fd - A(j, i + v_rand.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
//            }
//        }
//
//        // Now check wrt inputs
//        for (int i = 0; i < input_rand.size(); i++) {
//            vectorx_t input_d = input_rand;
//
//            input_d(i) += DELTA;
//
//            vectorx_t deriv_d = model.GetDynamics(x_rand, input_d);
//
//            for (int j = 0; j < deriv_d.size(); j++) {
//                double fd = (deriv_d(j) - xdot_test(j)) / DELTA;
//                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
//            }
//        }
    }
}

void CheckImpulseDerivatives(torc::models::FullOrderRigidBody& model, const torc::models::RobotContactInfo& contact_info) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 9e-4;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        vectorx_t x_rand = model.GetRandomState();
        vectorx_t q_rand, v_rand;
        model.ParseState(x_rand, q_rand, v_rand);

        // Get random input
        vectorx_t input_rand;
        input_rand.setRandom(model.GetNumInputs());

        // Hold analytic derivatives
        matrixx_t A, B;
        A = matrixx_t::Zero(model.GetDerivativeDim(), model.GetDerivativeDim());
        B = matrixx_t::Zero(model.GetDerivativeDim(), model.GetNumInputs());

        // Calculate analytic derivatives
        model.ImpulseDerivative(x_rand, input_rand, contact_info, A, B);

//        std::cout << "A: \n" << A << std::endl;
        matrixx_t Afd = A;
        Afd.setZero();

        // Check wrt Configs
        vectorx_t xdot_test = model.GetImpulseDynamics(x_rand, input_rand, contact_info);
        vectorx_t q_xdot_test, v_xdot_test;
        model.ParseState(xdot_test, q_xdot_test, v_xdot_test);

        for (int i = 0; i < v_rand.size(); i++) {
            vectorx_t qd = q_rand;
            vectorx_t vd = v_rand;
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                qd.array().segment<4>(3) = (model.GetBaseOrientation(qd) * pinocchio::quaternion::exp3(v)).coeffs();
            } else {
                if (i >= 6) {
                    qd(i + 1) += DELTA;
                } else {
                    qd(i) += DELTA;
                }
            }
            vectorx_t x_d = FullOrderRigidBody::BuildState(qd, vd);

            vectorx_t imp_d = model.GetImpulseDynamics(x_d, input_rand, contact_info);
            vectorx_t q_imp, v_imp;
            model.ParseState(imp_d, q_imp, v_imp);

            // Configurations should be unchanged
            for (int j = 0; j < q_rand.size(); j++) {
                REQUIRE_THAT(q_imp(j) - q_xdot_test(j),
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < v_rand.size(); j++) {
                if (j == i) {
                    REQUIRE(A(j, i) == 1);
                    Afd(j, i) = 1;
                } else {
                    REQUIRE(A(j, i) == 0);
                }
            }

            for (int j = 0; j < v_rand.size(); j++) {
                double fd = (v_imp(j) - v_xdot_test(j)) / DELTA;
                Afd(j + v_rand.size(), i) = fd;
                REQUIRE_THAT(fd - A(j + v_rand.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < v_rand.size(); i++) {
            vectorx_t qd = q_rand;
            vectorx_t vd = v_rand;
            vd(i) += DELTA;
            vectorx_t x_d = FullOrderRigidBody::BuildState(qd, vd);

            vectorx_t imp_d = model.GetImpulseDynamics(x_d, input_rand, contact_info);
            vectorx_t q_imp, v_imp;
            model.ParseState(imp_d, q_imp, v_imp);

            // Configurations should be unchanged
            for (int j = 0; j < q_rand.size(); j++) {
                REQUIRE_THAT(q_imp(j) - q_xdot_test(j),
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < v_rand.size(); j++) {
                REQUIRE(A(j, i + v_rand.size()) == 0);
            }

            for (int j = 0; j < v_rand.size(); j++) {
                double fd = (v_imp(j) - v_xdot_test(j)) / DELTA;
                Afd(j + v_rand.size(), i + v_rand.size()) = fd;
                REQUIRE_THAT(fd - A(j + v_rand.size(), i + v_rand.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // std::cout << "Afd: \n" << Afd << std::endl;

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            vectorx_t imp_d = model.GetImpulseDynamics(x_rand, input_d, contact_info);
            vectorx_t q_imp, v_imp;
            model.ParseState(imp_d, q_imp, v_imp);

            for (int j = 0; j < v_rand.size(); j++) {
                double fd = (v_imp(j) - v_xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd - B(j + v_rand.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
                REQUIRE(B(j + v_rand.size(), i) == 0);
            }

            for (int j = 0; j < q_rand.size(); j++) {
                double fd = (q_imp(j) - q_xdot_test(j)) / DELTA;
                REQUIRE_THAT(fd,
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < v_rand.size(); j++) {
                REQUIRE(B(j,i) == 0);
            }
        }
    }
}
#endif //TORC_FULL_ORDER_TEST_FCNS_H
