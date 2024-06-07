#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "rigid_body.h"

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

void CheckDerivatives(torc::models::RigidBody& model) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 9e-4;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        RobotState x_rand = model.GetRandomState();

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
        RobotStateDerivative xdot_test = model.GetDynamics(x_rand, input_rand);
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                x_d.q.array().segment<4>(3) =
                        static_cast<Eigen::Vector4d>((x_d.Quat() * pinocchio::quaternion::exp3(v)).coeffs());
            } else {
                if (i >= 6) {
                    x_d.q(i + 1) += DELTA;
                } else {
                    x_d.q(i) += DELTA;
                }
            }
            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;

            x_d.v(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_rand, input_d);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - B(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }
    }
}

void CheckContactDerivatives(torc::models::RigidBody& model, const torc::models::RobotContactInfo& contact_info) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 1e-2;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        RobotState x_rand = model.GetRandomState();

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
        RobotStateDerivative xdot_test = model.GetDynamics(x_rand, input_rand, contact_info);
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                x_d.q.array().segment<4>(3) =
                        static_cast<Eigen::Vector4d>((x_d.Quat() * pinocchio::quaternion::exp3(v)).coeffs());
            } else {
                if (i >= 6) {
                    x_d.q(i + 1) += DELTA;
                } else {
                    x_d.q(i) += DELTA;
                }
            }
            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand, contact_info);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;

            x_d.v(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_d, input_rand, contact_info);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - A(j, i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            RobotStateDerivative deriv_d = model.GetDynamics(x_rand, input_d, contact_info);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - B(j, i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (deriv_d.a(j) - xdot_test.a(j)) / DELTA;
                REQUIRE_THAT(fd - B(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }
    }
}

void CheckImpulseDerivatives(torc::models::RigidBody& model, const torc::models::RobotContactInfo& contact_info) {
    using namespace torc::models;
    constexpr double FD_MARGIN = 9e-4;
    constexpr double DELTA = 1e-8;

    // Checking derivatives with finite differences
    srand(Catch::getSeed());        // Set the srand seed manually

    constexpr int NUM_CONFIGS = 10;
    for (int k = 0; k < NUM_CONFIGS; k++) {
        // Get a random configuration
        RobotState x_rand = model.GetRandomState();

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
        RobotState xdot_test = model.GetImpulseDynamics(x_rand, input_rand, contact_info);
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;
            if (i >= 3 && i < 6) {
                // Need to take a step in lie algebra space
                Eigen::Vector3d v = Eigen::Vector3d::Zero();
                v(i - 3) += DELTA;

                x_d.q.array().segment<4>(3) =
                        static_cast<Eigen::Vector4d>((x_d.Quat() * pinocchio::quaternion::exp3(v)).coeffs());
            } else {
                if (i >= 6) {
                    x_d.q(i + 1) += DELTA;
                } else {
                    x_d.q(i) += DELTA;
                }
            }
            RobotState imp_d = model.GetImpulseDynamics(x_d, input_rand, contact_info);

            // Configurations should be unchanged
            for (int j = 0; j < x_rand.q.size(); j++) {
                REQUIRE_THAT(imp_d.q(j) - xdot_test.q(j),
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                if (j == i) {
                    REQUIRE(A(j, i) == 1);
                    Afd(j, i) = 1;
                } else {
                    REQUIRE(A(j, i) == 0);
                }
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (imp_d.v(j) - xdot_test.v(j)) / DELTA;
                Afd(j + x_rand.v.size(), i) = fd;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

        // Now check wrt velocities
        for (int i = 0; i < x_rand.v.size(); i++) {
            RobotState x_d = x_rand;

            x_d.v(i) += DELTA;

            RobotState imp_d = model.GetImpulseDynamics(x_d, input_rand, contact_info);

            // Configurations should be unchanged
            for (int j = 0; j < x_rand.q.size(); j++) {
                REQUIRE_THAT(imp_d.q(j) - xdot_test.q(j),
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                REQUIRE(A(j, i + x_rand.v.size()) == 0);
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (imp_d.v(j) - xdot_test.v(j)) / DELTA;
                Afd(j + x_rand.v.size(), i + x_rand.v.size()) = fd;
                REQUIRE_THAT(fd - A(j + x_rand.v.size(), i + x_rand.v.size()), Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }
        }

//        std::cout << "Afd: \n" << Afd << std::endl;

        // Now check wrt inputs
        for (int i = 0; i < input_rand.size(); i++) {
            vectorx_t input_d = input_rand;

            input_d(i) += DELTA;

            RobotState imp_d = model.GetImpulseDynamics(x_rand, input_d, contact_info);

            for (int j = 0; j < x_rand.v.size(); j++) {
                double fd = (imp_d.v(j) - xdot_test.v(j)) / DELTA;
                REQUIRE_THAT(fd - B(j + x_rand.v.size(), i), Catch::Matchers::WithinAbs(0, FD_MARGIN));
                REQUIRE(B(j + x_rand.v.size(), i) == 0);
            }

            for (int j = 0; j < x_rand.q.size(); j++) {
                double fd = (imp_d.q(j) - xdot_test.q(j)) / DELTA;
                REQUIRE_THAT(imp_d.q(j) - xdot_test.q(j),
                             Catch::Matchers::WithinAbs(0, FD_MARGIN));
            }

            for (int j = 0; j < x_rand.v.size(); j++) {
                REQUIRE(B(j,i) == 0);
            }
        }
    }
}

TEST_CASE("Quadruped", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";
    const std::string bad_urdf = "fake_urdf.urdf";

    constexpr double MARGIN = 1e-6;

    // Check that a bad urdf throws an error
    REQUIRE_THROWS_AS(RigidBody(pin_model_name, bad_urdf), std::runtime_error);

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("FR_foot", Contact(ContactType::PointContact));
    contact_info.contacts.emplace("FL_foot", Contact(ContactType::PointContact, false));
    contact_info.contacts.emplace("RR_foot", Contact(ContactType::PointContact, false));
    contact_info.contacts.emplace("RL_foot", Contact(ContactType::PointContact));

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    RigidBody a1_model(pin_model_name, a1_urdf);
    REQUIRE(a1_model.GetName() == pin_model_name);

    constexpr int INPUT_SIZE = 12;
    constexpr int CONFIG_SIZE = 19;
    constexpr int VEL_SIZE = 18;
    constexpr int STATE_SIZE = 37;
    constexpr int DERIV_SIZE = 36;
    constexpr int JOINT_SIZE = 14;
    constexpr int FRAME_SIZE = 47;

    REQUIRE(a1_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(a1_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(a1_model.GetVelDim() == VEL_SIZE);
    REQUIRE(a1_model.GetStateDim() == STATE_SIZE);
    REQUIRE(a1_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(a1_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(a1_model.GetNumFrames() == FRAME_SIZE);
    REQUIRE(a1_model.GetSystemType() == HybridSystemImpulse);


    // ----------------------------------------------- //
    // ------------------- Dynamics ------------------ //
    // ----------------------------------------------- //
    RobotState x(a1_model.GetConfigDim(), a1_model.GetVelDim());
    const vectorx_t input = vectorx_t::Zero(INPUT_SIZE);

    // No contacts
    RobotStateDerivative xdot = a1_model.GetDynamics(x, input);
    RobotStateDerivative true_deriv(VEL_SIZE);
    true_deriv.a(2) = -9.81;
    REQUIRE(VectorEqualWithMargin(xdot.v, true_deriv.v, MARGIN));
    REQUIRE(VectorEqualWithMargin(xdot.a, true_deriv.a, MARGIN));

    // With contacts
    xdot = a1_model.GetDynamics(x, input, contact_info);
    REQUIRE(VectorEqualWithMargin(xdot.v, true_deriv.v, MARGIN));

    // Impulse dynamics
    x = a1_model.GetImpulseDynamics(x, input, contact_info);

    // ----------------------------------------------- //
    // ------------- Dynamics Derivatives ------------ //
    // ----------------------------------------------- //
    matrixx_t A, B, Aimp, Bimp;
    A = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetDerivativeDim());
    B = matrixx_t::Zero(a1_model.GetDerivativeDim(), INPUT_SIZE);
    Aimp = A;
    Bimp = B;

    bool dyn_deriv = GENERATE(true, false);
    if (dyn_deriv) {
        // No contact dynamics derivatives
        a1_model.DynamicsDerivative(x, input, A, B);
    }

    bool contact_deriv = GENERATE(true, false);
    if (contact_deriv) {
        // Contact dynamics derivatives
        a1_model.DynamicsDerivative(x, input, contact_info, A, B);
    }

    bool impulse_deriv = GENERATE(true, false);
    if (impulse_deriv) {
        // Impulse derivative
        a1_model.ImpulseDerivative(x, input, contact_info, Aimp, Bimp);
    }

    bool deriv_sec = GENERATE(true, false);
    if (deriv_sec) {
        SECTION("Dynamics Derivatives") {
            // TODOl: Only works when ImpulseDerivative is called above this
            CheckDerivatives(a1_model);
        }
    }

    bool contact_sec = GENERATE(true, false);
    if (contact_sec) {
        SECTION("Contact Dynamics Derivatives") {
            CheckContactDerivatives(a1_model, contact_info);
        }
    }

    bool impulse_sec = GENERATE(true, false);
    if (impulse_sec) {
        SECTION("Impulse Dynamics Derivatives") {
            CheckImpulseDerivatives(a1_model, contact_info);

            contact_info.contacts.at("FR_foot").state = true;
            CheckImpulseDerivatives(a1_model, contact_info);
        }
    }
}

TEST_CASE("Quadruped Benchmarks", "[model][pinocchio][benchmarks]") {
    using namespace torc::models;

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    RigidBody a1_model("a1", a1_urdf);

    matrixx_t A, B, Aimp, Bimp;
    A = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetDerivativeDim());
    B = matrixx_t::Zero(a1_model.GetDerivativeDim(), a1_model.GetNumInputs());

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("FR_foot", Contact(ContactType::PointContact));
    contact_info.contacts.emplace("FL_foot", Contact(ContactType::PointContact, true));
    contact_info.contacts.emplace("RR_foot", Contact(ContactType::PointContact, true));
    contact_info.contacts.emplace("RL_foot", Contact(ContactType::PointContact));

    RobotState x_rand = a1_model.GetRandomState();
    vectorx_t input_rand;
    input_rand.setRandom(a1_model.GetNumInputs());
    // Benchmarks
    BENCHMARK("Dynamics Derivatives") {
        return a1_model.DynamicsDerivative(x_rand, input_rand, A, B);
    };

    BENCHMARK("Contact Dynamics Derivatives") {
        return a1_model.DynamicsDerivative(x_rand, input_rand, contact_info, A, B);
    };

    BENCHMARK("Impulse Dynamics Derivatives") {
        return a1_model.ImpulseDerivative(x_rand, input_rand, contact_info, A, B);
    };

    BENCHMARK("Dynamics") {
        return a1_model.GetDynamics(x_rand, input_rand);
    };

    BENCHMARK("Contact Dynamics") {
        return a1_model.GetDynamics(x_rand, input_rand, contact_info);
    };

    BENCHMARK("Impulse Dynamics") {
        return a1_model.GetImpulseDynamics(x_rand, input_rand, contact_info);
    };
}

TEST_CASE("Double Integrator", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path integrator_urdf = std::filesystem::current_path();
    integrator_urdf += "/test_data/integrator.urdf";

    // No contact_info
    RobotContactInfo contact_info;

    RigidBody int_model(pin_model_name, integrator_urdf);

    REQUIRE(int_model.GetMass() == 1);
}

TEST_CASE("Hopper", "[model][pinocchio]") {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path hopper_urdf = std::filesystem::current_path();
    hopper_urdf += "/test_data/hopper.urdf";

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("foot", Contact(PointContact, true));

    RigidBody hopper_model(pin_model_name, hopper_urdf);

    int constexpr INPUT_SIZE = 4;
    constexpr int CONFIG_SIZE = 11;
    constexpr int VEL_SIZE = 10;
    constexpr int STATE_SIZE = 21;
    constexpr int DERIV_SIZE = 20;
    constexpr int JOINT_SIZE = 6;

    RobotState x(hopper_model.GetConfigDim(), hopper_model.GetVelDim());
    const vectorx_t input = vectorx_t::Zero(INPUT_SIZE);
    RobotStateDerivative xdot = hopper_model.GetDynamics(x, input, contact_info);

    REQUIRE(hopper_model.GetNumInputs() == INPUT_SIZE);
    REQUIRE(hopper_model.GetConfigDim() == CONFIG_SIZE);
    REQUIRE(hopper_model.GetVelDim() == VEL_SIZE);
    REQUIRE(hopper_model.GetStateDim() == STATE_SIZE);
    REQUIRE(hopper_model.GetDerivativeDim() == DERIV_SIZE);
    REQUIRE(hopper_model.GetNumJoints() == JOINT_SIZE);
    REQUIRE(hopper_model.GetSystemType() == HybridSystemImpulse);

    SECTION("Dynamics Derivatives") {
        CheckDerivatives(hopper_model);
    }

    SECTION("Contact Dynamics Derivatives") {
        CheckContactDerivatives(hopper_model, contact_info);

        contact_info.contacts.at("foot").state = false;
        CheckImpulseDerivatives(hopper_model, contact_info);
    }

    SECTION("Impulse Dynamics Derivatives") {
        CheckImpulseDerivatives(hopper_model, contact_info);

        contact_info.contacts.at("foot").state = false;
        CheckImpulseDerivatives(hopper_model, contact_info);
    }
}