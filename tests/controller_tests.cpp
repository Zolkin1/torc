//
// Created by zolkin on 7/18/24.
//
#include <catch2/catch_test_macros.hpp>

#include "whole_body_qp_controller.h"

TEST_CASE("Basic WBC Tests", "[controllers][wbc]") {
    using namespace torc::controllers;
    using namespace torc::models;

    std::filesystem::path config_file = std::filesystem::current_path();
    config_file += "/test_data/wbc_controller_config.yaml";

    std::filesystem::path a1_urdf = std::filesystem::current_path();
    a1_urdf += "/test_data/test_a1.urdf";

    FullOrderRigidBody a1_model("a1", a1_urdf);

    WholeBodyQPController controller("test_controller", a1_model, config_file);

    torc::controllers::vectorx_t state = a1_model.GetRandomState();
    torc::controllers::vectorx_t target = 0.9*state + 0.1*a1_model.GetRandomState();
    torc::models::vectorx_t force_target(6);
    force_target << 0, 0, a1_model.GetMass()*4.5, 0, 0, a1_model.GetMass()*4.5;

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("FR_foot", Contact(ContactType::PointContact));
    contact_info.contacts.emplace("FL_foot", Contact(ContactType::PointContact, true));
    contact_info.contacts.emplace("RR_foot", Contact(ContactType::PointContact, true));
    contact_info.contacts.emplace("RL_foot", Contact(ContactType::PointContact));

    controller.ComputeControl(target, force_target, state, contact_info);
}