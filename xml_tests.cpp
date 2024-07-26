//
// Created by zolkin on 7/24/24.
//

#include "full_order_rigid_body.h"

int main() {
    using namespace torc::models;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path achilles_mjcf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles.xml";

    RobotContactInfo contact_info;
    contact_info.contacts.emplace("foot", Contact(PointContact, true));

    FullOrderRigidBody achilles_model(pin_model_name, achilles_mjcf, false);
}