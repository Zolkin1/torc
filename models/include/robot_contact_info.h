//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_ROBOT_CONTACT_INFO_H
#define TORC_ROBOT_CONTACT_INFO_H

#include <vector>
#include <map>
#include <string>

namespace torc::models {
    enum ContactType {
        PointContact,
        PatchContact
    };

    struct Contact {
        ContactType type;
        bool state;

        /**
         * Create a contact with a given type and contact state
         * @param ctype
         * @param cstate
         */
        Contact(ContactType ctype, bool cstate) {
            type = ctype;
            state = cstate;
        }

        /**
         * Create a contact with a specific type, state defaults to false (out of contact)
         * @param ctype
         */
        Contact(ContactType ctype) {
            type = ctype;
            state = false;
        }
    };

    struct RobotContactInfo {
        std::map<std::string, Contact> contacts;

        [[nodiscard]] int GetNumContacts() const {
            int num_contacts = 0;
            for (const auto& contact : contacts) {
                if (contact.second.state) {
                    num_contacts++;
                }
            }

            return num_contacts;
        }
    };
} // torc::models


#endif //TORC_ROBOT_CONTACT_INFO_H
