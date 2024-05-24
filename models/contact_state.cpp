//
// Created by zolkin on 5/23/24.
//

#include "contact_state.h"

#include <utility>
#include <stdexcept>
#include <iostream>

namespace torc::models {
    ContactState::ContactState(const std::vector<int>& contact_frames)
        : frames_(contact_frames) {
        for (int i = 0; i < frames_.size(); i++) {
            contact_.push_back(false);
        }
    }

    ContactState::ContactState(const std::vector<int>& contact_frames, const std::vector<bool>& contact_state)
        : frames_(contact_frames), contact_(contact_state) {}

    void ContactState::ChangeContact(int contact_frame) {
        for (int i : frames_) {
            if (i == contact_frame) {
                contact_.at(i) = !contact_.at(i);
                return;
            }
        }

        throw std::runtime_error("Provided contact frame is not valid.");
    }

    void ContactState::UpdateContact(int contact_frame, bool contact_state) {
        for (int i : frames_) {
            if (i == contact_frame) {
                contact_.at(i) = contact_state;
                return;
            }
        }

        throw std::runtime_error("Provided contact frame is not valid.");
    }

    void ContactState::UpdateAllContacts(const std::vector<int>& contact_frames,
                                         const std::vector<bool>& contact_state) {
        for (int i : frames_) {
            bool matching_frame = false;
            for (int j : contact_frames) {
                if (i == j) {
                    contact_.at(i) = contact_state.at(j);
                    matching_frame = true;
                }
            }
            if (!matching_frame) {
                throw std::runtime_error("At least one provided frame does not match.");
            }
        }
    }

    const std::vector<int>& ContactState::GetFrames() const {
        return frames_;
    }

    const std::vector<bool>& ContactState::GetContacts() const {
        return contact_;
    }

    bool ContactState::GetContact(int contact_frame) const {
        for (int i : frames_) {
            if (i == contact_frame) {
                return contact_.at(i);
            }
        }

        throw std::runtime_error("Contact frame provided is not valid.");
    }

    int ContactState::GetNumContacts() const {
        int num_contacts = 0;
        for (bool i : contact_) {
            if (i) {
                num_contacts++;
            }
        }

        return num_contacts;
    }

} // torc::models