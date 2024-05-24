//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_CONTACT_STATE_H
#define TORC_CONTACT_STATE_H

#include <vector>

namespace torc::models {
    class ContactState {
    public:
        ContactState(const std::vector<int>& contact_frames);

        ContactState(const std::vector<int>& contact_frames, const std::vector<bool>& contact_state);

        void ChangeContact(int contact_frame);

        void UpdateContact(int contact_frame, bool contact_state);

        void UpdateAllContacts(const std::vector<int>& contact_frames, const std::vector<bool>& contact_state);

        const std::vector<int>& GetFrames() const;

        const std::vector<bool>& GetContacts() const;

        bool GetContact(int contact_frame) const;

        int GetNumContacts() const;

    protected:
    private:
        std::vector<int> frames_;
        std::vector<bool> contact_;
    };
} // torc::models


#endif //TORC_CONTACT_STATE_H
