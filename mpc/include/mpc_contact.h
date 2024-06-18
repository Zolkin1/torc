//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_MPC_CONTACT_H
#define TORC_MPC_CONTACT_H

#include <memory>

#include "contact_model.h"
#include "mpc_base.h"

namespace torc::mpc {
    class MPCContact : public MPCBase {
    public:
        MPCContact(const models::ContactModel& model);

        // Want to provide this function for all the generic data types and any of the special ones
        //  that I provide an interface for.
//        void GetQPData()

    protected:
       std::unique_ptr<models::ContactModel> model_;

    private:
    };
} // torc::mpc


#endif //TORC_MPC_CONTACT_H
