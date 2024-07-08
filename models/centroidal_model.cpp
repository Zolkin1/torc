#include "centroidal_model.h"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hxx>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>

namespace torc::models {
    Centroid::Centroid(const std::string &name,
                                     const std::filesystem::path &urdf,
                                     const std::vector<std::string>& contact_frames,
                                     const std::vector<std::string>& unactuated_joints)
        :PinocchioModel(name, urdf),
        n_contacts_(contact_frames.size()),
        n_actuated_(pin_model_.nv - BASE_DOF - unactuated_joints.size()){
        for (const auto& frame_name : contact_frames) {
            const size_t frame_id = pin_model_.getFrameId(frame_name);
            if (frame_id == -1) {
                throw std::runtime_error("Frame does not exist!");
            }
            contact_frames_idxs_.emplace_back(frame_id);
        }

        unactuated_joint_idxs_.emplace(0);   // Universe joint is never actuated
        unactuated_joint_idxs_.emplace(1);   // Base joint is never actuated

        for (const std::string& joint_name : unactuated_joints) {
            for (int i = 0; i < pin_model_.njoints; i++) {
                if (joint_name == pin_model_.names.at(i)) {
                    unactuated_joint_idxs_.emplace(i);
                    break;
                }
            }
        }
    }

    vectorx_t Centroid::GetDynamics(const vectorx_t &state,
                                    const vectorx_t& input)
    {
        vectorx_t q, hcom, dqj;
        ParseState(state, hcom, q);
        std::vector<vector3_t> forces;
        ParseInput(input, forces, dqj);
        // update all joint velocities, except base joint, compute dq_joints

        pinocchio::crba(pin_model_, *pin_data_, q);
        pinocchio::computeCentroidalMap(pin_model_, *pin_data_, q);
        const matrixx_t &cmm = pin_data_->Ag;
        const matrixx_t A_b = cmm.leftCols(BASE_DOF);
        const matrixx_t A_j = cmm.rightCols(cmm.cols() - BASE_DOF);
        const vectorx_t dqb = A_b.inverse() * (hcom - A_j * dqj);

        // Join together dqb and dqj
        vectorx_t dq(dqb.size() + dqj.size());
        dq << dqb, dqj;

        // compute CoM position, CMM, all frame positions, rotational inertia
        pinocchio::computeCentroidalMomentum(pin_model_, *pin_data_, q, dq); // need to add dqb
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);

        // determine dh_com by aggregating all the forces
        const vectorx_t com_pos = pin_data_->com[0];
        vector3_t force_sum = {0, 0, -9.81*mass_};        // gravity is always present
        vector3_t torque_sum = vector3_t::Zero();                     // gravity provides no torque
        // iterate through the forces
        for (int i=0; i<n_contacts_; i++) {
            // assume that all the forces are expressed in the CoM frame
            const vector3_t& force = forces.at(i);
            force_sum += force;
            vector3_t r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos;
            torque_sum += r.cross(force);
        }
        vectorx_t dh_com(force_sum.size() + torque_sum.size());
        dh_com << force_sum, torque_sum;

        return BuildState(dq, dh_com);
    }


    void Centroid::DynamicsDerivative(const vectorx_t &state,
                                      const vectorx_t &input, matrixx_t &A,
                                      matrixx_t &B) {
        const matrixx_t iden3 = matrixx_t::Identity(LINEAR_DIM, FORCE_DIM);

        vectorx_t q, hcom, dqj;
        ParseState(state, hcom, q);
        std::vector<vector3_t> forces;
        ParseInput(input, forces, dqj);

        forwardKinematics(pin_model_, *pin_data_, q);
        centerOfMass(pin_model_, *pin_data_, q, true);

        const matrixx_t A_g = computeCentroidalMap(pin_model_, *pin_data_, q);
        const matrixx_t A_b_inv = A_g.leftCols(BASE_DOF).inverse();
        const matrixx_t A_j = A_g.rightCols(pin_data_->Ag.cols() - BASE_DOF);
        const vectorx_t dqb = A_b_inv * (hcom - A_j * dqj);
        const matrixx_t &Jcom = pin_data_->Jcom;
        const vectorx_t com_pos = pin_data_->com[0];

        vectorx_t dq(dqj.size() + dqb.size());
        dq << dqb, dqj;

        computeCentroidalMomentum(pin_model_, *pin_data_, q, dq);

        // calculate ddx_dx (A)
        A = matrixx_t::Zero(COM_DOF + pin_model_.nv, COM_DOF + pin_model_.nv);
        A.topRightCorner(COM_DOF, COM_DOF) = A_b_inv;

        matrixx_t dhtau_dq(ANGULAR_DIM, pin_model_.nv);
        pinocchio::impl::computeJointJacobians(pin_model_, *pin_data_, q);
        // allocate outside of the loop to prevent reinitialization
        matrixx_t J_frame = matrixx_t::Zero(COM_DOF, pin_model_.nv);
        // iterate through all the contact forces so we don't have to recalculate
        // frame jacobians every time
        for (int i = 0; i < n_contacts_; i++) {
          const vector3_t &force = forces.at(i);
          J_frame =
              getFrameJacobian(pin_model_, *pin_data_, contact_frames_idxs_.at(i),
                               pinocchio::WORLD);
          const matrixx_t dr_dq = J_frame.topRows(LINEAR_DIM) - Jcom;
          // calculate the partial derivatives' effect on the angular momentum
          for (int j = 0; j < pin_model_.nv; j++) {
            dhtau_dq.col(j) += vectorx_t(vector3_t(dr_dq.col(j)).cross(force));
          }
          A.bottomLeftCorner(ANGULAR_DIM, pin_model_.nv) = dhtau_dq;
        }

        // calculate ddx_du (B)
        B = matrixx_t::Zero(COM_DOF + pin_model_.nv, input.size());
        B.topRightCorner(BASE_DOF, dqj.size()) = -A_b_inv * A_j;;
        B.rightCols(n_actuated_).middleRows(BASE_DOF, n_actuated_) =
            matrixx_t::Identity(n_actuated_, n_actuated_);
        // iterate through the forces
        for (int i = 0; i < n_contacts_; i++) {
            B.block(0, i * FORCE_DIM, LINEAR_DIM, FORCE_DIM) = iden3; // linear
            vector3_t r = pin_data_->oMi[contact_frames_idxs_.at(i)].translation() - com_pos; // rotational
            matrixx_t dhtau_dfi = matrixx_t::Zero(ANGULAR_DIM, FORCE_DIM);
            for (int j = 0; j < FORCE_DIM; j++) {
                dhtau_dfi.col(j) = r.cross(vector3_t(iden3.col(j)));
            }
            B.bottomRows(ANGULAR_DIM).middleCols(i*FORCE_DIM, FORCE_DIM) = dhtau_dfi;
        }
    }

    vectorx_t Centroid::BuildState(const vectorx_t &q, const vectorx_t &hcom) {
      vectorx_t x(q.size() + hcom.size());
      x << q, hcom;
      return x;
    }
    vectorx_t Centroid::BuildStateDerivative(const vectorx_t &v,
                                             const vectorx_t &dhcom) {
        return BuildState(v, dhcom);
    }

    void Centroid::ParseState(const vectorx_t &state, vectorx_t &hcom,
                              vectorx_t &q) const {
      q = state.topRows(pin_model_.nq);
      hcom = state.bottomRows(COM_DOF);
    }

    void Centroid::ParseStateDerivative(const vectorx_t &dstate,
                                        vectorx_t &dhcom, vectorx_t &v) const {
      v = dstate.topRows(pin_model_.nq);
      dhcom = dstate.bottomRows(COM_DOF);
    }

    quat_t Centroid::ParseBaseOrientation(const vectorx_t &q) {
        return quat_t(q.segment<4>(3));
    }

    void Centroid::ParseInput(const vectorx_t &input,
                              std::vector<vector3_t> &forces,
                              vectorx_t &vj) const {
        forces.clear();
        for (int i=0; i<n_contacts_; i++) {
            forces.emplace_back(input.segment(i*FORCE_DIM, FORCE_DIM));
        }
        vj = input.bottomRows(pin_model_.nv - VBASE_DOF);
    }

    vectorx_t Centroid::GetRandomState() const {
        vectorx_t q_rand(pin_model_.nq);
        q_rand << vectorx_t::Random(LINEAR_DIM), Eigen::Quaterniond::UnitRandom().coeffs(), vectorx_t::Random(n_actuated_);
        vectorx_t x_rand(GetConfigDim() + COM_DOF);
        x_rand << q_rand, vectorx_t::Random(COM_DOF);
        return x_rand;
    }

    vectorx_t Centroid::GetRandomInput() const {
        return vectorx_t::Random(GetInputDim());
    }

    int Centroid::GetDerivativeDim() const {
        return pin_model_.nv + COM_DOF;
    }

    int Centroid::GetInputDim() const {
        return n_contacts_ * FORCE_DIM + n_actuated_;
    }
}
