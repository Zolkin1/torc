//
// Created by zolkin on 9/1/24.
//

#ifndef SIMPLE_TRAJECTORY_H
#define SIMPLE_TRAJECTORY_H

#include <Eigen/Core>

namespace torc::mpc {
	using vectorx_t = Eigen::VectorXd;
	using vector3_t = Eigen::Vector3d;

	/**
	* @brief A simple trajectory with no semantic meaning.
	* Held as a single vector for cache friendlyness.
	*/
	class SimpleTrajectory {
	public:
		SimpleTrajectory(int size, int nodes);

		void InsertData(int node, const vectorx_t& data);

		void SetAllData(const vectorx_t& data);

		Eigen::VectorBlock<vectorx_t> GetNodeData(int node);

		Eigen::VectorBlock<const vectorx_t> GetNodeData(int node) const;

		Eigen::VectorBlock<vectorx_t> operator[](int node);

		Eigen::VectorBlock<const vectorx_t> operator[](int node) const;

		[[nodiscard]] int GetNumNodes() const;

		[[nodiscard]] int GetSize() const;

		void SetSizes(int size, int nodes);

	protected:
		int GetStartIdx(int node) const;
	private:
		int size_;
		int nodes_;
		vectorx_t data_;
	};
}


#endif //SIMPLE_TRAJECTORY_H
