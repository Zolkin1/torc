//
// Created by zolkin on 9/1/24.
//

#ifndef SIMPLE_TRAJECTORY_H
#define SIMPLE_TRAJECTORY_H
#include "cost_function.h"

namespace torc::mpc {
	class SimpleTrajectory {
	public:
		SimpleTrajectory(int size, int nodes);

		void InsertData(int node, const vectorx_t& data);

		void SetAllData(const vectorx_t& data);

		Eigen::VectorBlock<vectorx_t> GetNodeData(int node);

		// const Eigen::VectorBlock<vectorx_t> GetNodeData(int node) const;

		Eigen::VectorBlock<vectorx_t> operator[](int node);

		// const Eigen::VectorBlock<vectorx_t> operator[](int node) const;

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
