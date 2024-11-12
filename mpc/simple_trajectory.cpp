//
// Created by zolkin on 9/1/24.
//

#include "simple_trajectory.h"

namespace torc::mpc {
	SimpleTrajectory::SimpleTrajectory(int size, int nodes)
		: size_(size), nodes_(nodes) {
		data_.resize(size_*nodes_);
	}

	void SimpleTrajectory::InsertData(int node, const vectorx_t& data) {
		if (data.size() != size_) {
			throw std::runtime_error("[Simple Trajectory] Data size does not match! Expected " + std::to_string(size_) + ", got: " + std::to_string(data.size()));
		}

		data_.segment(GetStartIdx(node), size_) = data;
	}

	void SimpleTrajectory::SetAllData(const vectorx_t& data) {
		if (data.size() != size_) {
			throw std::runtime_error("[Simple Trajectory] Data size does not match! Expected " + std::to_string(size_) + ", got: " + std::to_string(data.size()));
		}

		for (int node = 0; node < nodes_; node++) {
			data_.segment(GetStartIdx(node), size_) = data;
		}
	}


	Eigen::VectorBlock<vectorx_t> SimpleTrajectory::GetNodeData(int node) {
		return data_.segment(GetStartIdx(node), size_);
	}

	Eigen::VectorBlock<vectorx_t> SimpleTrajectory::operator[](int node) {
		return GetNodeData(node);
	}

	void SimpleTrajectory::SetSizes(int size, int nodes) {
		size_ = size;
		nodes_ = nodes;
		data_.resize(size_*nodes_);
	}



	// const Eigen::VectorBlock<vectorx_t> SimpleTrajectory::GetNodeData(int node) const {
	// 	return data_.segment(GetStartIdx(node), size_);
	// }
	//
	// const Eigen::VectorBlock<vectorx_t> SimpleTrajectory::operator[](int node) const {
	// 	return GetNodeData(node);
	// }

	int SimpleTrajectory::GetNumNodes() const {
		return nodes_;
	}

	int SimpleTrajectory::GetSize() const {
		return size_;
	}

	int SimpleTrajectory::GetStartIdx(int node) const {
		return node * size_;
	}


}