//
// Created by zolkin on 8/7/24.
//

#include "cross_entropy.h"

namespace torc::sample {
    CrossEntropy::CrossEntropy(const std::string& xml_path, int num_samples)
        : dispatcher_(xml_path, num_samples) {
    }

} // namespace torc::sample