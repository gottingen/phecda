//
// Copyright (C) 2024 EA group inc.
// Author: Jeff.li lijippy@163.com
// All rights reserved.
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
//
// Created by jeff on 24-6-11.
//

#pragma once

#include <cstdint>
#include <tuple>
#include <queue>
#include <phekda/core/aligned_allocator.h>
#include <turbo/numeric/bits.h>

#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

namespace phekda {

    enum class DataType{
        NONE,
        UINT8,
        FLOAT16,
        FLOAT32
    };

    inline constexpr size_t data_type_size(DataType type) {
        switch (type) {
            case DataType::UINT8:
                return 1;
            case DataType::FLOAT16:
                return 2;
            case DataType::FLOAT32:
                return 4;
            default:
                return 0;
        }
    }

    enum class MetricType {
        METRIC_NONE,
        METRIC_L1,
        METRIC_L2,
        METRIC_IP,
        METRIC_COSINE
    };

    using DistanceType = float;

    using LocationType = uint32_t;

    using LabelType = uint64_t;

    // for performance, we need to make
    // sure the data is aligned to instruction set
    // for avx512, the data should be aligned to 64 bytes
    // for avx2, the data should be aligned to 32 bytes
#if defined(__AVX512F__)
    static constexpr uint32_t aligned_bytes = 64;
#elif defined(__AVX2__)
    static constexpr uint32_t aligned_bytes = 32;
#endif

    static constexpr uint32_t dimension_alignment(DataType data_type) {
        return aligned_bytes / data_type_size(data_type);
    }

    static constexpr uint32_t aligned_dimension(DataType data_type, uint32_t dimension) {
        return ROUND_UP(dimension, dimension_alignment(data_type));
    }

    using AlingedQueryVector = std::vector<uint8_t, aligned_allocator<float, aligned_bytes>>;

    enum class IndexType {
        INDEX_NONE,
        INDEX_HNSW_FLAT,
        INDEX_HNSWLIB
    };

    struct ConsolidationReport {
        enum status_code {
            SUCCESS = 0,
            FAIL = 1,
            LOCK_FAIL = 2,
            INCONSISTENT_COUNT_ERROR = 3
        };
        status_code status{status_code::SUCCESS};
        size_t active_points{0};
        size_t max_points{0};
        size_t empty_slots{0};
        size_t slots_released{0};
        size_t delete_set_size{0};
        size_t num_calls_to_process_delete{0};
        double time{0.0};
        ConsolidationReport() = default;
    };

    struct ResultEntity {
        ResultEntity(DistanceType d, LabelType l, LocationType loc) : distance(d), label(l), location(loc) {}
        DistanceType distance{0.0};
        LabelType label{0};
        LocationType location{0};
    };

    struct LessResultEntity {
        bool operator()(const ResultEntity &lhs, const ResultEntity &rhs) const {
            return lhs.distance < rhs.distance;
        }
    };

    struct GreatResultEntity {
        bool operator()(const ResultEntity &lhs, const ResultEntity &rhs) const {
            return lhs.distance >= rhs.distance;
        }
    };

    using MaxResultQueue = std::priority_queue<ResultEntity, std::vector<ResultEntity>, LessResultEntity>;
    using MinResultQueue = std::priority_queue<ResultEntity, std::vector<ResultEntity>, GreatResultEntity>;
    using ResultVector = std::vector<ResultEntity>;

}  // namespace phekda
