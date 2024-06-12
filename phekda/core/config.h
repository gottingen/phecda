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
#include <any>
#include <phekda/core/defines.h>

namespace phekda {

    struct CoreConfig {
        IndexType index_type{IndexType::INDEX_HNSWLIB};
        DataType data{DataType::FLOAT32};
        MetricType metric{MetricType::METRIC_L2};
        uint32_t dimension{0};
        uint32_t worker_num{0};
        uint32_t max_elements{0};
    };

    struct IndexConfig {
        CoreConfig core;
        std::any index_conf;

        IndexConfig() = default;

        IndexConfig &with_metric(MetricType metric) {
            core.metric = metric;
            return *this;
        }

        IndexConfig &with_data_type(DataType dt) {
            core.data = dt;
            return *this;
        }

        IndexConfig &with_dimension(uint32_t dimension) {
            core.dimension = dimension;
            return *this;
        }

        IndexConfig &with_worker_num(uint32_t worker_num) {
            core.worker_num = worker_num;
            return *this;
        }

        IndexConfig &with_index(std::any index) {
            this->index_conf = std::move(index);
            return *this;
        }

        IndexConfig &with_max_elements(uint32_t max_elements) {
            core.max_elements = max_elements;
            return *this;
        }
    };

}  // namespace phekda

