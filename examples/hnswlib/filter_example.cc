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

#include <phekda/hnswlib/index.h>
#include <phekda/unified.h>
#include <thread>


// Filter that allows labels divisible by divisor
class PickDivisibleIds: public phekda::SearchCondition {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool  is_exclude(phekda::LabelType label) const {
        return label % divisor != 0;
    }
};


int main() {

    phekda::CoreConfig core_config;
    core_config.max_elements = 10000;
    core_config.dimension = 16;
    core_config.data = phekda::DataType::FLOAT32;
    core_config.metric = phekda::MetricType::METRIC_L2;
    core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;

    phekda::HnswlibConfig config;
    config.M = 16;
    config.ef_construction = 200;
    config.random_seed = 123;
    config.allow_replace_deleted = true;

    // Initing index
    auto alg_hnsw = phekda::UnifiedIndex::create_index(core_config.index_type);
    auto rs =alg_hnsw->initialize({core_config, config});
    if(!rs.ok()) {
        std::cout << "Error: " << rs.message() << std::endl;
        return 1;
    }

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[core_config.dimension * core_config.max_elements];
    for (int i = 0; i < core_config.dimension * core_config.max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < core_config.max_elements; i++) {
        rs = alg_hnsw->add_vector(reinterpret_cast<const uint8_t*> (data + i * core_config.dimension), i);
        if(!rs.ok()) {
            std::cout << "Error: " << rs.message() << std::endl;
            exit(1);
        }
    }

    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    // Query the elements for themselves with filter and check returned labels
    int k = 10;
    for (int i = 0; i < core_config.max_elements; i++) {
        auto context = alg_hnsw->create_search_context();
        context.with_query(reinterpret_cast<const uint8_t *>(data + i * core_config.dimension)).with_top_k(k).with_condition(&pickIdsDivisibleByTwo);
        rs = alg_hnsw->search(context);
        if(!rs.ok()) {
            std::cout << "Error: " << rs.message() << std::endl;
            exit(1);
        }
        for (auto item: context.results) {
            if (item.label % 2 == 1) std::cout << "Error: found odd label\n";
        }
    }

    delete[] data;
    delete alg_hnsw;
    std::cout << "All tests passed\n";
    return 0;
}
