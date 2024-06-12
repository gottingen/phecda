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

#include <assert.h>

#include <vector>
#include <iostream>

namespace {

using idx_t = phekda::LabelType;

void test() {
    int d = 4;
    idx_t n = 100;
    idx_t nq = 10;
    size_t k = 10;

    phekda::CoreConfig core_config;
    core_config.max_elements = 2*n;
    core_config.dimension = d;
    core_config.data = phekda::DataType::FLOAT32;
    core_config.metric = phekda::MetricType::METRIC_L2;
    core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;

    phekda::CoreConfig flat_config;
    flat_config.max_elements = 2*n;
    flat_config.dimension = d;
    flat_config.data = phekda::DataType::FLOAT32;
    flat_config.metric = phekda::MetricType::METRIC_L2;
    flat_config.index_type = phekda::IndexType::INDEX_HNSW_FLAT;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    phekda::L2Space space(d);
    phekda::HnswlibConfig config;
    config.M = 16;
    config.ef_construction = 200;
    config.random_seed = 123;
    config.allow_replace_deleted = true;
    config.space = &space;
    phekda::AlgorithmInterface* alg_brute  = new phekda::BruteforceSearch();
    auto rs = alg_brute->initialize(flat_config, config);
    if(!rs.ok()) {
        std::cout << "Failed to initialize FLAT HNSW: " << rs<< std::endl;
        exit(1);
    }
    phekda::AlgorithmInterface* alg_hnsw = new phekda::HierarchicalNSW();
    rs = alg_hnsw->initialize(core_config, config);
    if(!rs.ok()) {
        std::cout << "Failed to initialize HNSW: " << rs<< std::endl;
        exit(1);
    }

    for (size_t i = 0; i < n; ++i) {
        alg_brute->addPoint(data.data() + d * i, i);
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    // test searchKnnCloserFirst of BruteforceSearch
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k);
        auto res = alg_brute->searchKnnCloserFirst(p, k);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }

    delete alg_brute;
    delete alg_hnsw;
}

}  // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}
