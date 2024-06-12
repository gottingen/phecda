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

class PickDivisibleIds: public phekda::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(idx_t label_id) {
        return label_id % divisor == 0;
    }
};

class PickNothing: public phekda::BaseFilterFunctor {
 public:
    bool operator()(idx_t label_id) {
        return false;
    }
};

void test_some_filtering(phekda::BaseFilterFunctor& filter_func, size_t div_num, size_t label_id_start) {
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
        // `label_id_start` is used to ensure that the returned IDs are labels and not internal IDs
        alg_brute->addPoint(data.data() + d * i, label_id_start + i);
        alg_hnsw->addPoint(data.data() + d * i, label_id_start + i);
    }

    // test searchKnnCloserFirst of BruteforceSearch with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k, &filter_func);
        auto res = alg_brute->searchKnnCloserFirst(p, k, &filter_func);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            assert((gd.top().second % div_num) == 0);
            gd.pop();
        }
    }

    // test searchKnnCloserFirst of hnsw with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k, &filter_func);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k, &filter_func);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            assert((gd.top().second % div_num) == 0);
            gd.pop();
        }
    }

    delete alg_brute;
    delete alg_hnsw;
}

void test_none_filtering(phekda::BaseFilterFunctor& filter_func, size_t label_id_start) {
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
        // `label_id_start` is used to ensure that the returned IDs are labels and not internal IDs
        alg_brute->addPoint(data.data() + d * i, label_id_start + i);
        alg_hnsw->addPoint(data.data() + d * i, label_id_start + i);
    }

    // test searchKnnCloserFirst of BruteforceSearch with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k, &filter_func);
        auto res = alg_brute->searchKnnCloserFirst(p, k, &filter_func);
        assert(gd.size() == res.size());
        assert(0 == gd.size());
    }

    // test searchKnnCloserFirst of hnsw with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k, &filter_func);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k, &filter_func);
        assert(gd.size() == res.size());
        assert(0 == gd.size());
    }

    delete alg_brute;
    delete alg_hnsw;
}

}  // namespace

class CustomFilterFunctor: public phekda::BaseFilterFunctor {
    std::unordered_set<idx_t> allowed_values;

 public:
    explicit CustomFilterFunctor(const std::unordered_set<idx_t>& values) : allowed_values(values) {}

    bool operator()(idx_t id) {
        return allowed_values.count(id) != 0;
    }
};

int main() {
    std::cout << "Testing ..." << std::endl;

    // some of the elements are filtered
    PickDivisibleIds pickIdsDivisibleByThree(3);
    test_some_filtering(pickIdsDivisibleByThree, 3, 17);
    PickDivisibleIds pickIdsDivisibleBySeven(7);
    test_some_filtering(pickIdsDivisibleBySeven, 7, 17);

    // all of the elements are filtered
    PickNothing pickNothing;
    test_none_filtering(pickNothing, 17);

    // functor style which can capture context
    CustomFilterFunctor pickIdsDivisibleByThirteen({26, 39, 52, 65});
    test_some_filtering(pickIdsDivisibleByThirteen, 13, 21);

    std::cout << "Test ok" << std::endl;

    return 0;
}
