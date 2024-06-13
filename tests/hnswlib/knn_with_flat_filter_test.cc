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
#include <phekda/unified.h>
#include <phekda/hnswlib/index.h>
#include <random>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <gtest/gtest.h>

namespace {

    using idx_t = phekda::LabelType;

    class PickDivisibleIdsCondition : public phekda::SearchCondition {
        unsigned int divisor = 1;
    public:
        PickDivisibleIdsCondition(unsigned int divisor) : divisor(divisor) {
            assert(divisor != 0);
        }
        virtual bool is_exclude(phekda::LabelType label) const override {
            return label % divisor != 0;
        }
    };

    class PickNothingCondition : public phekda::SearchCondition {
    public:
        virtual bool is_exclude(phekda::LabelType label) const override {
            return true;
        }
    };

    class PickDivisibleIds : public phekda::BaseFilterFunctor {
        unsigned int divisor = 1;
    public:
        PickDivisibleIds(unsigned int divisor) : divisor(divisor) {
            assert(divisor != 0);
        }

        bool operator()(idx_t label_id) {
            return label_id % divisor == 0;
        }
    };

    class PickNothing : public phekda::BaseFilterFunctor {
    public:
        bool operator()(idx_t label_id) {
            return false;
        }
    };

    void test_some_flat_filtering(phekda::SearchCondition &filter_func, phekda::BaseFilterFunctor&filter_base, size_t div_num, size_t label_id_start) {
        int d = 4;
        idx_t n = 100;
        idx_t nq = 10;
        size_t k = 10;

        phekda::CoreConfig flat_config;
        flat_config.max_elements = 2 * n;
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

        phekda::HnswlibConfig config;
        config.M = 16;
        config.ef_construction = 200;
        config.random_seed = 123;
        config.allow_replace_deleted = true;

        phekda::IndexConfig flat_index_config;
        flat_index_config.core = flat_config;
        flat_index_config.index_conf = config;

        auto *alg_brute = phekda::UnifiedIndex::create_index(flat_config.index_type);
        auto rs = alg_brute->initialize(flat_index_config);
        if (!rs.ok()) {
            std::cout << "Failed to initialize FLAT HNSW: " << rs << std::endl;
            exit(1);
        }
        phekda::L2Space space(d);
        config.space = &space;
        phekda::AlgorithmInterface *alg_brute_base = new phekda::BruteforceSearch();
        rs = alg_brute_base->initialize(flat_config, config);
        if (!rs.ok()) {
            std::cout << "Failed to initialize FLAT HNSW: " << rs << std::endl;
            exit(1);
        }
        for (size_t i = 0; i < n; ++i) {
            // `label_id_start` is used to ensure that the returned IDs are labels and not internal IDs
            rs = alg_brute->add_vector(reinterpret_cast<const uint8_t *>(data.data() + d * i), label_id_start + i);
            if (!rs.ok()) {
                std::cout << "Failed to add point to FLAT HNSW: " << rs << std::endl;
                exit(1);
            }
            rs = alg_brute_base->addPoint(data.data() + d * i, label_id_start + i, {false});
            if (!rs.ok()) {
                std::cout << "Failed to add point to FLAT HNSW: " << rs << std::endl;
                exit(1);
            }
        }

        // test searchKnnCloserFirst of BruteforceSearch with filtering
        for (size_t j = 0; j < nq; ++j) {
            const uint8_t *p = reinterpret_cast<const uint8_t*> (query.data() + j * d);
            auto context = alg_brute->create_search_context();
            context.with_top_k(k).with_query(p).with_condition(&filter_func);
            auto gd = alg_brute->search(context);
            auto res = alg_brute_base->searchKnnCloserFirst(p, k, &filter_base);
            std::cout<<"Results:\n";
            std::cout<<"results size: "<<context.results.size()<<std::endl;
            ASSERT_TRUE(context.results.size() == res.size());
            for (auto i = 0l; i < context.results.size(); ++i) {
                std::cout<<"result: "<<context.results[i].label<<"dist: "<<context.results[i].distance<<std::endl;
                ASSERT_TRUE(context.results[i].label == res[i].second);
                ASSERT_TRUE(context.results[i].distance == res[i].first);
            }
        }
        delete alg_brute;
        delete alg_brute_base;
    }

}  // namespace

class CustomFilterFunctor : public phekda::BaseFilterFunctor {
    std::unordered_set<idx_t> allowed_values;

public:
    explicit CustomFilterFunctor(const std::unordered_set<idx_t> &values) : allowed_values(values) {}

    bool operator()(idx_t id) {
        return allowed_values.count(id) != 0;
    }
};

class CustomFilterFunctorCondition : public phekda::SearchCondition {
    std::unordered_set<idx_t> allowed_values;

public:
    explicit CustomFilterFunctorCondition(const std::unordered_set<idx_t> &values) : allowed_values(values) {}

    bool is_exclude(phekda::LabelType label) const override {
        return allowed_values.count(label) == 0;
    }
};

TEST(Hnswlib, knn_with_flat_filter_test) {
    std::cout << "Testing ..." << std::endl;

    // some of the elements are filtered
    PickDivisibleIdsCondition pickIdsDivisibleByThree(3);
    PickDivisibleIds pickIdsDivisibleByThreeBase(3);
    test_some_flat_filtering(pickIdsDivisibleByThree, pickIdsDivisibleByThreeBase, 3, 17);
    PickDivisibleIdsCondition pickIdsDivisibleBySeven(7);
    PickDivisibleIds pickIdsDivisibleBySevenBase(7);
    std::cout << "pickIdsDivisibleBySeven: " << &pickIdsDivisibleBySeven << std::endl;
    test_some_flat_filtering(pickIdsDivisibleBySeven, pickIdsDivisibleBySevenBase, 7, 17);

    // functor style which can capture context
    CustomFilterFunctor pickIdsDivisibleByThirteen({26, 39, 52, 65});
    CustomFilterFunctorCondition pickIdsDivisibleByThirteenCondition({26, 39, 52, 65});
    test_some_flat_filtering(pickIdsDivisibleByThirteenCondition, pickIdsDivisibleByThirteen, 13, 21);

    std::cout << "Test ok" << std::endl;
}
