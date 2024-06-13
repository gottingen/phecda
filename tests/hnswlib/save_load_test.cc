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
// Created by jeff on 24-6-12.
//
#include <phekda/hnswlib/index.h>
#include <gtest/gtest.h>

#include <vector>
#include <iostream>

class HnswIndexTest : public ::testing::Test {
public:
    HnswIndexTest() = default;
    ~HnswIndexTest() override = default;

    void SetUp() override {
        // code here will execute just before the test ensues
        core_config.max_elements = 2 * n;
        core_config.dimension = d;
        core_config.data = phekda::DataType::FLOAT32;
        core_config.metric = phekda::MetricType::METRIC_L2;
        core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;

        std::vector<float> data(n * d);
        std::vector<float> query(nq * d);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib;

        for (phekda::LabelType i = 0; i < n * d; ++i) {
            data[i] = distrib(rng);
        }
        for (phekda::LabelType i = 0; i < nq * d; ++i) {
            query[i] = distrib(rng);
        }
        config.M = 16;
        config.ef_construction = 200;
        config.random_seed = 123;
        config.allow_replace_deleted = true;
    }

    void TearDown() override {
        // code here will be called just after the test completes
        if(alg_brute) {
            delete alg_brute;
            alg_brute = nullptr;
        }
    }

    int d = 4;
    phekda::LabelType n = 100;
    phekda::LabelType nq = 10;
    size_t k = 10;
    phekda::CoreConfig core_config;
    phekda::AlgorithmInterface *alg_brute{nullptr};
    phekda::HnswlibConfig config;
};

TEST_F(HnswIndexTest, Initialize) {
    alg_brute = new phekda::BruteforceSearch();
    EXPECT_TRUE(alg_brute != nullptr);
    phekda::L2Space space(d);
    config.space = &space;
    auto rs = alg_brute->initialize(core_config, config);
    EXPECT_TRUE(rs.ok());
}

TEST_F(HnswIndexTest, flat_save_load) {
    alg_brute = new phekda::BruteforceSearch();
    EXPECT_TRUE(alg_brute != nullptr);
    phekda::L2Space space(d);
    config.space = &space;
    auto rs = alg_brute->initialize(core_config, config);
    EXPECT_TRUE(rs.ok());
    // add some points
    for (phekda::LabelType i = 0; i < n; ++i) {
        std::vector<float> data(d);
        for (int j = 0; j < d; ++j) {
            data[j] = i * d + j;
        }
        rs = alg_brute->addPoint(data.data(), i, phekda::kHnswNotReplaceDeleted);
        EXPECT_TRUE(rs.ok());
    }
    // save the index
    uint64_t snapshot =  11;
    rs = alg_brute->saveIndex("brute_index", snapshot);
    EXPECT_TRUE(rs.ok());
    // load
    phekda::AlgorithmInterface *alg_brute_load = new phekda::BruteforceSearch();
    rs = alg_brute_load->loadIndex("brute_index", core_config, config);
    EXPECT_TRUE(rs.ok());
    // check core config
    auto core_config_load = alg_brute_load->get_core_config();
    EXPECT_EQ(core_config_load.max_elements, core_config.max_elements);
    EXPECT_EQ(11, alg_brute_load->snapshot_id());
}

TEST_F(HnswIndexTest, save_load) {
    alg_brute = new phekda::HierarchicalNSW();
    EXPECT_TRUE(alg_brute != nullptr);
    phekda::L2Space space(d);
    config.space = &space;
    auto rs = alg_brute->initialize(core_config, config);
    EXPECT_TRUE(rs.ok());
    // add some points
    for (phekda::LabelType i = 0; i < n; ++i) {
        std::vector<float> data(d);
        for (int j = 0; j < d; ++j) {
            data[j] = i * d + j;
        }
        rs = alg_brute->addPoint(data.data(), i, phekda::kHnswNotReplaceDeleted);
        EXPECT_TRUE(rs.ok());
    }
    // save the index
    uint64_t snapshot =  11;
    rs = alg_brute->saveIndex("brute_index", snapshot);
    EXPECT_TRUE(rs.ok());
    // load
    phekda::AlgorithmInterface *alg_brute_load = new phekda::BruteforceSearch();
    rs = alg_brute_load->loadIndex("brute_index", core_config, config);
    EXPECT_TRUE(rs.ok());
    // check core config
    auto core_config_load = alg_brute_load->get_core_config();
    EXPECT_EQ(core_config_load.max_elements, core_config.max_elements);
    EXPECT_EQ(11, alg_brute_load->snapshot_id());
}