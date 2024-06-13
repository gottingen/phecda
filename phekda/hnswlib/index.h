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

#include <phekda/unified.h>
#include <phekda/hnswlib/hnswalg.h>
#include <phekda/hnswlib/bruteforce.h>
#include <phekda/hnswlib/space_ip.h>
#include <phekda/hnswlib/space_l2.h>
#include <turbo/synchronization/mutex.h>

namespace phekda {

    class HnswIndex : public UnifiedIndex {
    public:
        HnswIndex() = default;

        ~HnswIndex() override = default;

        // initialize index
        turbo::Status initialize(const IndexConfig &config) override;

        // add vector to index with label
        turbo::Status add_vector(turbo::Nonnull<const uint8_t *> data, LabelType label, std::any write_conf) override;

        // add vectors to index with labels
        turbo::Status
        add_vectors(turbo::Nonnull<const uint8_t *> data, turbo::Nonnull<const LabelType *> labels, uint32_t num,
                    std::any write_conf) override;

        // remove vector from index, return false if not found
        // the data will should allocate by the caller
        turbo::Status get_vector(LabelType label, turbo::Nonnull<uint8_t *> data) override;

        // get vectors from index with labels
        turbo::Status
        get_vectors(turbo::Nonnull<const LabelType *> labels, uint32_t num, turbo::Nonnull<uint8_t *> data) override;

        // search vectors in index
        // this is the only way to search in index
        // if the index search in different way, it should be specified in the
        // config.index_conf, and the index should be able to parse the config
        // and search in the way specified in the config
        turbo::Status search(SearchContext &context) override;

        // remove vector from index, just mark it as deleted
        // the vector should not present in search result, but
        // in some index,may using it as a way to link to other vectors
        // some index may delete directly
        turbo::Status lazy_delete(LabelType label) override;

        // remove vectors solidly from index
        turbo::Result<ConsolidationReport> consolidate(const std::any &conf) override;

        // get index snapshot
        // the index should be able to provide a snapshot
        LabelType snapshot_id() const override;

        // install snapshot to index
        // the index should be able to install the snapshot
        // at this time, the index can be searched, but add and delete
        // should be blocked
        // for search engine, although the index need not be so strict
        // as the traditional database, but it should be able to provide
        turbo::Status save(LabelType snapshot_id, const std::string &path, const std::any &save_conf) override;

        // load snapshot to index
        // all the operation should be blocked until the snapshot is loaded
        turbo::Status load(const std::string &path, const IndexConfig &config) override;

        // index is real time or not
        bool support_dynamic() const override {
            return true;
        }

        /// for index for not support dynamic
        // index need train or not
        bool need_train() const override {
            return false;
        }

        // train index
        turbo::Status train(std::any conf) override {
            return turbo::OkStatus();
        }

        // is index trained
        bool is_trained() const override {
            return true;
        }

        // index need build or not
        // various index may need build or not
        // and the build may be different
        // from the data source, eg sift data or
        // data from h5 file, or from rocksdb
        // let it configurable in the build function's
        // conf parameter, and trans the ownership of the
        // conf to index, let index judge if it can build
        bool support_build(std::any conf) const override {
            return false;
        }

        // build index
        // build index should not modify the data in the index
        // only can be visited the parameters in the index
        // after build it should output a new index data.
        // next time a new index should be call load to load the new index data
        turbo::Status build(std::any conf) const override {
            return turbo::unavailable_error("build not supported");
        }

        CoreConfig get_core_config() const override ;

        IndexConfig get_index_config() const override;

        IndexInitializationType get_initialization_type() const override {
            return init_type_;
        }
    private:
        turbo::Mutex         init_mutex_;
        IndexInitializationType                init_type_{IndexInitializationType::INIT_NONE};
        std::unique_ptr<AlgorithmInterface> alg_{nullptr};
        std::unique_ptr<SpaceInterface<float>> space_{nullptr};
    };
}  // namespace phekda
