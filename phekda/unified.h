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

#include <turbo/utility/status.h>
#include <phekda/core/config.h>
#include <phekda/core/search_context.h>
#include <phekda/version.h>

namespace phekda {

    enum class IndexInitializationType {
        INIT_NONE = 0,
        INIT_LOAD = 1,
        INIT_INIT = 2
    };

    inline std::string version() {
        return PHEKDA_VERSION_STRING;
    }

    class UnifiedIndex {
    public:
        virtual ~UnifiedIndex() = default;

        // initialize index
        virtual turbo::Status initialize(const IndexConfig &config) = 0;

        // add vector to index with label
        // implement should care the write_conf parameter to be empty write_conf.has_value() == false
        // if the parameter must be specified, it should be override the function
        // with no default parameter for write_conf
        virtual turbo::Status add_vector(turbo::Nonnull<const uint8_t *> data, LabelType label, std::any write_conf) = 0;

        virtual turbo::Status add_vector(turbo::Nonnull<const uint8_t *> data, LabelType label) {
            return add_vector(data, label, {});
        }

        // add vectors to index with labels
        // implement should care the write_conf parameter to be empty write_conf.has_value() == false
        // if the parameter must be specified, it should be override the function
        // with no default parameter for write_conf
        virtual turbo::Status
        add_vectors(turbo::Nonnull<const uint8_t *> data, turbo::Nonnull<const LabelType *> labels, uint32_t num, std::any write_conf) = 0;

        virtual turbo::Status
        add_vectors(turbo::Nonnull<const uint8_t *> data, turbo::Nonnull<const LabelType *> labels, uint32_t num) {
            return add_vectors(data, labels, num, {});
        }
        // remove vector from index, return false if not found
        // the data will should allocate by the caller
        virtual turbo::Status get_vector(LabelType label, turbo::Nonnull<uint8_t *> data) = 0;

        // get vectors from index with labels
        virtual turbo::Status
        get_vectors(turbo::Nonnull<const LabelType *> labels, uint32_t num, turbo::Nonnull<uint8_t *> data) = 0;

        // create meta,
        // default is init core config
        // if index need more meta, it should be specified in the config.index_conf
        // and override this function to create the meta
        virtual SearchContext create_search_context() const;
        // search vectors in index
        // this is the only way to search in index
        // if the index search in different way, it should be specified in the
        // config.index_conf, and the index should be able to parse the config
        // and search in the way specified in the config
        virtual turbo::Status search(SearchContext &context) = 0;

        // remove vector from index, just mark it as deleted
        // the vector should not present in search result, but
        // in some index,may using it as a way to link to other vectors
        // some index may delete directly
        virtual turbo::Status lazy_delete(LabelType label) = 0;

        // remove vectors solidly from index
        virtual turbo::Result<ConsolidationReport> consolidate(const std::any &conf) = 0;

        // get index snapshot
        // the index should be able to provide a snapshot
        virtual LabelType snapshot_id() const = 0;

        // install snapshot to index
        // the index should be able to install the snapshot
        // at this time, the index can be searched, but add and delete
        // should be blocked
        // for search engine, although the index need not be so strict
        // as the traditional database, but it should be able to provide
        virtual turbo::Status save(LabelType snapshot_id, const std::string &path, const std::any &save_conf) = 0;

        // load snapshot to index
        // all the operation should be blocked until the snapshot is loaded
        virtual turbo::Status load(const std::string &path, const IndexConfig &config) = 0;

        // index is real time or not
        virtual bool support_dynamic() const = 0;

        /// for index for not support dynamic
        // index need train or not
        virtual bool need_train() const = 0;

        // train index
        virtual turbo::Status train(std::any conf) = 0;

        // is index trained
        virtual bool is_trained() const = 0;

        // index need build or not
        // various index may need build or not
        // and the build may be different
        // from the data source, eg sift data or
        // data from h5 file, or from rocksdb
        // let it configurable in the build function's
        // conf parameter, and trans the ownership of the
        // conf to index, let index judge if it can build
        virtual bool support_build(std::any conf) const = 0;

        // build index
        // build index should not modify the data in the index
        // only can be visited the parameters in the index
        // after build it should output a new index data.
        // next time a new index should be call load to load the new index data
        virtual turbo::Status build(std::any conf) const = 0;

        virtual CoreConfig get_core_config() const = 0;

        virtual IndexConfig get_index_config() const = 0;

        virtual IndexInitializationType get_initialization_type() const = 0;

    public:

        /// create index
        static UnifiedIndex* create_index(IndexType type);
    };

}  // namespace phekda

