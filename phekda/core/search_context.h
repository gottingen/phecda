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

#include <vector>
#include <phekda/core/defines.h>
#include <phekda/core/search_condition.h>
#include <turbo/times/time.h>
#include <any>

namespace phekda {

    class UnifiedIndex;
    struct SearchContext {
        ~SearchContext() = default;
        // no copy able, if need copy, use move
        SearchContext(const SearchContext&) = delete;
        SearchContext& operator=(const SearchContext&) = delete;
        SearchContext(SearchContext&&) = default;
        SearchContext& operator=(SearchContext&&) = default;
        /// basic information for search
        /// set by UnifiedIndex
        MetricType metric_type{MetricType::METRIC_NONE};
        IndexType index_type{IndexType::INDEX_NONE};
        // data type
        DataType data_type{DataType::NONE};
        // original dimension
        uint32_t  dimension{0};
        // uint8_t bytes data size
        // data size = dimension * data_type_size(data_type)
        size_t data_size{0};
        // for index private meta
        // default is empty
        std::any index_meta;
        /// control section
        // worker number
        // maybe not effective in some index
        uint32_t worker_num{1};

        // query vector
        AlingedQueryVector query;
        // top k search result
        uint32_t top_k{0};
        // search list size
        uint32_t search_list_size{0};
        // if true, search result will take location into account
        // this mostly used in debug mode
        bool with_location{false};
        // if true, raw vectors will be returned with search result
        bool with_raw_vector{false};
        // specify config for corresponding index
        // this field is set by user, coresponding the
        // index_meta field is setting by index, expose
        // index meta to user
        // the flow assume like this:
        //
        // 1.auto context = index.create_search_context();
        // 2. auto index_setting = context.index_meta;
        // ... do some about the index_setting
        // 3. user_set = xxxx;
        // then set up your setting
        // 4. context.with_index_conf(user_set);
        std::any index_conf;
        /// data section
        // query start time
        turbo::Time start_time;
        // query start to schedule compute time
        turbo::Time schedule_time;
        // query end time
        turbo::Time end_time;
        // search condition, can be nullptr
        SearchCondition *condition{nullptr};
        // search labels
        // if with_location is true, the location field will be filled
        // otherwise, the location field always be 0
        std::vector<ResultEntity> results;
        // raw vectors if with_raw_vector is true
        std::vector<std::vector<uint8_t>> raw_vectors;

        /// builder section
        SearchContext& with_worker_num(uint32_t num) {
            this->worker_num = num;
            return *this;
        }

        SearchContext& with_query(const turbo::Nonnull<uint8_t *> query_ptr, uint32_t bytes) {
            this->query.assign(query_ptr, query_ptr + bytes);
            return *this;
        }

        SearchContext& with_top_k(uint32_t k) {
            this->top_k = k;
            return *this;
        }

        SearchContext& with_search_list_size(uint32_t search_list) {
            this->search_list_size = search_list;
            return *this;
        }

        SearchContext& with_with_location(bool flags) {
            this->with_location = flags;
            return *this;
        }

        SearchContext& with_with_raw_vector(bool flags) {
            this->with_raw_vector = flags;
            return *this;
        }

        SearchContext& with_index_conf(std::any conf) {
            this->index_conf = std::move(conf);
            return *this;
        }

        SearchContext& with_condition(turbo::Nullable<SearchCondition*> cond) {
            this->condition = cond;
            return *this;
        }

        TURBO_MUST_USE_RESULT const void* get_query() const {
            return query.data();
        }

        // some index need normalized query
        // to the corresponding metric space
        // eg. cosine space, l2 space
        void * mutable_query() {
            return query.data();
        }

        TURBO_MUST_USE_RESULT bool has_condition() const {
            return condition != nullptr;
        }

        TURBO_MUST_USE_RESULT bool is_exclude(LabelType label) const {
            if (condition == nullptr) {
                return false;
            }
            return condition->is_exclude(label);
        }

        TURBO_MUST_USE_RESULT bool is_whitelist(LabelType label) const {
            if (condition == nullptr) {
                return false;
            }
            return condition->is_whitelist(label);
        }

        TURBO_MUST_USE_RESULT bool should_stop_search(DistanceType dis) const {
            if (condition == nullptr) {
                return false;
            }
            return condition->should_stop_search(dis);
        }

        TURBO_MUST_USE_RESULT bool should_explain() const {
            if (condition == nullptr) {
                return false;
            }
            return condition->should_explain();
        }
    private:
        friend class UnifiedIndex;
        // only UnifiedIndex can create SearchContext
        // and make up basic information
        SearchContext() = default;
    };

}  // namespace phekda
