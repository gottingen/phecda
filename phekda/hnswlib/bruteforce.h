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

#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <assert.h>
#include <turbo/log/logging.h>

namespace phekda {

    class BruteforceSearch : public AlgorithmInterface {
    public:
        char *data_;
        size_t cur_element_count;
        size_t size_per_element_;
        uint64_t snapshot_id_{0};

        size_t data_size_;
        DISTFUNC<DistanceType> fstdistfunc_;
        void *dist_func_param_;
        std::mutex index_lock;

        HnswlibConfig hnsw_conf;
        CoreConfig core_conf;

        std::unordered_map<LabelType, size_t> dict_external_to_internal;


        BruteforceSearch()
                : data_(nullptr),
                  cur_element_count(0),
                  size_per_element_(0),
                  data_size_(0),
                  dist_func_param_(nullptr) {
        }

        ~BruteforceSearch() {
            free(data_);
        }

        turbo::Status initialize(const CoreConfig &config, const HnswlibConfig &hnswlib_config) override {
            hnsw_conf = hnswlib_config;
            core_conf = config;
            if(!hnswlib_config.space) {
                return turbo::invalid_argument_error("space is null");
            }
            data_size_ = hnswlib_config.space->get_data_size();
            fstdistfunc_ = hnswlib_config.space->get_dist_func();
            dist_func_param_ = hnswlib_config.space->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(LabelType);
            data_ = (char *) malloc(core_conf.max_elements * size_per_element_);
            if (data_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: BruteforceSearch failed to allocate data");
            }
            cur_element_count = 0;
            return turbo::OkStatus();
        }

        HnswlibConfig get_index_config() const override {
            return hnsw_conf;
        }

        CoreConfig get_core_config() const override {
            return core_conf;
        }

        uint64_t snapshot_id() const override {
            return snapshot_id_;
        }

        turbo::Status addPoint(const void *datapoint, LabelType label, HnswlibWriteConfig wconf) override {
            int idx;
            {
                std::unique_lock<std::mutex> lock(index_lock);

                auto search = dict_external_to_internal.find(label);
                if (search != dict_external_to_internal.end()) {
                    idx = search->second;
                } else {
                    if (cur_element_count >= core_conf.max_elements) {
                        return turbo::resource_exhausted_error(
                                "The number of elements exceeds the specified limit [%d:%d]", cur_element_count,
                                core_conf.max_elements);
                    }
                    idx = cur_element_count;
                    dict_external_to_internal[label] = idx;
                    cur_element_count++;
                }
            }
            memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(LabelType));
            memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
            return turbo::OkStatus();
        }


        turbo::Status markDelete(LabelType cur_external) override{
            size_t cur_c = dict_external_to_internal[cur_external];

            dict_external_to_internal.erase(cur_external);

            LabelType label = *((LabelType *) (data_ + size_per_element_ * (cur_element_count - 1) + data_size_));
            dict_external_to_internal[label] = cur_c;
            memcpy(data_ + size_per_element_ * cur_c,
                   data_ + size_per_element_ * (cur_element_count - 1),
                   data_size_ + sizeof(LabelType));
            cur_element_count--;
            return turbo::OkStatus();
        }


        std::priority_queue<std::pair<DistanceType, LabelType >>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
            assert(k <= cur_element_count);
            std::priority_queue<std::pair<DistanceType, LabelType >> topResults;
            if (cur_element_count == 0) return topResults;
            for (int i = 0; i < k; i++) {
                DistanceType dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                LabelType label = *((LabelType *) (data_ + size_per_element_ * i + data_size_));
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.push(std::pair<DistanceType, LabelType>(dist, label));
                }
            }
            DistanceType lastdist = topResults.empty() ? std::numeric_limits<DistanceType>::max()
                                                       : topResults.top().first;
            for (int i = k; i < cur_element_count; i++) {
                DistanceType dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) {
                    LabelType label = *((LabelType *) (data_ + size_per_element_ * i + data_size_));
                    if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                        topResults.push(std::pair<DistanceType, LabelType>(dist, label));
                    }
                    if (topResults.size() > k)
                        topResults.pop();

                    if (!topResults.empty()) {
                        lastdist = topResults.top().first;
                    }
                }
            }
            return topResults;
        }

        turbo::Status search(SearchContext &context) override {
            MaxResultQueue topResults;
            if (cur_element_count == 0) return turbo::OkStatus();
            auto query_data = context.get_query();
            for (int i = 0; i < context.top_k; i++) {
                DistanceType dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                LabelType label = *((LabelType *) (data_ + size_per_element_ * i + data_size_));
                if (!context.is_exclude(label)) {
                    topResults.push({dist, label, i});
                }
            }
            DistanceType lastdist = topResults.empty() ? std::numeric_limits<DistanceType>::max()
                                                       : topResults.top().distance;
            for (int i = context.top_k; i < cur_element_count; i++) {
                DistanceType dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
                if (dist <= lastdist) {
                    LabelType label = *((LabelType *) (data_ + size_per_element_ * i + data_size_));
                    if (!context.is_exclude(label)) {
                        topResults.push({dist, label, i});
                    }
                    if (topResults.size() > context.top_k)
                        topResults.pop();

                    if (!topResults.empty()) {
                        lastdist = topResults.top().distance;
                    }
                }
            }
            auto with_location = context.with_location;
            while (!topResults.empty() && context.results.size() < context.top_k) {
                auto &top = topResults.top();
                with_location ? context.results.push_back(top) : context.results.push_back({top.distance, top.label,0});
                topResults.pop();
            }
            return turbo::OkStatus();
        }

        virtual turbo::Status getVector(LabelType label, void *data) override{
            std::unique_lock<std::mutex> lock(index_lock);
            auto search = dict_external_to_internal.find(label);
            if (search == dict_external_to_internal.end()) {
                return turbo::not_found_error("label not found");
            }
            size_t idx = search->second;
            memcpy(data, data_ + size_per_element_ * idx, data_size_);
            return turbo::OkStatus();
        }

        turbo::Status saveIndex(const std::string &location, uint64_t snapshot) override {
            try {
                std::ofstream output(location, std::ios::binary);
                std::streampos position;
                snapshot_id_ = snapshot;
                // save core config
                writeBinaryPOD(output, static_cast<uint32_t>(core_conf.index_type));
                writeBinaryPOD(output, static_cast<uint32_t>(core_conf.data));
                writeBinaryPOD(output, static_cast<uint32_t>(core_conf.metric));
                writeBinaryPOD(output, core_conf.dimension);
                writeBinaryPOD(output, core_conf.worker_num);
                writeBinaryPOD(output, core_conf.max_elements);
                writeBinaryPOD(output, snapshot_id_);
                writeBinaryPOD(output, size_per_element_);
                writeBinaryPOD(output, cur_element_count);

                output.write(data_, core_conf.max_elements * size_per_element_);

                output.close();
            } catch (std::exception &e) {
                return turbo::internal_error(e.what());
            }
            return turbo::OkStatus();
        }


        turbo::Status
        loadIndex(const std::string &location, const CoreConfig &config, const HnswlibConfig &hnswlib_config) override {
            std::ifstream input(location, std::ios::binary);
            std::streampos position;

            CoreConfig tmp_core_conf;
            uint32_t tmp;
            readBinaryPOD(input, tmp);
            tmp_core_conf.index_type = static_cast<IndexType>(tmp);
            readBinaryPOD(input, tmp);
            tmp_core_conf.data = static_cast<DataType>(tmp);
            readBinaryPOD(input, tmp);
            tmp_core_conf.metric = static_cast<MetricType>(tmp);
            readBinaryPOD(input, tmp_core_conf.dimension);
            readBinaryPOD(input, tmp_core_conf.worker_num);
            readBinaryPOD(input, tmp_core_conf.max_elements);
            core_conf = tmp_core_conf;
            readBinaryPOD(input, snapshot_id_);
            readBinaryPOD(input, size_per_element_);
            readBinaryPOD(input, cur_element_count);
            hnsw_conf = hnswlib_config;
            data_size_ = hnswlib_config.space->get_data_size();
            fstdistfunc_ = hnswlib_config.space->get_dist_func();
            dist_func_param_ = hnswlib_config.space->get_dist_func_param();
            size_per_element_ = data_size_ + sizeof(LabelType);
            data_ = (char *) malloc(core_conf.max_elements * size_per_element_);
            if (data_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: loadIndex failed to allocate data");
            }

            input.read(data_, core_conf.max_elements * size_per_element_);

            input.close();
            return turbo::OkStatus();
        }
    };
}  // namespace phekda
