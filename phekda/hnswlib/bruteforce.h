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

namespace phekda {

class BruteforceSearch : public AlgorithmInterface {
 public:
    char *data_;
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_per_element_;

    size_t data_size_;
    DISTFUNC <DistanceType> fstdistfunc_;
    void *dist_func_param_;
    std::mutex index_lock;

    std::unordered_map<LabelType, size_t > dict_external_to_internal;


    BruteforceSearch()
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }

    ~BruteforceSearch() {
        free(data_);
    }

    turbo::Status initialize(const CoreConfig &config, const HnswlibConfig &hnswlib_config) override {
        maxelements_ = config.max_elements;
        data_size_ = hnswlib_config.space->get_data_size();
        fstdistfunc_ = hnswlib_config.space->get_dist_func();
        dist_func_param_ = hnswlib_config.space->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(LabelType);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr) {
            return turbo::resource_exhausted_error("Not enough memory: BruteforceSearch failed to allocate data");
        }
        cur_element_count = 0;
        return turbo::OkStatus();
    }

    void addPoint(const void *datapoint, LabelType label, bool replace_deleted = false) {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);

            auto search = dict_external_to_internal.find(label);
            if (search != dict_external_to_internal.end()) {
                idx = search->second;
            } else {
                if (cur_element_count >= maxelements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit\n");
                }
                idx = cur_element_count;
                dict_external_to_internal[label] = idx;
                cur_element_count++;
            }
        }
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(LabelType));
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
    }


    void removePoint(LabelType cur_external) {
        size_t cur_c = dict_external_to_internal[cur_external];

        dict_external_to_internal.erase(cur_external);

        LabelType label = *((LabelType*)(data_ + size_per_element_ * (cur_element_count-1) + data_size_));
        dict_external_to_internal[label] = cur_c;
        memcpy(data_ + size_per_element_ * cur_c,
                data_ + size_per_element_ * (cur_element_count-1),
                data_size_+sizeof(LabelType));
        cur_element_count--;
    }


    std::priority_queue<std::pair<DistanceType, LabelType >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        assert(k <= cur_element_count);
        std::priority_queue<std::pair<DistanceType, LabelType >> topResults;
        if (cur_element_count == 0) return topResults;
        for (int i = 0; i < k; i++) {
            DistanceType dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            LabelType label = *((LabelType*) (data_ + size_per_element_ * i + data_size_));
            if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                topResults.push(std::pair<DistanceType, LabelType>(dist, label));
            }
        }
        DistanceType lastdist = topResults.empty() ? std::numeric_limits<DistanceType>::max() : topResults.top().first;
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


    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, cur_element_count);

        output.write(data_, maxelements_ * size_per_element_);

        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<DistanceType> *s) {
        std::ifstream input(location, std::ios::binary);
        std::streampos position;

        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, cur_element_count);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(LabelType);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);

        input.close();
    }
};
}  // namespace phekda
