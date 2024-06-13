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

#include <phekda/hnswlib/visited_list_pool.h>
#include <phekda/hnswlib/hnswlib.h>
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <turbo/log/logging.h>

namespace phekda {

    class HierarchicalNSW : public AlgorithmInterface {
    public:
        static const LocationType MAX_LABEL_OPERATION_LOCKS = 65536;
        static const unsigned char DELETE_MARK = 0x01;

        mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
        size_t size_data_per_element_{0};
        size_t size_links_per_element_{0};
        mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
        size_t maxM_{0};
        size_t maxM0_{0};
        size_t ef_{0};

        HnswlibConfig hnsw_conf;
        CoreConfig core_conf;
        uint64_t snapshot_id_{0};

        double mult_{0.0}, revSize_{0.0};
        int maxlevel_{0};

        VisitedListPool *visited_list_pool_{nullptr};

        // Locks operations with element by label value
        mutable std::vector<std::mutex> label_op_locks_;

        std::mutex global;
        std::vector<std::mutex> link_list_locks_;

        LocationType enterpoint_node_{0};

        size_t size_links_level0_{0};
        size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

        char *data_level0_memory_{nullptr};
        char **linkLists_{nullptr};
        std::vector<int> element_levels_;  // keeps level of each element

        size_t data_size_{0};

        DISTFUNC<DistanceType> fstdistfunc_;
        void *dist_func_param_{nullptr};

        mutable std::mutex label_lookup_lock;  // lock for label_lookup_
        std::unordered_map<LabelType, LocationType> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        mutable std::atomic<long> metric_distance_computations{0};
        mutable std::atomic<long> metric_hops{0};

        std::mutex deleted_elements_lock;  // lock for deleted_elements
        std::unordered_set<LocationType> deleted_elements;  // contains internal ids of deleted elements


        HierarchicalNSW() {
        }

        ~HierarchicalNSW() {
            free(data_level0_memory_);
            for (LocationType i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        turbo::Status initialize(const CoreConfig &config, const HnswlibConfig &hnswlib_config) override {
            hnsw_conf = hnswlib_config;
            if(hnsw_conf.space == nullptr) {
                return turbo::invalid_argument_error("SpaceInterface is not set");
            }
            core_conf = config;
            std::vector<std::mutex> tmp(core_conf.max_elements);
            link_list_locks_ = std::move(tmp);
            std::vector<std::mutex> label_op_locks_tmp(MAX_LABEL_OPERATION_LOCKS);
            label_op_locks_ = std::move(label_op_locks_tmp);
            element_levels_.resize(core_conf.max_elements);
            num_deleted_ = 0;
            data_size_ = hnsw_conf.space->get_data_size();
            fstdistfunc_ = hnsw_conf.space->get_dist_func();
            dist_func_param_ = hnsw_conf.space->get_dist_func_param();
            maxM_ = hnsw_conf.M;
            maxM0_ = hnsw_conf.M * 2;
            hnsw_conf.ef_construction = std::max(hnsw_conf.ef_construction, hnsw_conf.M);
            ef_ = 10;

            level_generator_.seed(hnsw_conf.random_seed);
            update_probability_generator_.seed(hnsw_conf.random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(LocationType) + sizeof(LocationType);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(LabelType);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(core_conf.max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: HierarchicalNSW failed to allocate data");
            }

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, core_conf.max_elements);

            // initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * core_conf.max_elements);
            if (linkLists_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            }
            size_links_per_element_ = maxM_ * sizeof(LocationType) + sizeof(LocationType);
            mult_ = 1 / log(1.0 * hnsw_conf.M);
            revSize_ = 1.0 / mult_;
            return turbo::OkStatus();
        }

        HnswlibConfig  get_index_config() const override {
            return hnsw_conf;
        }

        CoreConfig  get_core_config() const override {
            return core_conf;
        }

        uint64_t snapshot_id() const override {
            return snapshot_id_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<DistanceType, LocationType> const &a,
                                      std::pair<DistanceType, LocationType> const &b) const noexcept {
                return a.first < b.first;
            }
        };


        void setEf(size_t ef) {
            ef_ = ef;
        }


        inline std::mutex &getLabelOpMutex(LabelType label) const {
            // calculate hash
            size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
            return label_op_locks_[lock_id];
        }


        inline LabelType getExternalLabel(LocationType internal_id) const {
            LabelType return_label;
            memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
                   sizeof(LabelType));
            return return_label;
        }


        inline void setExternalLabel(LocationType internal_id, LabelType label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label,
                   sizeof(LabelType));
        }


        inline LabelType *getExternalLabeLp(LocationType internal_id) const {
            return (LabelType *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }


        inline char *getDataByInternalId(LocationType internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }


        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        size_t getMaxElements() {
            return core_conf.max_elements;
        }

        size_t getCurrentElementCount() {
            return cur_element_count;
        }

        size_t getDeletedCount() {
            return num_deleted_;
        }

        std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst>
        searchBaseLayer(LocationType ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> candidateSet;

            DistanceType lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                DistanceType dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<DistanceType>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<DistanceType, LocationType> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == hnsw_conf.ef_construction) {
                    break;
                }
                candidateSet.pop();

                LocationType curNodeNum = curr_el_pair.second;

                std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int *) get_linklist0(curNodeNum);
                } else {
                    data = (int *) get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((LocationType *) data);
                LocationType *datal = (LocationType *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    LocationType candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    DistanceType dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < hnsw_conf.ef_construction || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > hnsw_conf.ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }


        template<bool has_deletions, bool collect_metrics = false>
        std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst>
        searchBaseLayerST(LocationType ep_id, const void *data_point, size_t ef,
                          BaseFilterFunctor *isIdAllowed = nullptr) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> candidate_set;

            DistanceType lowerBound;
            if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
                DistanceType dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<DistanceType>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {
                std::pair<DistanceType, LocationType> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound &&
                    (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                    break;
                }
                candidate_set.pop();

                LocationType current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((LocationType *) data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);  ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        DistanceType dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                         _MM_HINT_T0);  ////////////////////////
#endif

                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                                ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }


        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> &top_candidates,
                const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<DistanceType, LocationType>> queue_closest;
            std::vector<std::pair<DistanceType, LocationType>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<DistanceType, LocationType> curent_pair = queue_closest.top();
                DistanceType dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<DistanceType, LocationType> second_pair: return_list) {
                    DistanceType curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<DistanceType, LocationType> curent_pair: return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        LocationType *get_linklist0(LocationType internal_id) const {
            return (LocationType *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }


        LocationType *get_linklist0(LocationType internal_id, char *data_level0_memory_) const {
            return (LocationType *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        }


        LocationType *get_linklist(LocationType internal_id, int level) const {
            return (LocationType *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        }


        LocationType *get_linklist_at_level(LocationType internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        }


        LocationType mutuallyConnectNewElement(
                const void *data_point,
                LocationType cur_c,
                std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> &top_candidates,
                int level,
                bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, hnsw_conf.M);
            if (top_candidates.size() > hnsw_conf.M)
                throw std::runtime_error("Should be not be more than hnsw_conf.M candidates returned by the heuristic");

            std::vector<LocationType> selectedNeighbors;
            selectedNeighbors.reserve(hnsw_conf.M);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            LocationType next_closest_entry_point = selectedNeighbors.back();

            {
                // lock only during the update
                // because during the addition the lock for cur_c is already acquired
                std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
                if (isUpdate) {
                    lock.lock();
                }
                LocationType *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur, selectedNeighbors.size());
                LocationType *data = (LocationType *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                LocationType *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                LocationType *data = (LocationType *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        DistanceType d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                                          getDataByInternalId(selectedNeighbors[idx]),
                                                          dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]),
                                                 getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            DistanceType d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }


        void resizeIndex(size_t new_max_elements) {
            if (new_max_elements < cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);

            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char *data_level0_memory_new = (char *) realloc(data_level0_memory_,
                                                            new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char **linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            core_conf.max_elements = new_max_elements;
        }

        turbo::Status saveIndex(const std::string &location, uint64_t snapshot) override{
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
                writeBinaryPOD(output, offsetLevel0_);
                writeBinaryPOD(output, cur_element_count);
                writeBinaryPOD(output, size_data_per_element_);
                writeBinaryPOD(output, label_offset_);
                writeBinaryPOD(output, offsetData_);
                writeBinaryPOD(output, maxlevel_);
                writeBinaryPOD(output, enterpoint_node_);
                writeBinaryPOD(output, maxM_);

                writeBinaryPOD(output, maxM0_);
                writeBinaryPOD(output, hnsw_conf.M);
                writeBinaryPOD(output, mult_);
                writeBinaryPOD(output, hnsw_conf.ef_construction);

                output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

                for (size_t i = 0; i < cur_element_count; i++) {
                    unsigned int linkListSize =
                            element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                    writeBinaryPOD(output, linkListSize);
                    if (linkListSize)
                        output.write(linkLists_[i], linkListSize);
                }
                output.close();
            } catch (std::exception &e) {
                return turbo::internal_error(e.what());
            }
            return turbo::OkStatus();
        }

        turbo::Status loadIndex(const std::string &location, const CoreConfig &config, const HnswlibConfig &hnswlib_config) override {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open()) {
                return turbo::internal_error("Cannot open file");
            }
            hnsw_conf = hnswlib_config;
            core_conf = config;
            // get file size:
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            // load core config
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
            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = config.max_elements;
            if (max_elements < cur_element_count)
                max_elements =  core_conf.max_elements;
            core_conf.max_elements = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, hnsw_conf.M);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, hnsw_conf.ef_construction);

            data_size_ = hnsw_conf.space->get_data_size();
            fstdistfunc_ = hnsw_conf.space->get_dist_func();
            dist_func_param_ = hnsw_conf.space->get_dist_func_param();

            auto pos = input.tellg();

            /// Optional - check if index is ok:
            input.seekg(cur_element_count * size_data_per_element_, input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                    return turbo::internal_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize, input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if (input.tellg() != total_filesize) {
                return turbo::internal_error("Index seems to be corrupted or unsupported");
            }

            input.clear();
            /// Optional check end

            input.seekg(pos, input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: loadIndex failed to allocate level0");
            }
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(LocationType) + sizeof(LocationType);

            size_links_level0_ = maxM0_ * sizeof(LocationType) + sizeof(LocationType);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr) {
                return turbo::resource_exhausted_error("Not enough memory: loadIndex failed to allocate linklists");
            }
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;
                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr) {
                        return turbo::resource_exhausted_error("Not enough memory: loadIndex failed to allocate linklist");
                    }
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if (isMarkedDeleted(i)) {
                    num_deleted_ += 1;
                    if (hnsw_conf.allow_replace_deleted) deleted_elements.insert(i);
                }
            }

            input.close();

            return turbo::OkStatus();
        }


        template<typename data_t>
        std::vector<data_t> getDataByLabel(LabelType label) const {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            LocationType internalId = search->second;
            lock_table.unlock();

            char *data_ptrv = getDataByInternalId(internalId);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t *data_ptr = (data_t *) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }


        /*
        * Marks an element with the given label deleted, does NOT really change the current graph.
        */
        turbo::Status markDelete(LabelType label) override {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                return turbo::not_found_error("Label not found");
            }
            LocationType internalId = search->second;
            lock_table.unlock();

            return markDeletedInternal(internalId);
        }

        virtual turbo::Status getVector(LabelType label, void *data) override{
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                return turbo::not_found_error("Label not found");
            }
            LocationType internalId = search->second;
            lock_table.unlock();
            auto *ptr = getDataByInternalId(internalId);
            std::memcpy(data, ptr, data_size_);
            return turbo::OkStatus();
        }


        /*
        * Uses the last 16 bits of the memory for the linked list size to store the mark,
        * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
        */
        turbo::Status markDeletedInternal(LocationType internalId) {
            if(internalId >= cur_element_count) {
                return turbo::out_of_range_error("The requested to delete element is already deleted");
            }
            if (!isMarkedDeleted(internalId)) {
                unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
                if (hnsw_conf.allow_replace_deleted) {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.insert(internalId);
                }
            } else {
                return turbo::not_found_error("The requested to delete element is already deleted");
            }
            return turbo::OkStatus();
        }


        /*
        * Removes the deleted mark of the node, does NOT really change the current graph.
        *
        * Note: the method is not safe to use when replacement of deleted elements is enabled,
        *  because elements marked as deleted can be completely removed by addPoint
        */
        void unmarkDelete(LabelType label) {
            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

            std::unique_lock<std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            LocationType internalId = search->second;
            lock_table.unlock();

            unmarkDeletedInternal(internalId);
        }


        /*
        * Remove the deleted mark of the node.
        */
        void unmarkDeletedInternal(LocationType internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId)) {
                unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
                if (hnsw_conf.allow_replace_deleted) {
                    std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                    deleted_elements.erase(internalId);
                }
            } else {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }


        /*
        * Checks the first 16 bits of the memory to see if the element is marked deleted.
        */
        bool isMarkedDeleted(LocationType internalId) const {
            unsigned char *ll_cur = ((unsigned char *) get_linklist0(internalId)) + 2;
            return *ll_cur & DELETE_MARK;
        }


        unsigned short int getListCount(LocationType *ptr) const {
            return *((unsigned short int *) ptr);
        }


        void setListCount(LocationType *ptr, unsigned short int size) const {
            *((unsigned short int *) (ptr)) = *((unsigned short int *) &size);
        }


        /*
        * Adds point. Updates the point if it is already in the index.
        * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
        */
        turbo::Status addPoint(const void *data_point, LabelType label, HnswlibWriteConfig wconf) override {
            if ((hnsw_conf.allow_replace_deleted == false) && (wconf.replace_deleted == true)) {
                return turbo::invalid_argument_error("Replacement of deleted elements is disabled in constructor");
            }

            // lock all operations with element by label
            std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
            if (!wconf.replace_deleted) {
                return add_point_impl(data_point, label, -1).status();
            }
            // check if there is vacant place
            LocationType internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
            bool is_vacant_place = !deleted_elements.empty();
            if (is_vacant_place) {
                internal_id_replaced = *deleted_elements.begin();
                deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock();

            // if there is no vacant place then add or update point
            // else add point to vacant place
            if (!is_vacant_place) {
                return add_point_impl(data_point, label, -1).status();
            } else {
                // we assume that there are no concurrent operations on deleted element
                LabelType label_replaced = getExternalLabel(internal_id_replaced);
                setExternalLabel(internal_id_replaced, label);

                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                label_lookup_.erase(label_replaced);
                label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                unmarkDeletedInternal(internal_id_replaced);
                updatePoint(data_point, internal_id_replaced, 1.0);
            }
            return turbo::OkStatus();
        }


        void updatePoint(const void *dataPoint, LocationType internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            LocationType entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<LocationType> sCand;
                std::unordered_set<LocationType> sNeigh;
                std::vector<LocationType> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto &&elOneHop: listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<LocationType> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto &&elTwoHop: listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto &&neigh: sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() -
                                                                                    1;  // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(hnsw_conf.ef_construction, size);
                    for (auto &&cand: sCand) {
                        if (cand == neigh)
                            continue;

                        DistanceType distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand),
                                                             dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock<std::mutex> lock(link_list_locks_[neigh]);
                        LocationType *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        LocationType *data = (LocationType *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        }


        void repairConnectionsForUpdate(
                const void *dataPoint,
                LocationType entryPointInternalId,
                LocationType dataPointInternalId,
                int dataPointLevel,
                int maxLevel) {
            LocationType currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                DistanceType curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj, level);
                        int size = getListCount(data);
                        LocationType *datal = (LocationType *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            LocationType cand = datal[i];
                            DistanceType d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(
                                fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_),
                                entryPointInternalId);
                        if (filteredTopCandidates.size() > hnsw_conf.ef_construction)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level,
                                                        true);
                }
            }
        }


        std::vector<LocationType> getConnectionsWithLock(LocationType internalId, int level) {
            std::unique_lock<std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<LocationType> result(size);
            LocationType *ll = (LocationType *) (data + 1);
            memcpy(result.data(), ll, size * sizeof(LocationType));
            return result;
        }


        turbo::Result<LocationType> add_point_impl(const void *data_point, LabelType label, int level) {
            LocationType cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock<std::mutex> lock_table(label_lookup_lock);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    LocationType existingInternalId = search->second;
                    if (hnsw_conf.allow_replace_deleted) {
                        if (isMarkedDeleted(existingInternalId)) {
                            return turbo::invalid_argument_error(
                                    "Can't use add point to update deleted elements if replacement of deleted elements is enabled.");
                        }
                    }
                    lock_table.unlock();

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= core_conf.max_elements) {
                    return turbo::out_of_range_error("The number of elements exceeds the specified limit");
                }

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            LocationType currObj = enterpoint_node_;
            LocationType enterpoint_copy = enterpoint_node_;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(LabelType));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr) {
                    return turbo::out_of_range_error("Not enough memory: add point failed to allocate linklist");
                }
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed) currObj != -1) {
                if (curlevel < maxlevelcopy) {
                    DistanceType curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj, level);
                            int size = getListCount(data);

                            LocationType *datal = (LocationType *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                LocationType cand = datal[i];
                                if (cand < 0 || cand > core_conf.max_elements) {
                                    return turbo::internal_error("cand error");
                                }
                                DistanceType d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0) {
                        return turbo::internal_error("Level error");
                    }

                    std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(
                                fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_),
                                enterpoint_copy);
                        if (top_candidates.size() > hnsw_conf.ef_construction)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }
            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            // Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        }

        template<bool has_deletions, bool collect_metrics = false>
        turbo::Status search_impl(LocationType ep_id, SearchContext&context, size_t ef, MaxResultQueue &queue) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            MinResultQueue candidate_set;
            auto data_point = context.get_query();
            DistanceType lowerBound;
            auto ep_label = getExternalLabel(ep_id);
            if ((!has_deletions || !isMarkedDeleted(ep_id)) && !context.is_exclude(ep_label)) {
                DistanceType dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                queue.emplace(dist, ep_label, ep_id);
                candidate_set.emplace(dist, ep_label, ep_id);
            } else {
                lowerBound = std::numeric_limits<DistanceType>::max();
                candidate_set.emplace(lowerBound, ep_label, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {
                auto current_node_pair = candidate_set.top();

                if ((-current_node_pair.distance) > lowerBound &&
                    (queue.size() == ef || (!context.has_condition() && !has_deletions))) {
                    break;
                }
                candidate_set.pop();

                LocationType current_node_id = current_node_pair.location;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((LocationType *) data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);  ////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        visited_array[candidate_id] = visited_array_tag;
                        auto candidate_label = getExternalLabel(candidate_id);
                        char *currObj1 = (getDataByInternalId(candidate_id));
                        DistanceType dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (queue.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(dist, candidate_label, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().location * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                         _MM_HINT_T0);  ////////////////////////
#endif
                            if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&!context.is_exclude(candidate_label)) {
                                queue.emplace(dist, candidate_label, candidate_id);
                            }

                            if (queue.size() > ef)
                                queue.pop();

                            if (!queue.empty())
                                lowerBound = queue.top().distance;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return turbo::OkStatus();
        }

        turbo::Status search(SearchContext &context) override {
            context.schedule_time = turbo::Time::current_time();
            if (cur_element_count == 0) {
                context.end_time = turbo::Time::current_time();
                return turbo::OkStatus();
            }

            auto query_data = context.get_query();

            LocationType currObj = enterpoint_node_;
            DistanceType curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    LocationType *datal = (LocationType *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        LocationType cand = datal[i];
                        if (cand < 0 || cand > core_conf.max_elements) {
                            context.end_time = turbo::Time::current_time();
                            return turbo::internal_error("cand error");
                        }
                        DistanceType d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            MaxResultQueue top_candidates;
            if (num_deleted_) {
                auto rs = search_impl<true, true>(
                        currObj, context, std::max(ef_, static_cast<size_t>(context.top_k)), top_candidates);
                if(!rs.ok()) {
                    context.end_time = turbo::Time::current_time();
                    return rs;
                }
            } else {
                auto rs = search_impl<false, true>(
                        currObj, context, std::max(ef_, static_cast<size_t>(context.top_k)), top_candidates);
                if(!rs.ok()) {
                    context.end_time = turbo::Time::current_time();
                    return rs;
                }
            }

            while (top_candidates.size() > context.top_k) {
                top_candidates.pop();
            }
            if(context.reverse_result) {
                context.results.reserve(top_candidates.size());
                while (!top_candidates.empty()) {
                    auto rez = top_candidates.top();
                    context.results.emplace_back(rez.distance, rez.label, context.with_location ? rez.location: 0);
                    top_candidates.pop();
                }
            } else {
                context.results.resize(top_candidates.size());
                for(int i = top_candidates.size() - 1; i >= 0; i--) {
                    auto rez = top_candidates.top();
                    context.results[i] = ResultEntity(rez.distance, rez.label, context.with_location ? rez.location: 0);
                    top_candidates.pop();
                }
            }
            context.end_time = turbo::Time::current_time();
            return turbo::OkStatus();
        }

        std::priority_queue<std::pair<DistanceType, LabelType >>
        searchKnn(const void *query_data, size_t k, BaseFilterFunctor *isIdAllowed = nullptr) const {
            std::priority_queue<std::pair<DistanceType, LabelType >> result;
            if (cur_element_count == 0) return result;

            LocationType currObj = enterpoint_node_;
            DistanceType curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    LocationType *datal = (LocationType *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        LocationType cand = datal[i];
                        if (cand < 0 || cand > core_conf.max_elements)
                            throw std::runtime_error("cand error");
                        DistanceType d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<DistanceType, LocationType>, std::vector<std::pair<DistanceType, LocationType>>, CompareByFirst> top_candidates;
            if (num_deleted_) {
                top_candidates = searchBaseLayerST<true, true>(
                        currObj, query_data, std::max(ef_, k), isIdAllowed);
            } else {
                top_candidates = searchBaseLayerST<false, true>(
                        currObj, query_data, std::max(ef_, k), isIdAllowed);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<DistanceType, LocationType> rez = top_candidates.top();
                result.push(std::pair<DistanceType, LabelType>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        }


        void checkIntegrity() {
            int connections_checked = 0;
            std::vector<int> inbound_connections_num(cur_element_count, 0);
            for (int i = 0; i < cur_element_count; i++) {
                for (int l = 0; l <= element_levels_[i]; l++) {
                    LocationType *ll_cur = get_linklist_at_level(i, l);
                    int size = getListCount(ll_cur);
                    LocationType *data = (LocationType *) (ll_cur + 1);
                    std::unordered_set<LocationType> s;
                    for (int j = 0; j < size; j++) {
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert(data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;
                    }
                    assert(s.size() == size);
                }
            }
            if (cur_element_count > 1) {
                int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
                for (int i = 0; i < cur_element_count; i++) {
                    assert(inbound_connections_num[i] > 0);
                    min1 = std::min(inbound_connections_num[i], min1);
                    max1 = std::max(inbound_connections_num[i], max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";
        }
    };
}  // namespace phekda
