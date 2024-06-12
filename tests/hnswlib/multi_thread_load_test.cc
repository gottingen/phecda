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

#include <thread>
#include <chrono>


int main() {
    std::cout << "Running multithread load test" << std::endl;
    int d = 16;
    int max_elements = 1000;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    phekda::CoreConfig core_config;
    core_config.max_elements = 2 * max_elements;
    core_config.dimension = d;
    core_config.data = phekda::DataType::FLOAT32;
    core_config.metric = phekda::MetricType::METRIC_L2;
    core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;
    phekda::L2Space space(d);
    phekda::HnswlibConfig config;
    config.M = 16;
    config.ef_construction = 200;
    config.random_seed = 123;
    config.allow_replace_deleted = true;
    config.space = &space;
    phekda::HierarchicalNSW* alg_hnsw = new phekda::HierarchicalNSW();
    auto rs = alg_hnsw->initialize(core_config, config);
    if (!rs.ok()) {
        std::cerr << "Error: " << rs.message() << std::endl;
        return 1;
    }

    std::cout << "Building index" << std::endl;
    int num_threads = 40;
    int num_labels = 10;

    int num_iterations = 10;
    int start_label = 0;

    // run threads that will add elements to the index
    // about 7 threads (the number depends on num_threads and num_labels)
    // will add/update element with the same label simultaneously
    while (true) {
        // add elements by batches
        std::uniform_int_distribution<> distrib_int(start_label, start_label + num_labels - 1);
        std::vector<std::thread> threads;
        for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.push_back(
                std::thread(
                    [&] {
                        for (int iter = 0; iter < num_iterations; iter++) {
                            std::vector<float> data(d);
                            phekda::LabelType label = distrib_int(rng);
                            for (int i = 0; i < d; i++) {
                                data[i] = distrib_real(rng);
                            }
                            alg_hnsw->addPoint(data.data(), label);
                        }
                    }
                )
            );
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (alg_hnsw->cur_element_count > max_elements - num_labels) {
            break;
        }
        start_label += num_labels;
    }

    // insert remaining elements if needed
    for (phekda::LabelType label = 0; label < max_elements; label++) {
        auto search = alg_hnsw->label_lookup_.find(label);
        if (search == alg_hnsw->label_lookup_.end()) {
            std::cout << "Adding " << label << std::endl;
            std::vector<float> data(d);
            for (int i = 0; i < d; i++) {
                data[i] = distrib_real(rng);
            }
            alg_hnsw->addPoint(data.data(), label);
        }
    }

    std::cout << "Index is created" << std::endl;

    bool stop_threads = false;
    std::vector<std::thread> threads;

    // create threads that will do markDeleted and unmarkDeleted of random elements
    // each thread works with specific range of labels
    std::cout << "Starting markDeleted and unmarkDeleted threads" << std::endl;
    num_threads = 20;
    int chunk_size = max_elements / num_threads;
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(
            std::thread(
                [&, thread_id] {
                    std::uniform_int_distribution<> distrib_int(0, chunk_size - 1);
                    int start_id = thread_id * chunk_size;
                    std::vector<bool> marked_deleted(chunk_size);
                    while (!stop_threads) {
                        int id = distrib_int(rng);
                        phekda::LabelType label = start_id + id;
                        if (marked_deleted[id]) {
                            alg_hnsw->unmarkDelete(label);
                            marked_deleted[id] = false;
                        } else {
                            alg_hnsw->markDelete(label);
                            marked_deleted[id] = true;
                        }
                    }
                }
            )
        );
    }

    // create threads that will add and update random elements
    std::cout << "Starting add and update elements threads" << std::endl;
    num_threads = 20;
    std::uniform_int_distribution<> distrib_int_add(max_elements, 2 * max_elements - 1);
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        threads.push_back(
            std::thread(
                [&] {
                    std::vector<float> data(d);
                    while (!stop_threads) {
                        phekda::LabelType label = distrib_int_add(rng);
                        for (int i = 0; i < d; i++) {
                            data[i] = distrib_real(rng);
                        }
                        alg_hnsw->addPoint(data.data(), label);
                        std::vector<float> data = alg_hnsw->getDataByLabel<float>(label);
                        float max_val = *max_element(data.begin(), data.end());
                        // never happens but prevents compiler from deleting unused code
                        if (max_val > 10) {
                            throw std::runtime_error("Unexpected value in data");
                        }
                    }
                }
            )
        );
    }

    std::cout << "Sleep and continue operations with index" << std::endl;
    int sleep_ms = 60 * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    stop_threads = true;
    for (auto &thread : threads) {
        thread.join();
    }
    
    std::cout << "Finish" << std::endl;
    return 0;
}
