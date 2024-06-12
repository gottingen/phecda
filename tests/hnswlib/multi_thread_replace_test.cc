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


template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main() {
    std::cout << "Running multithread load test" << std::endl;
    int d = 16;
    int num_elements = 1000;
    int max_elements = 2 * num_elements;
    int num_threads = 50;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    phekda::CoreConfig core_config;
    core_config.max_elements = max_elements;
    core_config.dimension = d;
    core_config.data = phekda::DataType::FLOAT32;
    core_config.metric = phekda::MetricType::METRIC_L2;
    core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;
    phekda::L2Space space(d);

    // generate batch1 and batch2 data
    float* batch1 = new float[d * max_elements];
    for (int i = 0; i < d * max_elements; i++) {
        batch1[i] = distrib_real(rng);
    }
    float* batch2 = new float[d * num_elements];
    for (int i = 0; i < d * num_elements; i++) {
        batch2[i] = distrib_real(rng);
    }

    // generate random labels to delete them from index
    std::vector<int> rand_labels(max_elements);
    for (int i = 0; i < max_elements; i++) {
        rand_labels[i] = i;
    }
    std::shuffle(rand_labels.begin(), rand_labels.end(), rng);

    int iter = 0;
    while (iter < 200) {
        phekda::HnswlibConfig config;
        config.M = 16;
        config.ef_construction = 200;
        config.random_seed = 123;
        config.allow_replace_deleted = true;
        config.space = &space;
        phekda::HierarchicalNSW* alg_hnsw = new phekda::HierarchicalNSW();
        auto rs = alg_hnsw->initialize(core_config, config);
        if(!rs.ok()) {
            std::cout << "Failed to initialize HNSW: " << rs<< std::endl;
            exit(1);
        }

        // add batch1 data
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw->addPoint((void*)(batch1 + d * row), row);
        });

        // delete half random elements of batch1 data
        for (int i = 0; i < num_elements; i++) {
            alg_hnsw->markDelete(rand_labels[i]);
        }

        // replace deleted elements with batch2 data
        ParallelFor(0, num_elements, num_threads, [&](size_t row, size_t threadId) {
            int label = rand_labels[row] + max_elements;
            alg_hnsw->addPoint((void*)(batch2 + d * row), label, true);
        });

        iter += 1;

        delete alg_hnsw;
    }
    
    std::cout << "Finish" << std::endl;

    delete[] batch1;
    delete[] batch2;
    return 0;
}
