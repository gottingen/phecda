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
#include <phekda/unified.h>
#include <thread>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
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


// Filter that allows labels divisible by divisor
class PickDivisibleIds: public phekda::SearchCondition {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool is_exclude(phekda::LabelType label) const override {
        return label % divisor != 0;
    }
};


int main() {

    int num_threads = 20;       // Number of threads for operations with index

    phekda::CoreConfig core_config;
    core_config.max_elements = 10000;
    core_config.dimension = 16;
    core_config.data = phekda::DataType::FLOAT32;
    core_config.metric = phekda::MetricType::METRIC_L2;
    core_config.index_type = phekda::IndexType::INDEX_HNSWLIB;

    phekda::HnswlibConfig config;
    config.M = 16;
    config.ef_construction = 200;
    config.random_seed = 123;
    config.allow_replace_deleted = true;

    auto alg_hnsw = phekda::UnifiedIndex::create_index(core_config.index_type);
    auto rs =alg_hnsw->initialize({core_config, config});
    if(!rs.ok()) {
        std::cout << "Error: " << rs.message() << std::endl;
        return 1;
    }

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[core_config.dimension * core_config.max_elements];
    for (int i = 0; i < core_config.dimension * core_config.max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    ParallelFor(0, core_config.max_elements, num_threads, [&](size_t row, size_t threadId) {
        auto rs = alg_hnsw->add_vector((const uint8_t *)(data + core_config.dimension * row), row);
        if(!rs.ok()) {
            std::cout << "Error: " << rs.message() << std::endl;
            exit(1);
        }
    });

    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    // Query the elements for themselves with filter and check returned labels
    int k = 10;
    std::vector<phekda::LabelType> neighbors(core_config.max_elements * k);
    LOG(INFO)<<"start multi-thread search with filter...";
    ParallelFor(0, core_config.max_elements, num_threads, [&](size_t row, size_t threadId) {
        auto context = alg_hnsw->create_search_context();
        context.with_query(reinterpret_cast<const uint8_t *>(data + core_config.dimension * row)).with_top_k(k).with_condition(&pickIdsDivisibleByTwo);
        auto rs = alg_hnsw->search(context);
        if(!rs.ok()) {
            std::cout << "Error: " << rs.message() << std::endl;
            exit(1);
        }
        for (int i = 0; i < k; i++) {
            neighbors[row * k + i] = context.results[i].label;
        }
    });

    for (phekda::LabelType label: neighbors) {
        if (label % 2 == 1) std::cout << "Error: found odd label\n";
    }

    delete[] data;
    delete alg_hnsw;
    LOG(INFO)<<"done........";
    return 0;
}
