# phekda

phekda is a unified interface for various vector search engines. It provides a simple and unified interface to search for
similar vectors in different vector search engines. It supports the following vector search engines:

* hnswlib

# the features of pherda:

* real time
* no exception of unified interface
* support filter

filter is a feature that can be used to filter out some vectors that do not meet the requirements before the search engine is used to search for similar vectors. This can greatly reduce the number of vectors that need to be searched, and improve the search efficiency.
whe provide a default bitmap filter, will meet most of the requirements. If you have special requirements, you can implement your own filter.
bitmap filter see [bitmap_condition](phekda/conditions/bitmap_condition.h)

see [examples](examples/hnswlib/mt_filter_example.cc)

```c++
        phekda::HnswlibConfig config;
        config.M = 16;
        config.ef_construction = 200;
        config.random_seed = 123;
        config.allow_replace_deleted = true;

        phekda::IndexConfig flat_index_config;
        flat_index_config.core = flat_config;
        flat_index_config.index_conf = config;

        auto *alg_brute = phekda::UnifiedIndex::create_index(flat_config.index_type);
        auto rs = alg_brute->initialize(flat_index_config);
        if (!rs.ok()) {
            std::cout << "Failed to initialize FLAT HNSW: " << rs << std::endl;
            exit(1);
        }
        
```