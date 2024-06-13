# phekda

phekda is a unified interface for various vector search engines. It provides a simple and unified interface to search for
similar vectors in different vector search engines. It supports the following vector search engines:

* hnswlib

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