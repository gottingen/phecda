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

namespace phekda {

    turbo::Status HnswIndex::initialize(const IndexConfig &config) {
        if(init_type_ != IndexInitializationType::INIT_NONE) {
            return turbo::OkStatus();
        }
        turbo::MutexLock lock(&init_mutex_);
        if(init_type_ != IndexInitializationType::INIT_NONE) {
            return turbo::OkStatus();
        }
        HnswlibConfig hnswlib_config;
        try{
            hnswlib_config = std::any_cast<HnswlibConfig>(config.index_conf);
        } catch (const std::bad_any_cast &e) {
            return turbo::invalid_argument_error("index_conf is not HnswlibConfig");
        }
        if(config.core.dimension == 0) {
            return turbo::invalid_argument_error("dimension should not be 0");
        }

        switch (config.core.metric) {
            case MetricType::METRIC_L2:
                space_ = std::make_unique<L2Space>(config.core.dimension);
                break;
            case MetricType::METRIC_IP:
                space_ = std::make_unique<InnerProductSpace>(config.core.dimension);
                break;
            case MetricType::METRIC_COSINE:
                return turbo::invalid_argument_error("unsupported metric type");
            case MetricType::METRIC_NONE:
                return turbo::invalid_argument_error("unsupported metric type");
            default:
                return turbo::invalid_argument_error("unsupported metric type");
        };
        if(!space_) {
            return turbo::invalid_argument_error("unsupported metric type");
        }
        if(config.core.index_type == IndexType::INDEX_HNSWLIB) {
            alg_ = std::make_unique<HierarchicalNSW>();
        } else if(config.core.index_type == IndexType::INDEX_HNSW_FLAT) {
            alg_ = std::make_unique<BruteforceSearch>();
        } else {
            return turbo::invalid_argument_error("unsupported index type");
        }
        hnswlib_config.space = space_.get();
        auto rs = alg_->initialize(config.core, hnswlib_config);
        if(!rs.ok()) {
            return rs;
        }
        init_type_ = IndexInitializationType::INIT_INIT;
        return turbo::OkStatus();
    }

    turbo::Status HnswIndex::add_vector(turbo::Nonnull<const uint8_t *> data, LabelType label, std::any write_conf) {
        if(init_type_ == IndexInitializationType::INIT_NONE) {
            return turbo::invalid_argument_error("index not initialized");
        }
        HnswlibWriteConfig hnswlib_write_conf{false};
        try{
            if(write_conf.has_value()) {
                hnswlib_write_conf = std::any_cast<HnswlibWriteConfig>(write_conf);
            }
        } catch (const std::bad_any_cast &e) {
            return turbo::invalid_argument_error("write_conf is not HnswlibWriteConfig");
        }
        return alg_->addPoint(data, label, hnswlib_write_conf);
    }

    turbo::Status HnswIndex::add_vectors(turbo::Nonnull<const uint8_t *> data, turbo::Nonnull<const LabelType *> labels, uint32_t num,
                std::any write_conf) {
        if(init_type_ == IndexInitializationType::INIT_NONE) {
            return turbo::invalid_argument_error("index not initialized");
        }
        HnswlibWriteConfig hnswlib_write_conf{false};
        try{
            if(write_conf.has_value()) {
                hnswlib_write_conf = std::any_cast<HnswlibWriteConfig>(write_conf);
            }
        } catch (const std::bad_any_cast &e) {
            return turbo::invalid_argument_error("write_conf is not HnswlibWriteConfig");
        }
        auto size = space_->get_data_size();
        for(uint32_t i = 0; i < num; ++i) {
            auto rs = alg_->addPoint(data + i * size, labels[i], hnswlib_write_conf);
            if(!rs.ok()) {
                return rs;
            }
        }
        return turbo::OkStatus();
    }

    turbo::Status HnswIndex::get_vector(LabelType label, turbo::Nonnull<uint8_t *> data) {
        return alg_->getVector(label, data);
    }
    turbo::Status
    HnswIndex::get_vectors(turbo::Nonnull<const LabelType *> labels, uint32_t num, turbo::Nonnull<uint8_t *> data) {
        auto size = space_->get_data_size();
        for(uint32_t i = 0; i < num; ++i) {
            auto rs = alg_->getVector(labels[i], data + i * size);
            if(!rs.ok()) {
                return rs;
            }
        }
        return turbo::OkStatus();
    }

    turbo::Status HnswIndex::search(SearchContext &context) {
        return alg_->search(context);
    }

    turbo::Status HnswIndex::lazy_delete(LabelType label) {
        return alg_->markDelete(label);
    }

    turbo::Result<ConsolidationReport> HnswIndex::consolidate(const std::any &conf) {
        return turbo::OkStatus();
    }

    LabelType HnswIndex::snapshot_id() const {
        return alg_->snapshot_id();
    }

    turbo::Status HnswIndex::save(LabelType snapshot_id, const std::string &path, const std::any &save_conf) {
        return alg_->saveIndex(path, snapshot_id);
    }

    turbo::Status HnswIndex::load(const std::string &path, const IndexConfig &config) {
        turbo::MutexLock lock(&init_mutex_);
        if(init_type_ != IndexInitializationType::INIT_NONE) {
            return turbo::already_exists_error("index already initialized, can not load");
        }
        auto core_config = config.core;
        HnswlibConfig hnswlib_config;
        try{
            hnswlib_config = std::any_cast<HnswlibConfig>(config.index_conf);
        } catch (const std::bad_any_cast &e) {
            return turbo::invalid_argument_error("index_conf is not HnswlibConfig");
        }
        init_type_ = IndexInitializationType::INIT_LOAD;
        return alg_->loadIndex(path, core_config, hnswlib_config);
    }

    CoreConfig HnswIndex::get_core_config() const {
        if(init_type_ == IndexInitializationType::INIT_NONE) {
            return CoreConfig();
        }
        return alg_->get_core_config();
    }

    IndexConfig HnswIndex::get_index_config() const {
        if (init_type_ == IndexInitializationType::INIT_NONE) {
            return IndexConfig();
        }
        IndexConfig config;
        config.core = alg_->get_core_config();
        config.index_conf = alg_->get_index_config();
        return config;

    }
}  // namespace phekda