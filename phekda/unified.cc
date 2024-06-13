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
// Created by jeff on 24-6-12.
//
#include <phekda/unified.h>
#include <phekda/hnswlib/index.h>

namespace phekda {

    SearchContext UnifiedIndex::create_search_context() const {
        SearchContext context;
        auto cc = get_core_config();
        context.metric_type = cc.metric;
        context.index_type = cc.index_type;
        context.data_type = cc.data;
        context.dimension = cc.dimension;
        context.data_size = cc.dimension * data_type_size(cc.data);
        return context;
    }

    UnifiedIndex* UnifiedIndex::create_index(IndexType type) {
        switch (type) {
            case IndexType::INDEX_HNSWLIB:
            case IndexType::INDEX_HNSW_FLAT:
                return new HnswIndex();
            default:
                return nullptr;
        }
    }
}  // namespace phekda