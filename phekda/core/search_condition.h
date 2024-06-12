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

#include <phekda/core/defines.h>
#include <turbo/base/nullability.h>
#include <vector>

namespace phekda {

    class SearchCondition {
    public:
        virtual bool is_exclude(LabelType label) const = 0;

        virtual bool is_whitelist(LabelType label) const {
            return false;
        }

        // if the distance not in the range of [min, max], return true
        virtual bool should_stop_search(DistanceType dis) const {
            return false;
        }

        virtual bool should_explain() const {
            return false;
        }

        virtual ~SearchCondition() = default;
    };

    class CompositeSearchCondition : public SearchCondition {
    public:
        CompositeSearchCondition() = default;

        void add_condition(turbo::Nonnull<SearchCondition*> condition) {

            conditions_.push_back(std::move(condition));
        }

        bool is_exclude(LabelType label) const override {
            for (const auto &condition : conditions_) {
                if (condition->is_exclude(label)) {
                    return true;
                }
            }
            return false;
        }

        bool is_whitelist(LabelType label) const override {
            for (const auto &condition : conditions_) {
                if (condition->is_whitelist(label)) {
                    return true;
                }
            }
            return false;
        }

        bool should_stop_search(DistanceType dis) const override {
            for (const auto &condition : conditions_) {
                if (condition->should_stop_search(dis)) {
                    return true;
                }
            }
            return false;
        }

        bool should_explain() const override {
            for (const auto &condition : conditions_) {
                if (condition->should_explain()) {
                    return true;
                }
            }
            return false;
        }
    private:
        std::vector<SearchCondition*> conditions_;
    };
}  // namespace phekda

