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
// Created by jeff on 24-6-13.
//

#include <turbo/utility/status.h>
#include <turbo/container/roaring.h>
#include <phekda/core/search_condition.h>
#include <turbo/container/span.h>
#include <string>

namespace phekda {

    class BitmapCondition : public SearchCondition {
    public:
        BitmapCondition() = default;

        turbo::Status exclude(turbo::span<LabelType> labels);

        turbo::Status exclude(LabelType label);

        turbo::Status remove_exclude(LabelType label);

        turbo::Status reset();

        turbo::Status load(turbo::span<const char> data);

        turbo::Status save(std::vector<char> &data) const;

        bool is_exclude(LabelType label) const override {
            return bitmap_.contains(label);
        }

        void printf() const;

    private:
        turbo::Roaring bitmap_;
    };

    inline std::ostream &operator<<(std::ostream &os, const BitmapCondition &condition) {
        condition.printf();
        return os;
    }
}  // namespace phekda

