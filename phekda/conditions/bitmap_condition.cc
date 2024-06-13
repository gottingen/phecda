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

#include <phekda/conditions/bitmap_condition.h>

namespace phekda {

    turbo::Status BitmapCondition::exclude(turbo::span<LabelType> labels) {
        for (auto label : labels) {
            bitmap_.add(label);
        }
        return turbo::OkStatus();
    }

    turbo::Status BitmapCondition::exclude(LabelType label) {
        bitmap_.add(label);
        return turbo::OkStatus();
    }

    turbo::Status BitmapCondition::remove_exclude(LabelType label) {
        bitmap_.remove(label);
        return turbo::OkStatus();
    }

    turbo::Status BitmapCondition::reset() {
        turbo::Roaring dummy;
        bitmap_.swap(dummy);
        return turbo::OkStatus();
    }

    turbo::Status BitmapCondition::load(turbo::span<const char> data) {
        try {
            bitmap_ = turbo::Roaring::read(data.data(), false);
        } catch (const std::exception &e) {
            return turbo::data_loss_error("load bitmap condition failed %s", e.what());
        }
        return turbo::OkStatus();
    }

    turbo::Status BitmapCondition::save(std::vector<char> &data) const {
        try {
            std::size_t bitmap_size = bitmap_.getSizeInBytes(false);
            data.resize(bitmap_size);
            std::size_t size_in_bytes = bitmap_.write(data.data(), false);
            data.resize(size_in_bytes);
        } catch (const std::exception &e) {
            return turbo::data_loss_error("save bitmap condition failed %s", e.what());
        }
        return turbo::OkStatus();
    }

    void BitmapCondition::printf() const {
        bitmap_.printf();
    }

}  // namespace phekda