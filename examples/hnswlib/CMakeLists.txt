#
# Copyright (C) 2024 EA group inc.
# Author: Jeff.li lijippy@163.com
# All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

carbin_cc_binary(
        NAMESPACE phekda
        NAME mt_filter_example
        SOURCES mt_filter_example.cc
        CXXOPTS ${CARBIN_CXX_OPTIONS}
        LINKS phekda::phekda ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE phekda
        NAME filter_example
        SOURCES filter_example.cc
        CXXOPTS ${CARBIN_CXX_OPTIONS}
        LINKS phekda::phekda ${CARBIN_DEPS_LINK}
)