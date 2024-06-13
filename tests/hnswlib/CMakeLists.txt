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

carbin_cc_test(
        NAME multi_thread_replace_test
        MODULE hnswlib
        SOURCES multi_thread_replace_test.cc
        LINKS ${CARBIN_DEPS_LINK}
        CXXOPTS ${CARBIN_CXX_OPTIONS}
        EXT
)

carbin_cc_test_ext(
        NAME multi_thread_replace_test
        MODULE hnswlib
        ALIAS 01
        PASS_EXP "finish;Finish"
)

carbin_cc_test(
        NAME multi_thread_load_test
        MODULE hnswlib
        SOURCES multi_thread_load_test.cc
        LINKS ${CARBIN_DEPS_LINK}
        CXXOPTS ${CARBIN_CXX_OPTIONS}
        EXT
)

carbin_cc_test_ext(
        NAME multi_thread_load_test
        MODULE hnswlib
        ALIAS 01
        PASS_EXP "finish;Finish"
)

carbin_cc_test(
        NAME search_knn_closer_first_test
        MODULE hnswlib
        SOURCES search_knn_closer_first_test.cc
        LINKS ${CARBIN_DEPS_LINK}
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)

carbin_cc_test(
        NAME search_knn_with_filter_test
        MODULE hnswlib
        SOURCES search_knn_with_filter_test.cc
        LINKS ${CARBIN_DEPS_LINK}
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)
#[[
carbin_cc_test(
        NAME updates_test
        MODULE hnswlib
        SOURCES updates_test.cc
        LINKS ${CARBIN_DEPS_LINK}
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)
]]

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
carbin_cc_test(
        NAME save_load_test
        MODULE hnswlib
        SOURCES save_load_test.cc
        LINKS ${CARBIN_DEPS_LINK} GTest::gtest GTest::gtest_main
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)

carbin_cc_test(
        NAME knn_with_flat_filter_test
        MODULE hnswlib
        SOURCES knn_with_flat_filter_test.cc
        LINKS ${CARBIN_DEPS_LINK} phekda::phekda GTest::gtest GTest::gtest_main
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)

carbin_cc_test(
        NAME knn_with_hnsw_filter_test
        MODULE hnswlib
        SOURCES knn_with_hnsw_filter_test.cc
        LINKS ${CARBIN_DEPS_LINK} phekda::phekda GTest::gtest GTest::gtest_main
        CXXOPTS ${CARBIN_CXX_OPTIONS}
)