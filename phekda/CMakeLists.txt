#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###########################################################################
# below are examples, after initialization, modify it to your own.
###########################################################################
###carbin_example
#[[
carbin_cc_library(
        NAMESPACE phekda
        NAME foo
        SOURCES
        foo.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        PLINKS
        ${CARBIN_DEPS_LINK}
        PUBLIC
)

carbin_cc_library(
        NAMESPACE phekda
        NAME object
        SOURCES
        object.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        PLINKS
        ${CARBIN_DEPS_LINK}
        PUBLIC
)

carbin_cc_library(
        NAMESPACE phekda
        NAME share
        SOURCES
        shared.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::object
        PLINKS
        ${CARBIN_DEPS_LINK}
        phekda::object
        PUBLIC
)

carbin_cc_interface(
        NAMESPACE phekda
        NAME api
        HEADERS
        api.h
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE phekda
        NAME shared_main
        SOURCES
        main.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::share
        LINKS
        ${CARBIN_DEPS_LINK}
        phekda::share
        PUBLIC
)

carbin_cc_object(
        NAMESPACE phekda
        NAME obj1
        SOURCES object1.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::object
)

carbin_cc_object(
        NAMESPACE phekda
        NAME obj2
        SOURCES object2.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::object
)

carbin_cc_library(
        NAMESPACE phekda
        NAME obm_source
        OBJECTS obj1 obj2
        SOURCES shared.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::object
        PLINKS
        ${CARBIN_DEPS_LINK}
        phekda::object
)

carbin_cc_library(
        NAMESPACE phekda
        NAME obm_no_source
        OBJECTS obj1 obj2
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        phekda::object
        PLINKS
        ${CARBIN_DEPS_LINK}
        phekda::object
)
]]

###carbin_example
set(PHEKDA_SRC
        hnswlib/index.cc
        unified.cc
        conditions/bitmap_condition.cc
)
carbin_cc_library(
        NAMESPACE phekda
        NAME phekda
        SOURCES
        ${PHEKDA_SRC}
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        PLINKS
        ${CARBIN_DEPS_LINK}
        PUBLIC
)
