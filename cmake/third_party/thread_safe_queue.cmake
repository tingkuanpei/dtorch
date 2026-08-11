# Copyright 2026 The DTorch Authors. All rights reserved.
#
# Author: Tingkuan Pei(contact: peitingkuan@163.com)

set(THREAD_SAFE_QUEUE_INCLUDE_DIR "")

if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
    set(CAMERON314_CONCURRENTQUEUE_URL "${DTORCH_LOCAL_URL_DIR}/concurrentqueue-1.0.3.tar.gz")
    if(NOT EXISTS ${CAMERON314_CONCURRENTQUEUE_URL})
        set(ErrorMessage "Can't find local file: ${CAMERON314_CONCURRENTQUEUE_URL}\n")
        string(APPEND ErrorMessage "Please download it manual or set DTORCH_THIRD_PARTY_USE_LOCAL_URL OFF.")
        message(FATAL_ERROR "${ErrorMessage}")
    endif()
else()
    set(CAMERON314_CONCURRENTQUEUE_URL "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.3.tar.gz")
endif()
set(CAMERON314_CONCURRENTQUEUE_URL_SHA256 "eb37336bf9ae59aca7b954db3350d9b30d1cab24b96c7676f36040aa76e915e8")

set(FETCHCONTENT_BASE_DIR   ${CMAKE_BINARY_DIR}/third_party/cameron314_concurrentqueue)
FetchContent_Declare(
    cameron314_concurrentqueue
    URL ${CAMERON314_CONCURRENTQUEUE_URL}
    URL_HASH SHA256=${CAMERON314_CONCURRENTQUEUE_URL_SHA256}
)
FetchContent_GetProperties(cameron314_concurrentqueue)

if(NOT cameron314_concurrentqueue_POPULATED)
    message(STATUS "Fetching cameron314_concurrentqueue")
    FetchContent_MakeAvailable(cameron314_concurrentqueue)
    message(STATUS "Fetching cameron314_concurrentqueue - done")
endif()

list(APPEND THREAD_SAFE_QUEUE_INCLUDE_DIR      "${cameron314_concurrentqueue_SOURCE_DIR}")

#-----------------------------------------------------------------------------------------------------------------------

if(DTORCH_THIRD_PARTY_USE_LOCAL_URL)
    set(CAMERON314_READERWRITERQUEUE_URL "${DTORCH_LOCAL_URL_DIR}/readerwriterqueue-1.0.6.tar.gz")
    if(NOT EXISTS ${CAMERON314_READERWRITERQUEUE_URL})
        set(ErrorMessage "Can't find local file: ${CAMERON314_READERWRITERQUEUE_URL}\n")
        string(APPEND ErrorMessage "Please download it manual or set DTORCH_THIRD_PARTY_USE_LOCAL_URL OFF.")
        message(FATAL_ERROR "${ErrorMessage}")
    endif()
else()
    set(CAMERON314_READERWRITERQUEUE_URL "https://github.com/cameron314/readerwriterqueue/archive/refs/tags/v1.0.6.tar.gz")
endif()
set(CAMERON314_READERWRITERQUEUE_URL_SHA256 "fc68f55bbd49a8b646462695e1777fb8f2c0b4f342d5e6574135211312ba56c1")

set(FETCHCONTENT_BASE_DIR   ${CMAKE_BINARY_DIR}/third_party/cameron314_readerwriterqueue)
FetchContent_Declare(
    cameron314_readerwriterqueue
    URL ${CAMERON314_READERWRITERQUEUE_URL}
    URL_HASH SHA256=${CAMERON314_READERWRITERQUEUE_URL_SHA256}
)
FetchContent_GetProperties(cameron314_readerwriterqueue)

if(NOT cameron314_readerwriterqueue_POPULATED)
    message(STATUS "Fetching cameron314_readerwriterqueue")
    FetchContent_MakeAvailable(cameron314_readerwriterqueue)
    message(STATUS "Fetching cameron314_readerwriterqueue - done")
endif()

list(APPEND THREAD_SAFE_QUEUE_INCLUDE_DIR      "${cameron314_readerwriterqueue_SOURCE_DIR}")
