# zlink.c is pure C (host-tested); zlink_server.c is the lwIP socket side.
idf_component_register(SRCS "zlink.c" "zlink_server.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES lwip zang_common esp_timer)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
