# app_sm.c is pure C (host-tested); app.c is the IDF glue.
idf_component_register(SRCS "app_sm.c" "app.c" "app_media.c"
                       INCLUDE_DIRS "include"
                       REQUIRES motion audio display camera imu zlink
                       PRIV_REQUIRES board storage net esp_timer zang_common)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
