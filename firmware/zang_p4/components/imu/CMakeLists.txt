idf_component_register(SRCS "imu.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board esp_driver_i2c esp_timer zang_common)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
