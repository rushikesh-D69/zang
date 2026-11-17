idf_component_register(SRCS "ds_proto.c" "servo_bus.c"
                       INCLUDE_DIRS "include"
                       REQUIRES esp_driver_uart
                       PRIV_REQUIRES board esp_timer esp_driver_gpio)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
