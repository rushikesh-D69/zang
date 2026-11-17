idf_component_register(SRCS "board.c" "board_pin_table.c"
                       INCLUDE_DIRS "include"
                       REQUIRES esp_driver_i2c
                       PRIV_REQUIRES esp_driver_gpio esp_adc)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
