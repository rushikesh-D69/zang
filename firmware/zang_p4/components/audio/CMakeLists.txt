idf_component_register(SRCS "audio.c" "wav.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board esp_driver_i2s esp_driver_gpio esp_timer zang_common)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
