idf_component_register(SRCS "storage.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board fatfs sdmmc esp_driver_sdmmc esp_driver_sdspi esp_driver_spi)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
