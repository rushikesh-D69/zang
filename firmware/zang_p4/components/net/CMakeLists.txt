idf_component_register(SRCS "net.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board esp_wifi esp_netif esp_event nvs_flash esp_driver_gpio
                                     espressif__esp_wifi_remote espressif__esp_hosted espressif__mdns zlink)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
