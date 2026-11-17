idf_component_register(SRCS "camera.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board esp_driver_cam esp_driver_i2c esp_timer zang_common
                                     espressif__esp_cam_sensor espressif__esp_sccb_intf)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
