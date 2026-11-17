# face_sdf.c, face_engine.c, faces_lib.c, canvas.c and img_asm.c are pure C (host-tested by test_host/).
idf_component_register(SRCS "display.c" "canvas.c" "face_sdf.c" "face_engine.c" "faces_lib.c" "img_asm.c" "zjpeg.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES board esp_lcd esp_driver_spi esp_driver_ledc esp_driver_jpeg esp_timer
                                     esp_hw_support zang_common
                                     espressif__esp_lcd_ili9341 espressif__esp_lcd_gc9a01)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
