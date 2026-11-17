# motion_math.c is pure C (host-tested). motion.c / motion_calib_nvs.c are the IDF runtime.
idf_component_register(SRCS "motion_math.c" "motion.c" "motion_calib_nvs.c"
                       INCLUDE_DIRS "include"
                       REQUIRES robot_def servo_bus
                       PRIV_REQUIRES board nvs_flash esp_timer zang_common)
include(${CMAKE_CURRENT_LIST_DIR}/../zang_common/zang_warnings.cmake)
