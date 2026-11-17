idf_component_register(SRCS "app_main.c"
                       PRIV_REQUIRES app board servo_bus motion storage display camera audio imu net zlink nvs_flash)
