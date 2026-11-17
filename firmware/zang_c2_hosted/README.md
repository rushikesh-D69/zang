# zang_c2_hosted: Wi-Fi co-processor firmware (ESP32-C2 / ESP8684)

This is the ESP-Hosted-MCU **slave** (`network_adapter`). It is vendored from `espressif/esp_hosted` 2.13.1
(`slave/`, Apache-2.0, see `LICENSE.esp_hosted`) and is unchanged apart from:
- `sdkconfig.defaults.zang`: the board v0.0.1 overlay. UART transport on the C2's UART0 (TX GPIO20, RX GPIO19),
  3 Mbaud, console off.
- `CMakeLists.txt`: loads that overlay after the vendored defaults.

The P4 side is `../zang_p4` (`CONFIG_ESP_HOSTED_UART_*` in its `sdkconfig.defaults`). Baud rate and pins must match.

```bash
source ~/.espressif/tools/activate_idf_v5.5.6.sh
idf.py set-target esp32c2
idf.py build -- -j2
```

## Flashing the C2

Today: hold the C2 in download mode and flash it over its UART0 pads with a USB-UART adapter. Hold BOOT_C2 (P4 GPIO28)
low while you pulse EN_C2 (P4 GPIO29). For example, run a P4 build that calls `net_c2_reset(true)` and keeps its own
UART pins high-Z, then attach the adapter to the C2's TX/RX test points.

Later option: the P4 owns BOOT_C2, EN_C2 and the C2's UART0, so it can flash the C2 itself with
[esp-serial-flasher](https://github.com/espressif/esp-serial-flasher). It would take the image from the SD card
(e.g. `/sdcard/fw/c2.bin`) or receive it over ZLink TCP. `net_c2_reset(true)` already handles the boot-mode sequence.

To update the vendored slave: copy `slave/` from a newer `managed_components/espressif__esp_hosted` and keep the
P4's `espressif/esp_hosted` version the same.
