# Datasheets

To re-download them, run `tools/fetch_datasheets.sh`. Status meanings:
- **on board**: the part is on `Zang v0.0.1 Schmeatics.pdf`.
- **assumed**: firmware assumes this part; check the physical module.
- **candidate**: the part isn't decided yet.
- **not used**: kept only for reference, in `not_used/`.

| File | Part | Status | Source |
|---|---|---|---|
| `esp32-p4_datasheet_en.pdf` | ESP32-P4 main SoC (datasheet v0.7) | on board | https://www.espressif.com/sites/default/files/documentation/esp32-p4_datasheet_en.pdf |
| `esp32-p4_technical_reference_manual_en.pdf` | ESP32-P4 TRM | on board | https://www.espressif.com/sites/default/files/documentation/esp32-p4_technical_reference_manual_en.pdf |
| `esp32-p4_hardware_design_guidelines_en.pdf` | ESP32-P4 HW design guidelines | on board | https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32p4/esp-hardware-design-guidelines-en-master-esp32p4.pdf |
| `esp32-c2_esp8684_datasheet_en.pdf` | ESP8684 (ESP32-C2) Wi-Fi co-processor | on board | https://www.espressif.com/sites/default/files/documentation/esp8684_datasheet_en.pdf |
| `esp32-c2_technical_reference_manual_en.pdf` | ESP32-C2 / ESP8684 TRM | on board | https://www.espressif.com/sites/default/files/documentation/esp8684_technical_reference_manual_en.pdf |
| `esp32-c2_hardware_design_guidelines_en.pdf` | ESP32-C2 HW design guidelines | on board | https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32c2/esp-hardware-design-guidelines-en-master-esp32c2.pdf |
| `NS4168_cn.pdf` | NS4168 I2S class-D speaker amp (Nsiway, **Chinese**, V1.2) | on board | https://atta.szlcsc.com/upload/public/pdf/source/20240722/6A2F478F91F6E02E759B805C93276334.pdf (LCSC C910588) |
| `OV2640.pdf` | OV2640 2 MP DVP camera (preliminary v2.2, Espressif copy) | **assumed**: the schematic shows a 24-pin DVP module with 1.2 V DVDD; confirm the part number | https://dl.espressif.com/dl/schematics/Camera_OV2640.pdf |
| `ST7789V.pdf` | ST7789V TFT controller | candidate (firmware default) | https://www.newhavendisplay.com/appnotes/datasheets/LCDs/ST7789V.pdf |
| `GC9A01A.pdf` | GC9A01A round TFT controller (Rev 1.0 preliminary) | candidate | https://files.waveshare.com/wiki/common/GC9A01A.pdf |
| `ILI9341.pdf` | ILI9341 TFT controller | candidate | https://cdn-shop.adafruit.com/datasheets/ILI9341.pdf |
| `QMI8658.pdf` | QMI8658C 6-axis IMU (rev 0.9) | candidate for v0.0.2 (no IMU on v0.0.1) | https://www.qstcorp.com/upload/pdf/202202/QMI8658C%20datasheet%20rev%200.9.pdf |
| `DSpower_DS-R003D_product_page.html` | DS-R003D serial bus servo (RS485, 35 kg) product page | closest public reference for the **DS-R003G** | https://www.dspowerservo.com/35kg-steel-gear-rs485-serial-bus-digital-servo-ds-r003d-product/ |
| `DSpower_DS-R003E_product_page.html` | DS-R003E TTL serial bus servo product page | closest public reference for the **DS-R003G** | http://jw.dspowerservo.com/dual-axis-35kg-high-torque-anti-burn-servo-ds-r003e-product/ |
| `not_used/ES8311.pdf` | ES8311 codec | not used (board has the NS4168 and PDM mics, no codec) | https://dl.espressif.com/dl/schematics/Audio_ES8311.pdf |
| `not_used/MAX98357A.pdf` | MAX98357A I2S amp | not used (the NS4168 replaces it) | https://www.analog.com/media/en/technical-documentation/data-sheets/MAX98357A-MAX98357B.pdf |
| `not_used/INMP441.pdf` | INMP441 I2S mic | not used (board uses PDM mics) | https://microcontroller-course.web.cern.ch/Project_Material/resources/INMP441.pdf |
| `not_used/OV5647.pdf` | OV5647 MIPI camera | not used (board is DVP) | https://cdn.sparkfun.com/datasheets/Dev/RaspberryPi/ov5647_full.pdf |

## Missing

- **DS-R003G servo**: DSpower publishes no datasheet or protocol document for it. The firmware relies on the protocol
  that Zing's working firmware uses with the same DS family (see `components/servo_bus/include/ds_proto.h`; registers are
  marked CONFIRMED or ASSUMED). We still need the resolution and angle range (firmware assumes 4096 ticks over
  270°), the maximum baud rate, whether SYNC WRITE (0x83) works, and the voltage/temperature registers. Ask DSpower,
  or measure on the bench.
- **PDM microphones**: the schematic doesn't name the part. Any standard PDM MEMS mic with an L/R select pin works.
- **SC2336**: dropped; the board has no MIPI camera.
