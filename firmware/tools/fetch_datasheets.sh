#!/usr/bin/env bash
# Re-download the datasheets listed in docs/datasheets/INDEX.md. Usage: tools/fetch_datasheets.sh
set -u
cd "$(dirname "$0")/../docs/datasheets"
UA="Mozilla/5.0 (X11; Linux x86_64) Firefox/140.0"
get() { # file url...
  local f=$1; shift
  for u in "$@"; do
    if wget -q -T 30 -t 2 -U "$UA" -O "$f.part" "$u" && head -c 4 "$f.part" | grep -q '%PDF'; then
      mv "$f.part" "$f"; echo "OK   $f  <- $u"; return 0
    fi
  done
  rm -f "$f.part"; echo "FAIL $f"; return 1
}
E=https://www.espressif.com/sites/default/files/documentation
get esp32-p4_datasheet_en.pdf $E/esp32-p4_datasheet_en.pdf
get esp32-p4_technical_reference_manual_en.pdf $E/esp32-p4_technical_reference_manual_en.pdf
get esp32-p4_hardware_design_guidelines_en.pdf \
  https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32p4/esp-hardware-design-guidelines-en-master-esp32p4.pdf
get esp32-c2_esp8684_datasheet_en.pdf $E/esp8684_datasheet_en.pdf $E/esp32-c2_datasheet_en.pdf
get esp32-c2_technical_reference_manual_en.pdf $E/esp8684_technical_reference_manual_en.pdf $E/esp32-c2_technical_reference_manual_en.pdf
get esp32-c2_hardware_design_guidelines_en.pdf \
  https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32c2/esp-hardware-design-guidelines-en-master-esp32c2.pdf
get ST7789V.pdf https://www.newhavendisplay.com/appnotes/datasheets/LCDs/ST7789V.pdf https://www.rhydolabz.com/documents/33/ST7789.pdf
get ILI9341.pdf https://cdn-shop.adafruit.com/datasheets/ILI9341.pdf
get GC9A01A.pdf https://files.waveshare.com/wiki/common/GC9A01A.pdf https://cdn.shopify.com/s/files/1/0798/6901/8399/files/GC9A01.pdf
# Board v0.0.1 parts
get OV2640.pdf https://dl.espressif.com/dl/schematics/Camera_OV2640.pdf https://files.waveshare.com/wiki/common/OV2640DS_en.pdf
get NS4168_cn.pdf https://atta.szlcsc.com/upload/public/pdf/source/20240722/6A2F478F91F6E02E759B805C93276334.pdf
# Future IMU candidate (no IMU on v0.0.1)
get QMI8658.pdf "https://www.qstcorp.com/upload/pdf/202202/QMI8658C%20datasheet%20rev%200.9.pdf" \
  https://files.waveshare.com/upload/5/5f/QMI8658C.pdf
# Servo: no DS-R003G datasheet is published. Keep the closest DSpower product pages (HTML) as reference.
page() { wget -q -T 30 -t 2 -U "$UA" -O "$1" "$2" && echo "OK   $1  <- $2" || { rm -f "$1"; echo "FAIL $1"; }; }
page DSpower_DS-R003D_product_page.html https://www.dspowerservo.com/35kg-steel-gear-rs485-serial-bus-digital-servo-ds-r003d-product/
page DSpower_DS-R003E_product_page.html http://jw.dspowerservo.com/dual-axis-35kg-high-torque-anti-burn-servo-ds-r003e-product/
