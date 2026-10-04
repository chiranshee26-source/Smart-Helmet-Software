# Hardware plan: parts, budget and how it fits together

_Prices checked 4 Oct 2026 on Indian stores, including GST, with links below.
Prices and stock change daily; check before ordering. Target from the IDP
document: ₹2,500–3,500 for the recommended build._

## 1. How the hardware fits together (decide this first)

The ESP32-CAM is a tiny microcontroller. It **cannot run the face-landmark
model** (mediapipe) that computes EAR; that needs a laptop or phone CPU. So
the realistic prototype architecture is:

```
 HELMET                                         LAPTOP / PHONE (same Wi-Fi or hotspot)
 ┌──────────────────────────────┐   video    ┌───────────────────────────────────────┐
 │ ESP32-CAM ── OV2640 camera   │ ─────────▶ │ this repo: EAR, PERCLOS, no-blink,    │
 │ (streams MJPEG over Wi-Fi)   │            │ sunglasses check, fusion, dashboard   │
 │                              │   pitch    │                                       │
 │ ESP32 DevKit ── MPU6050      │ ─────────▶ │ head-nod detection                    │
 │   ├─ vibration motor         │            │                                       │
 │   ├─ buzzer                  │ ◀───────── │ alert level GREEN / YELLOW / RED      │
 │   └─ OLED status display     │   alert    └───────────────────────────────────────┘
 │ 18650 battery → 5 V          │
 └──────────────────────────────┘
```

- **Why two boards:** the camera uses almost all of the ESP32-CAM's pins.
  A separate ESP32 DevKit handles the IMU, buzzer, vibration motor and OLED,
  and receives the alert level. The IDP document already lists both.
- **What changes in the software:** only the two input sources. The
  decision engine and dashboard stay as they are (`main.py` was designed for
  this swap).
- **Say this honestly in reviews:** the prototype's vision runs on a
  laptop or phone, not on the helmet. Running it on the helmet itself is
  research gap G5 / future scope (TinyML, or an ESP32-S3 with a small eye-state model).

## 2. Core parts (recommended build)

| # | Part | Qty | Why | Price checked | Where |
|---|---|---|---|---|---|
| 1 | **ESP32-CAM + ESP32-CAM-MB** USB programmer, OV2640 camera | 1 | Camera on the helmet, streams video. The MB board lets you program it over USB. | ₹478 (Robu) – ₹688 (Probots) | **Both out of stock today**: order early or check Robocraze / Amazon |
| 2 | **ESP32 DevKit V1**, 30-pin | 1 | Runs IMU + alerts + display, receives alert level | ₹399 (Robocraze, sold out today); typically ₹350–450 | Robocraze, Probots, Robu |
| 3 | **MPU6050** (GY-521) | 1 | Head pitch for nod detection | ₹199, in stock | Probots |
| 4 | **Coin vibration motor** (ERM, 3 V, 85 mA) | 1–2 | Silent YELLOW warning, felt through the helmet padding | ₹65 each, in stock | Robu |
| 5 | **Active buzzer**, 5 V | 1 | Audible RED warning | ~₹20–30 (not checked) | any store |
| 6 | **0.96″ OLED**, I2C, SSD1306 | 1 | Status display (also handy for debugging) | ₹188, in stock | Zbotic |
| 7 | **TP4056 Type-C** charger with protection | 1 | Charges the 18650 safely (over-charge and over-discharge protection) | ₹49, in stock | Probots |
| 8 | **MT3608 boost** converter | 1 | 3.7 V battery → 5 V for both boards (set to 5.0 V with a multimeter **before** connecting) | ₹49, in stock | Probots |
| 9 | **18650 Li-ion cell** (branded, ~2600 mAh) + holder | 1 | Power | ~₹150–300 (buy a branded cell from a reputable store; not checked) | Robu, Zbotic |
| 10 | Small parts: 2× NPN transistor (2N2222/BC547), 2× 1N4007 diode, 1 kΩ resistors, 1000 µF capacitor, slide switch, perfboard, jumper wires, headers | – | Drives the motor and buzzer (an ESP32 pin can't power them directly); the capacitor stops ESP32-CAM brown-out resets | ~₹200–300 | any store |
| 11 | **ISI-certified open-face helmet** | 1 | Platform. Open-face makes camera placement and testing much easier than full-face | ₹650–995 (e.g. Habsolite, Xinor, Studds) | Amazon / shops |
| 12 | Mounting: velcro, hot glue, small plastic box/arm | – | Holds camera in front of the face and electronics at the back | ~₹100–200 | – |

**Core total: about ₹2,500–3,600** (≈ ₹1,850–2,600 if a team member already
owns a helmet). That's inside the ₹2,500–3,500 target.

## 3. Optional upgrades

| Part | Why | Price |
|---|---|---|
| **OV2640 "night vision" camera (no IR-cut filter)** + 2–3 × 850 nm IR LEDs | Directly targets two findings from our own testing: **night riding** (too dark to see the face) and **sunglasses** (many lenses pass near-IR, so the eye can still be seen). Commercial driver-monitoring cameras use this approach. Keep IR LED current low and test eye-safety guidance before pointing them at eyes. | ₹499 (Probots, out of stock) + IR LEDs ~₹50 |
| **NEO-6M GPS** module | Optional emergency location alert on RED (IDP doc stage 2) | ~₹350–500 (not checked) |
| Second vibration motor | Left and right side, stronger RED warning | ₹65 |

## 4. Power budget (estimate; measure once built)

| Load at 5 V | Current |
|---|---|
| ESP32-CAM streaming (flash off) | ~100–160 mA (≈270 mA with the flash LED on; keep it off) |
| ESP32 DevKit with Wi-Fi | ~100–150 mA (typical figure, not measured) |
| Vibration motor (only while alerting) | 85 mA |
| OLED + MPU6050 | ~25 mA |
| **Total (typical)** | **~250–350 mA ≈ 1.5 W** |

One 2600 mAh 18650 ≈ 9.6 Wh; after ~85% boost efficiency ≈ 8 Wh, giving
**roughly 4–5 hours**. Quote it as an estimate until measured; battery runtime
is one of the evaluation metrics in the IDP document.

## 5. Things that will go wrong (plan for them)

1. **ESP32-CAM brown-outs.** A weak supply causes "Brownout detector was
   triggered" resets. Use a solid 5 V supply and the 1000 µF capacitor
   across 5 V/GND near the board. Don't power it from the programmer's USB
   alone while streaming.
2. **Camera placement is the hardest physical problem.** The camera must see
   both eyes from a few cm away without blocking vision. Prototype a small arm
   off the side/chin of an open-face helmet, then re-run
   `calibrate_occlusion.py` and an evaluation session with the camera in
   its real position, since all thresholds may shift.
3. **Stock.** The ESP32-CAM is out of stock at two big stores today. Order it
   first, from wherever has it.
4. **Wi-Fi latency.** Streaming adds delay. Measure the end-to-end time from
   eyes closing to the buzzer sounding (the IDP document's "response time"
   metric).
5. **Li-ion safety.** Use the protected TP4056 module, don't short or
   puncture the cell, and keep it in a holder, not taped bare inside the helmet.

## 6. Order checklist

- [ ] ESP32-CAM + MB programmer (check stock first)
- [ ] ESP32 DevKit V1 30-pin
- [ ] MPU6050 (GY-521)
- [ ] Coin vibration motor ×2
- [ ] Active buzzer 5 V
- [ ] 0.96″ I2C OLED
- [ ] TP4056 Type-C (with protection) + MT3608 boost
- [ ] Branded 18650 cell + holder
- [ ] 2N2222 ×2, 1N4007 ×2, 1 kΩ resistors, 1000 µF capacitor, slide switch, perfboard, jumpers, headers
- [ ] Open-face ISI helmet (if nobody has one)
- [ ] Optional: night-vision OV2640 + 850 nm IR LEDs

## Sources (prices and specs)

- [ESP32-CAM-MB with OV2640, Robu](https://st.robu.in/?p=1738329) · [Probots](https://probots.co.in/esp32-cam-mb-micro-usb-programmer-ch340g-serial-chip-ov2640-camera.html)
- [ESP32 DevKit 30-pin, Robocraze](https://robocraze.com/products/esp32-development-board-with-cp2102-wifi-bluetooth-dual-core-30-pin)
- [MPU6050 GY-521, Probots](https://probots.co.in/mpu6050-6dof-imu-sensor-module-gyroscope-accelerometer-gy521.html)
- [Coin vibration motor, Robu](https://stg.robu.in/?p=1358108)
- [0.96″ I2C OLED, Zbotic](https://zbotic.in/product/0-96-inch-i2c-iic-oled-lcd-module-4pin-with-gnd-vcc-white-ssd1306-chip/)
- [TP4056 Type-C, Probots](https://probots.co.in/tp4056-1a-li-ion-lithium-battery-charging-module-with-current-protection-type-c.html)
- [MT3608 boost, Probots](https://probots.co.in/mt3608-step-up-boost-module-3-24v-to-5-28v-2a-power-regulator.html)
- [OV2640 night-vision camera, Probots](https://probots.co.in/ov2640-2-mp-night-vision-camera-module-7-5cm-160-degree-24pin.html)
- [ISI helmets under ₹1000, Digit](https://www.godigit.com/road-safety/best-helmets-under-1000)
- [ESP32-CAM current draw, Mischianti](https://www.mischianti.org/2021/08/30/esp32-cam-pinout-specs-and-arduino-ide-configuration-1/)
