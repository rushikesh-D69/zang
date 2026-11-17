"""Golden face renders for the firmware's C face renderer (display/face_sdf.c host test).

    .venv/bin/python tools/gen_face_goldens.py        (from ai_stack/)

Writes ../firmware/zang_p4/components/display/test/faces/:
  goldens.json   one case per image: the fully RESOLVED parameter values (so the C test needs no JSON-library logic),
                 blink factor, and the image file name
  <case>.ppm     240x240 RGB reference from zang_ai/face/render.py (the canonical algorithm)
Acceptance (in the C test): mean |diff| <= 2.0 and >= 99 % of pixels within 24 levels per channel.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zang_ai.face.render import EMOTIONS, PARAMS, blend, render, resolve, write_ppm  # noqa: E402
from zang_ai.paths import WORKSPACE  # noqa: E402

OUT = os.path.join(WORKSPACE, "firmware", "zang_p4", "components", "display", "test", "faces")


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = [(e, resolve(e), 1.0) for e in EMOTIONS]
    cases += [
        ("blend_happy_sad_40", blend(resolve("happy"), resolve("sad"), 0.4), 1.0),
        ("blink_neutral", resolve("neutral"), 0.05),
        ("talk_happy_open", resolve("happy", {"mouth_open": 0.45}), 1.0),
        ("gaze_left_up", resolve("neutral", {"gaze_x": -20, "gaze_y": -12}), 1.0),
        ("custom_params", resolve("neutral", {"eye_w": 30, "eye_h": 70, "eye_round": 0.4, "hue": 300, "brow": 1,
                                              "brow_angle": -20, "mouth_curve": -0.5}), 1.0),
    ]
    index = []
    for name, p, blink in cases:
        fn = f"{name}.ppm"
        write_ppm(os.path.join(OUT, fn), render(p, blink=blink))
        index.append({"name": name, "file": fn, "blink": blink, "wink": int(p.get("_wink", 0) >= 0.5),
                      "params": {k: round(float(p[k]), 6) for k in PARAMS}})
    with open(os.path.join(OUT, "goldens.json"), "w") as fh:
        json.dump({"_doc": __doc__.strip().splitlines()[0] + " Generated; do not edit.", "param_order": PARAMS,
                   "cases": index}, fh, indent=1)
    print(f"wrote {len(index)} golden faces -> {OUT}")


if __name__ == "__main__":
    main()
