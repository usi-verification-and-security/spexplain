#!/usr/bin/env python3
"""Run ReX over a set of images with several masking strategies + ConvAD.

For every (image, method) pair we invoke the ReX CLI exactly as a user would,
and scrape the `--analyse` line plus the explanation-confidence log line.
Results are written to results.csv.
"""
import csv
import os
import re
import subprocess
import sys
import time

REX = "/Users/labbaf/miniconda3/envs/rex/bin/ReX"
ROOT = "/Users/labbaf/Documents/rex"
ENV = dict(os.environ, PYTHONPATH=f"{ROOT}/ReX:{ROOT}/models")

IMAGES = ["ladybird", "peacock", "dog", "starfish", "lizard"]

# name -> (script, extra CLI args)
# gamma is the paper's confidence threshold: the explanation must reach
# gamma * (model confidence on the whole image)
METHODS = {}
for _g in ("0.0", "0.5"):
    METHODS[f"resnet_plain_zero_g{_g}"] = ("scripts/resnet_plain.py", ["--mask_value", "0", "--confidence", _g])
    METHODS[f"resnet_plain_min_g{_g}"] = ("scripts/resnet_plain.py", ["--mask_value", "min", "--confidence", _g])
    METHODS[f"resnet_plain_mean_g{_g}"] = ("scripts/resnet_plain.py", ["--mask_value", "mean", "--confidence", _g])
    METHODS[f"resnet_convad_g{_g}"] = ("scripts/resnet_ad.py", ["--mask_value", "0", "--confidence", _g])

# RegNetY-12GF is far more expensive on MPS, so only a two-method confirmation
REGNET_METHODS = {
    "regnet_plain_zero_g0.0": ("scripts/regnet_plain.py", ["--mask_value", "0", "--confidence", "0.0"]),
    "regnet_convad_g0.0": ("scripts/regnet_ad.py", ["--mask_value", "0", "--confidence", "0.0"]),
}
REGNET_IMAGES = ["ladybird", "peacock"]

ANALYSE = re.compile(
    r"path (?P<path>\S+), classification (?P<cls>\d+), area (?P<area>[\d.e+-]+), "
    r"KL divergence (?P<kl>[\d.e+-]+), robustness (?P<rob>[\d.e+-]+) "
    r"insertion curve (?P<ins>[-\d.e+]+), deletion curve (?P<dele>[-\d.e+]+), "
    r"time (?P<time>[\d.e+-]+)"
)
CONF = re.compile(r"a sufficient explanation for (\d+) found with confidence ([\d.]+)")
TARGET = re.compile(r"image classified as (\d+) with ([\d.]+) confidence")

PIXELS = 224 * 224


def jobs():
    for img in IMAGES:
        for method, (script, extra) in METHODS.items():
            yield img, method, script, extra
    for img in REGNET_IMAGES:
        for method, (script, extra) in REGNET_METHODS.items():
            yield img, method, script, extra


def main():
    out_rows = []
    if True:
        for img, method, script, extra in jobs():
            cmd = [
                REX, f"imgs/{img}.jpg",
                "--script", script,
                "--output", f"out/{img}_{method}.png",
                "--analyse",
                "-v",
            ] + extra
            print(f"=== {img} / {method} ===", flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, env=ENV, capture_output=True, text=True, cwd=f"{ROOT}/work")
            wall = time.time() - t0
            blob = p.stdout + p.stderr

            if p.returncode != 0:
                print(f"  FAILED rc={p.returncode}", flush=True)
                print(blob[-2000:], flush=True)
                out_rows.append(dict(image=img, method=method, status="error"))
                write_csv(out_rows)
                continue

            m = ANALYSE.search(blob)
            c = CONF.search(blob)
            t = TARGET.search(blob)
            if m is None:
                print("  no analyse line found", flush=True)
                out_rows.append(dict(image=img, method=method, status="no_explanation"))
                write_csv(out_rows)
                continue

            area = float(m.group("area"))
            row = dict(
                image=img,
                method=method,
                status="ok",
                target_class=int(t.group(1)) if t else None,
                target_conf=float(t.group(2)) if t else None,
                exp_class=int(m.group("cls")),
                exp_conf=float(c.group(2)) if c else None,
                area_frac=area,
                area_px=round(area * PIXELS),
                robustness=float(m.group("rob")),
                insertion_auc=float(m.group("ins")),
                deletion_auc=float(m.group("dele")),
                kl=float(m.group("kl")),
                time_s=round(wall, 1),
            )
            out_rows.append(row)
            print(
                f"  area {row['area_px']}px ({area*100:.2f}%) conf {row['exp_conf']} "
                f"rob {row['robustness']} ins {row['insertion_auc']:.3f} "
                f"del {row['deletion_auc']:.3f} {row['time_s']}s",
                flush=True,
            )

            write_csv(out_rows)
    print("wrote results.csv", flush=True)


FIELDS = ["image", "method", "status", "target_class", "target_conf", "exp_class",
          "exp_conf", "area_frac", "area_px", "robustness", "insertion_auc",
          "deletion_auc", "kl", "time_s"]


def write_csv(rows):
    with open(f"{ROOT}/work/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
