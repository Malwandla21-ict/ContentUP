"""Builds ../video_upscaler.ipynb from engine.py + the cell texts below.

Run:  python dev/build_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INTRO = """# 🎬 Natural Video Upscaler (free, Kaggle T4 GPU)

Makes your videos bigger and sharper **without** the plastic "AI look".

**How to use:** run the cells **from top to bottom**, one at a time
(click a cell, then press **Shift + Enter**, or press the ▶ button on its left).

| Step | Cell | What it does |
|---|---|---|
| 1 | **SETTINGS** | You choose how the result should look. The only cell you normally change. |
| 2 | **ENGINE** | Loads the tools. Just run it. Nothing to change. |
| 3 | **CHECK** | Checks GPU, Internet, disk space, and finds your videos. |
| 4 | **PREVIEW** | Upscales a short clip (5–10 s) and shows **before / after**. Try settings here first! |
| 5 | **UPSCALE** | Upscales the whole video(s). Saves progress, so it can continue if Kaggle stops. |
| 6 | **CLEAN-UP** | Optional: deletes previews or old progress to free disk space. |

Your results appear in the **Output** section of the right-hand panel (`/kaggle/working`).
Full beginner guide: see `README.md` in the project.
"""

SETTINGS_MD = """## Step 1 — SETTINGS
Change the values after the `=` signs. Text needs quotes (`"fast"`), numbers don't (`2`), and `True`/`False` start with a capital letter.
After changing anything, **run this cell again**.
"""

SETTINGS = '''# ╔════════════════════════════════════════════════════════════════════╗
# ║  SETTINGS - the only cell you normally change.                     ║
# ║  After changing something, run this cell again (Shift + Enter).    ║
# ╚════════════════════════════════════════════════════════════════════╝

# ---------- Where are your videos? ----------
INPUT_FOLDER = "/kaggle/input"     # "/kaggle/input" searches EVERY dataset you added. Fine for most people.
ONLY_THESE_FILES = []              # [] = all videos found. Or pick some: ["holiday.mp4", "party.mp4"]
OUTPUT_FOLDER = "/kaggle/working"  # where results go (= the "Output" panel). Leave as it is.

# ---------- How much bigger? ----------
SCALE = 2          # 2 = twice as wide and twice as tall (720p -> 1440p, 1080p -> 4K).
                   # 1.5 = 720p -> 1080p.  4 is possible but usually looks less natural.

# ---------- Which AI model? ----------
MODEL = "fast"     # "fast"   = Real-ESRGAN: one picture at a time. Quick. Good for most videos.
                   # "steady" = RealBasicVSR: also looks at the frames before/after -> less flicker.
                   #            About 3-5x slower and can look a bit "painted" on clean videos.
DENOISE_STRENGTH = 0.5   # ("fast" only) 0 = keep ALL grain and noise (most natural, but noisy videos stay noisy)
                         #               1 = remove noise strongly (clean, but skin can look waxy/plastic)

# ---------- Keep it looking natural ----------
ORIGINAL_BLEND = 0.10    # mixes this much of the ORIGINAL video back in. 0 = off.
                         # 0.05-0.25 = natural. Higher = softer and closer to the original.
FILM_GRAIN = 0.0         # adds very fine film grain. 0 = off. 0.01-0.03 = light.
                         # Helps when the result looks "too clean" or "too digital".

# ---------- Faces ----------
FACE_RESTORE = False     # True = rebuild faces with GFPGAN. OFF by default:
                         # it can make faces look fake, "too perfect", or flicker.
FACE_STRENGTH = 0.5      # (only if FACE_RESTORE = True) 0 = no change ... 1 = full GFPGAN face

# ---------- Preview (test a short clip first!) ----------
PREVIEW_VIDEO = ""            # "" = the first video. Or a file name, like "holiday.mp4"
PREVIEW_START_SECONDS = 0     # where the test clip starts. Pick a part with faces or fine detail!
PREVIEW_LENGTH_SECONDS = 8    # 5-10 seconds is enough

# ---------- Output file ----------
VIDEO_QUALITY = 17          # lower number = better quality but bigger file. 16-20 is good.
ENCODER_SPEED = "medium"    # "fast" = quicker but bigger files. "slow" = smaller files but takes longer.

# ---------- Advanced (you can ignore these) ----------
SAVE_EVERY_SECONDS = 10       # progress is saved after every 10 seconds of video
MAX_RUNTIME_HOURS = 11.5      # stop safely before Kaggle's 12-hour limit
STEADY_CHUNK_FRAMES = 30      # ("steady") frames on the GPU at once. Lowered automatically if memory runs out.
USE_HALF_PRECISION = True     # faster on a T4. Switches itself off if it ever causes problems.
STEADY_WEIGHTS_FILE = ""      # only if the steady model fails to download: path to a RealBasicVSR .pth you uploaded
SKIP_PREVIEW_IN_SAVE_VERSION = True   # don't waste time on the preview during a "Save Version" background run

print("✅ Settings saved. Next: run the ENGINE cell.")
'''

ENGINE_MD = """## Step 2 — ENGINE
This cell only *loads* the tools (it takes a few seconds). You don't need to read or change it.
Tip: click the little arrow next to the cell to fold it away.
"""

CHECK_MD = """## Step 3 — CHECK
Checks that the GPU and Internet are on, downloads the AI model files (small, a few seconds),
and lists your videos with their new size and an estimate of the file size.
If something is wrong it tells you **exactly** how to fix it.
"""

CHECK = "check_setup()\n"

PREVIEW_MD = """## Step 4 — PREVIEW (test before the full run)
Upscales only a short clip and makes **before/after** files in `/kaggle/working/previews`:
* `..._upscaled.mp4` – the upscaled clip (with sound)
* `..._compare.mp4` – the middle part of the picture, *before* on the left, *after* on the right, at full size
* `..._compare.png` – one still frame of that comparison (it is also shown below this cell)

It also measures the speed and tells you **how long the whole video will take**.
Change settings → run SETTINGS again → run PREVIEW again, until you like it. Each preview gets its own file name, so you can compare.
"""

PREVIEW = "run_preview()\n"

UPSCALE_MD = """## Step 5 — UPSCALE the whole video(s)
Shows a progress bar with the time left. Progress is saved every few seconds of video, so if the
session stops you can continue later (see the guide: *"If the time runs out"*).

* Pressing **■ stop** is safe: run this cell again and it continues where it stopped.
* Finished videos are named like `myvideo_upscaled_2x_fast_dn0.5_blend0.1.mp4`.
* ⚠️ In a normal session, files in Output are **deleted when the session ends**. Download them first,
  or use **Save Version → Save & Run All** so Kaggle keeps them.
"""

UPSCALE = "run_upscale()\n"

CLEAN_MD = """## Step 6 — CLEAN-UP (optional)
Shows what is in your Output folder and how much space is free.
Change `False` to `True` to delete things you don't need any more:
* `delete_previews` – removes all preview clips
* `delete_progress` – removes **unfinished** work (only do this if you don't want to continue it!)
"""

CLEAN = "clean_up(delete_previews=False, delete_progress=False)\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text, hide=False):
    meta = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}} if hide else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main():
    engine = open(os.path.join(HERE, "engine.py"), encoding="utf-8").read()
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": [md(INTRO), md(SETTINGS_MD), code(SETTINGS), md(ENGINE_MD), code(engine),
                  md(CHECK_MD), code(CHECK), md(PREVIEW_MD), code(PREVIEW), md(UPSCALE_MD), code(UPSCALE),
                  md(CLEAN_MD), code(CLEAN)],
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
    out = os.path.join(ROOT, "video_upscaler.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
