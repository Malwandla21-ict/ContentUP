# 🎬 Natural Video Upscaler (free, runs on Kaggle)

A notebook that makes your videos **bigger and sharper while keeping them natural**:
no plastic/waxy skin, no over-sharpened "AI look", and an option that cuts down flicker.
It runs for free on Kaggle's T4 graphics card.

**The file you need:** [`video_upscaler.ipynb`](video_upscaler.ipynb).
(You can ignore the `dev/` folder. It holds the parts the notebook is built from, plus a test.)

---

## 📖 Mini dictionary (read this first)

| Word | What it means (simple version) |
|---|---|
| **Upscaling** | Making a video bigger, e.g. 1280×720 → 2560×1440. The AI guesses the missing detail. |
| **Frame** | A video is a fast flip-book of still pictures. Each picture is a *frame*. |
| **fps** | "Frames per second": how many pictures play each second (often 24, 30 or 60). |
| **Kaggle** | A free website (owned by Google) that lends you a powerful computer in the cloud. |
| **GPU** | A graphics card: a chip that is very fast at AI work. Kaggle lends you an "NVIDIA T4". |
| **Notebook** | A page with boxes of code you run one by one, like steps in a recipe. |
| **Cell** | One box in the notebook. You run it with **Shift + Enter** or the ▶ button. |
| **Dataset** | Kaggle's word for "a folder of files you uploaded". Your videos go in one. |
| **Session** | The time Kaggle's computer is switched on for you. One session can last up to 12 hours. |
| **ffmpeg** | A free tool that takes videos apart into frames and puts frames back into a video. |
| **Noise / grain** | The fine "sand" or speckle in a video. A little of it looks natural. Removing all of it looks fake. |
| **Flicker** | Small details that "boil" or shimmer from frame to frame. |
| **H.264 / MP4** | The most common video format. It plays on every phone, TV and computer. |

---

## 🛠️ One-time setup on Kaggle (step by step)

> Kaggle sometimes moves its buttons around. If a name below doesn't match exactly, look for something similar in the same place.

### 1. Make a Kaggle account and verify your phone
1. Go to **kaggle.com** and sign up (free).
2. Click your profile picture (top right) → **Settings**. Find **Phone verification** and verify your number.
   *Kaggle only lets you use the GPU and the Internet after you verify your phone.*

### 2. Upload your video(s) as a dataset
1. In the left menu click **Datasets**, then **+ New Dataset**.
2. Drag your MP4 file(s) into the box. You can add several.
3. Give it a title, e.g. `my-videos`. Leave it **Private**.
4. Click **Create** and wait until the upload finishes. Big videos take a while, so keep the tab open.

### 3. Get the notebook into Kaggle
1. On this GitHub page, open `video_upscaler.ipynb`. Click the **download** button (an arrow pointing down, near the top right of the file view, also called "Download raw file").
2. On Kaggle, in the left menu click **Code** → **+ New Notebook**.
3. In the notebook's top menu click **File** → **Import Notebook**, and choose the file you downloaded.

### 4. Turn on the GPU and the Internet
In the notebook, open the panel on the **right side** (if it's hidden, click the small arrow at the right edge). Find **Session options** (it may be called **Settings**):
- **Accelerator** → choose **GPU T4 x2**, then confirm. (The notebook uses one of the two cards.)
- **Internet** → switch it **On**. The notebook needs it once, to download the AI model files.

### 5. Attach your video dataset
In the same right-hand panel, under **Input**, click **+ Add Input**. Find your dataset (look under "Your Work" / "Datasets"), then click the **+** next to it.
Your videos now sit in a folder under `/kaggle/input/`. You don't need to know the exact path, because the notebook searches there automatically.

---

## ▶️ Using it (every time)

Run the cells **from top to bottom**, one at a time (click inside a cell, press **Shift + Enter**).

| Cell | What it does |
|---|---|
| **1. SETTINGS** | Your choices: how big, which model, how natural. **The only cell you normally change.** Run it again after every change. |
| **2. ENGINE** | Loads all the tools (takes a few seconds). Nothing to change. You can fold it away. |
| **3. CHECK** | Checks the GPU, the Internet and free disk space, downloads the model files, and lists your videos. It shows each video's new size and a rough file size. If something is wrong, it tells you how to fix it. |
| **4. PREVIEW** | Upscales only a **short clip** (default 8 seconds) and makes **before/after** files. It also measures the speed and tells you **how long the whole video will take**. Use this to test settings cheaply! |
| **5. UPSCALE** | Does the full video(s). Shows a progress bar with **time left**. Saves progress as it goes. |
| **6. CLEAN-UP** | Optional. Lists your results and free space, and can delete previews or old unfinished work. |

### Downloading your results
Right-hand panel → **Output** → `/kaggle/working`. Hover over a file → click **⋮** → **Download**.
Previews are in the `previews` folder. Finished videos are named like
`holiday_upscaled_2x_fast_dn0.5_blend0.1.mp4`, so the name tells you the settings you used.

> ⚠️ **Important:** in a normal session, the Output files are **deleted when the session ends**
> (when you close it or it times out). Download them before you leave, or use the background run below.

### For long videos: run it in the background (recommended)
1. Do your PREVIEW tests first, so you know you like the settings.
2. Click **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
3. Kaggle now runs the whole notebook by itself. **You can close the browser.** The notebook skips the preview step in this mode. (If it ever doesn't, the preview only costs a few minutes.)
4. Later, open your notebook from **Code → Your Work**, open the version, and go to its **Output** tab to download the video. Kaggle **keeps** the output of saved versions.

### If the time runs out (videos that need more than ~11.5 hours)
Kaggle stops any session after 12 hours. The notebook watches the clock: at **11.5 hours** it **stops by itself, safely**, and saves its progress.
- **Background run (Save Version):** open the notebook editor again → **+ Add Input** → find **this same notebook** (under "Your Work" / "Notebooks") and add its latest output. Then **Save Version → Save & Run All** again. The notebook finds the saved progress and **continues where it stopped**. Repeat until it's done. The finished video appears in the newest version's Output.
- **Normal session that is still open** (for example, you pressed stop): just run the UPSCALE cell again.

Kaggle gives you a weekly GPU allowance (around 30 hours; Kaggle shows how much you have left). The preview's time estimate helps you plan.

---

## 🎛️ What each setting changes (visually)

| Setting | Default | What you'll see |
|---|---|---|
| `SCALE` | `2` | How much bigger. `2` doubles width and height (1080p → 4K). `1.5` turns 720p → 1080p. Bigger factors make the AI invent more detail, and invented detail is where the fake look comes from. **2 usually looks most natural.** |
| `MODEL` | `"fast"` | **"fast"** = Real-ESRGAN (`realesr-general-x4v3`). It works on each picture separately: quick and natural, but fine details can shimmer a little. **"steady"** = RealBasicVSR. It looks at the frames before and after, so details stay **stable** and flicker much less. About 3–5× slower, and on very clean videos it can look slightly "painted". Try both in the preview. |
| `DENOISE_STRENGTH` | `0.5` | *(fast only)* `0` keeps all grain and noise (most natural texture, but a noisy video stays noisy). `1` removes noise strongly (very clean, but **skin can turn waxy/plastic**). 0.3–0.6 is the sweet spot for real people. |
| `ORIGINAL_BLEND` | `0.10` | Mixes some of the **original** picture back in. It tones down the AI look, brings back real texture and reduces flicker. `0` = pure AI. `0.05–0.25` = natural. Higher looks softer. |
| `FILM_GRAIN` | `0.0` | Adds very fine, film-like grain that changes every frame (like real film). Hides the "too clean / too digital" look. `0.01–0.03` = light. |
| `FACE_RESTORE` | `False` | Rebuilds faces with GFPGAN. Blurry faces can look much sharper, **but** they may look fake, "too perfect", or slightly different from the real person, and they can flicker. Off by default for that reason. |
| `FACE_STRENGTH` | `0.5` | *(only with FACE_RESTORE)* How much of the GFPGAN face to use. Lower = more natural. |
| `VIDEO_QUALITY` | `17` | How carefully the final MP4 is compressed. **Lower = better quality, bigger file.** 16–20 is good. (Techy name: x264 "CRF".) |
| `ENCODER_SPEED` | `"medium"` | `"slow"` makes slightly smaller files but takes longer. `"fast"` is quicker but makes bigger files. Quality stays about the same. |
| `PREVIEW_START_SECONDS` / `PREVIEW_LENGTH_SECONDS` | `0` / `8` | Which part of the video the preview tests. **Pick a part with faces, hair, grass or text**, because that's where problems show. |
| `ONLY_THESE_FILES` | `[]` | `[]` = all videos. `["a.mp4"]` = only that one. |
| Advanced ones | | You can leave them alone. They control how often progress is saved, the time limit, GPU memory use and speed. |

### Quick fixes after looking at a preview
- **Skin looks plastic / waxy:** lower `DENOISE_STRENGTH` (e.g. 0.3) and/or raise `ORIGINAL_BLEND` (e.g. 0.2).
- **Looks over-sharpened / "AI-ish":** raise `ORIGINAL_BLEND`, add `FILM_GRAIN = 0.015`, or use `SCALE = 1.5` or `2` instead of 4.
- **Still noisy or blocky:** raise `DENOISE_STRENGTH` (e.g. 0.7).
- **Details shimmer/flicker:** try `MODEL = "steady"`. A higher `ORIGINAL_BLEND` also helps.
- **Too slow:** use `"fast"`, a smaller `SCALE`, or `ENCODER_SPEED = "fast"`.

---

## ⚙️ How it works inside (simple version)

1. **ffmpeg** takes the video apart into frames.
2. Each frame goes through the AI on the GPU. It is made 4× bigger, then gently shrunk to your `SCALE`. Shrinking from 4× gives cleaner, more natural detail than making it 2× bigger directly.
3. The optional "natural" steps are applied: blend in some of the original, add film grain, fix faces.
4. **ffmpeg** packs the frames into an H.264 MP4 at the **same frame rate**. At the end it adds the **original sound**.

A few smart extras:
- **No pile of picture files.** You asked for temporary frames to be cleaned up as it goes. The notebook goes one step further: frames flow straight from ffmpeg → GPU → ffmpeg through memory and are never saved as thousands of image files. The disk can't fill up with frames, and it's faster.
- **Saves progress every 10 seconds of video.** The finished video is built from small pieces. A notes file records which pieces are done. If anything stops, it continues from the last finished piece. When the video is complete, the pieces are deleted automatically.
- **Runs out of GPU memory?** It automatically does fewer frames at once. If that's still too much, it works on the picture in overlapping squares ("tiles") and stitches them together without visible seams.
- **Phone videos** are handled: sideways (rotated) videos come out upright. Variable frame rate becomes a steady frame rate. iPhone HDR videos are converted to normal colours.
- **Clear error messages** for the common problems: GPU off, Internet off, no videos found, wrong setting, disk full.

---

## ✅ What has been tested (and what hasn't)

Tested on a normal computer (no GPU), using small test videos:
- the fast model loads the official files correctly and produces sensible pictures
- stopping halfway and continuing gives a result **identical** to doing it in one go
- frame count, frame rate, sound and colours are kept
- tiles stitch together without seams
- previews and before/after files are made
- rotated phone videos and HDR videos work
- the face add-on installs and runs
- error messages appear for wrong settings and missing files

**Not yet tested on Kaggle's GPU itself.** That needs your Kaggle account, so your first real run is the true test. In particular:
- **Speed** is unknown until you run it. The preview tells you.
- The **"steady" model**: its official file couldn't be downloaded on my test machine, so I tested its machinery but not the real model file. I built it to match the official code. If its file won't load on Kaggle, the notebook says so clearly and you can use "fast".

If something goes wrong, copy the message the notebook shows and ask. The messages are written to make fixing easy.

---

## Credits
- Real-ESRGAN (`realesr-general-x4v3`) by Xintao Wang et al., BSD-3-Clause.
- RealBasicVSR by Kelvin C.K. Chan et al. (via OpenMMLab), Apache-2.0.
- GFPGAN by TencentARC, Apache-2.0 (its face models have their own licence terms).
