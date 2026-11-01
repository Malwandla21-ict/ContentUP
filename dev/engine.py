# =============================================================================
#  THE ENGINE  -  you do NOT need to change anything in this cell.
#  It only *defines* the tools. Nothing heavy happens until you run the
#  "check", "preview" or "upscale" cells further down.
# =============================================================================
import os, sys, re, json, time, math, glob, shutil, hashlib, threading, queue
import subprocess, urllib.request, gc
from collections import deque
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    from tqdm import tqdm

ON_KAGGLE = os.path.isdir("/kaggle/working")
MODEL_DIR = "/tmp/upscaler_models"            # model files (not part of your output)
WORK_FOLDER_NAME = "upscaler_work"            # saved progress lives here
PREVIEW_FOLDER_NAME = "previews"
KAGGLE_OUTPUT_LIMIT_GB = 19.5                 # Kaggle keeps ~20 GB in /kaggle/working
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
#  Friendly messages
# -----------------------------------------------------------------------------
class UpscalerError(Exception):
    """An error with a plain-English explanation (no scary traceback needed)."""


def say(title, *lines):
    bar = "=" * 72
    print(bar)
    print(title)
    for line in lines:
        print("  " + line)
    print(bar)


def friendly(func):
    """Run a top-level step; show UpscalerErrors as clean messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UpscalerError as e:
            say("❌ STOPPED - here is what went wrong:", *str(e).splitlines())
        except KeyboardInterrupt:
            say("⏸  You stopped the cell.",
                "Any finished parts are saved. Run the cell again to continue",
                "from where it stopped.")
    wrapper.__name__ = func.__name__
    return wrapper


def human_time(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


def human_size(n_bytes):
    gb = n_bytes / 1024 ** 3
    return f"{gb:.2f} GB" if gb >= 1 else f"{n_bytes / 1024 ** 2:.0f} MB"


# -----------------------------------------------------------------------------
#  Settings
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    INPUT_FOLDER="/kaggle/input", ONLY_THESE_FILES=[], OUTPUT_FOLDER="/kaggle/working",
    SCALE=2, MODEL="fast", DENOISE_STRENGTH=0.5,
    ORIGINAL_BLEND=0.10, FILM_GRAIN=0.0,
    FACE_RESTORE=False, FACE_STRENGTH=0.5,
    PREVIEW_VIDEO="", PREVIEW_START_SECONDS=0, PREVIEW_LENGTH_SECONDS=8,
    VIDEO_QUALITY=17, ENCODER_SPEED="medium",
    STEADY_CHUNK_FRAMES=30, SAVE_EVERY_SECONDS=10, MAX_RUNTIME_HOURS=11.5,
    USE_HALF_PRECISION=True, STEADY_WEIGHTS_FILE="", SKIP_PREVIEW_IN_SAVE_VERSION=True,
)
X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow"]


class Settings:
    pass


def settings(source=None):
    """Read the settings from the SETTINGS cell and check they make sense."""
    g = source if source is not None else globals()
    S = Settings()
    for k, v in DEFAULTS.items():
        setattr(S, k, g.get(k, v))

    problems = []
    try:
        S.SCALE = float(S.SCALE)
    except Exception:
        problems.append(f"SCALE must be a number like 2 (you wrote {S.SCALE!r}).")
    else:
        if not 1.0 <= S.SCALE <= 4.0:
            problems.append(f"SCALE must be between 1 and 4 (you wrote {S.SCALE}).")
    S.MODEL = str(S.MODEL).strip().lower()
    if S.MODEL not in ("fast", "steady"):
        problems.append(f'MODEL must be "fast" or "steady" (you wrote {S.MODEL!r}).')
    for name, lo, hi in [("DENOISE_STRENGTH", 0, 1), ("ORIGINAL_BLEND", 0, 0.9),
                         ("FILM_GRAIN", 0, 0.2), ("FACE_STRENGTH", 0, 1),
                         ("VIDEO_QUALITY", 0, 40), ("MAX_RUNTIME_HOURS", 0.1, 100),
                         ("SAVE_EVERY_SECONDS", 1, 600), ("STEADY_CHUNK_FRAMES", 4, 200),
                         ("PREVIEW_LENGTH_SECONDS", 1, 60), ("PREVIEW_START_SECONDS", 0, 1e9)]:
        try:
            val = float(getattr(S, name))
        except Exception:
            problems.append(f"{name} must be a number (you wrote {getattr(S, name)!r}).")
            continue
        if not lo <= val <= hi:
            problems.append(f"{name} must be between {lo} and {hi} (you wrote {val}).")
        setattr(S, name, val)
    S.VIDEO_QUALITY = int(round(S.VIDEO_QUALITY)) if isinstance(S.VIDEO_QUALITY, float) else S.VIDEO_QUALITY
    S.STEADY_CHUNK_FRAMES = int(S.STEADY_CHUNK_FRAMES) if isinstance(S.STEADY_CHUNK_FRAMES, float) else S.STEADY_CHUNK_FRAMES
    S.ENCODER_SPEED = str(S.ENCODER_SPEED).strip().lower()
    if S.ENCODER_SPEED not in X264_PRESETS:
        problems.append(f"ENCODER_SPEED must be one of {X264_PRESETS}.")
    if isinstance(S.ONLY_THESE_FILES, str):
        S.ONLY_THESE_FILES = [S.ONLY_THESE_FILES] if S.ONLY_THESE_FILES.strip() else []
    S.FACE_RESTORE = bool(S.FACE_RESTORE)
    S.USE_HALF_PRECISION = bool(S.USE_HALF_PRECISION)
    S.SKIP_PREVIEW_IN_SAVE_VERSION = bool(S.SKIP_PREVIEW_IN_SAVE_VERSION)
    if problems:
        raise UpscalerError("There is a problem in the SETTINGS cell:\n" + "\n".join("• " + p for p in problems)
                            + "\nFix it, run the SETTINGS cell again, then re-run this cell.")
    if S.ORIGINAL_BLEND > 0.5:
        print(f"⚠️  ORIGINAL_BLEND = {S.ORIGINAL_BLEND} is high: the result will look close to the "
              "original (soft). 0.05-0.25 is the usual range.")
    if S.FILM_GRAIN > 0.05:
        print(f"⚠️  FILM_GRAIN = {S.FILM_GRAIN} is strong. 0.01-0.03 is usually enough.")
    return S


def settings_key(S):
    """A short code that changes whenever a setting that affects the picture changes."""
    parts = dict(scale=S.SCALE, model=S.MODEL, blend=S.ORIGINAL_BLEND, grain=S.FILM_GRAIN,
                 face=S.FACE_RESTORE, crf=S.VIDEO_QUALITY, preset=S.ENCODER_SPEED, v=1)
    if S.MODEL == "fast":
        parts["denoise"] = S.DENOISE_STRENGTH
    if S.FACE_RESTORE:
        parts["face_strength"] = S.FACE_STRENGTH
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8]


def settings_label(S):
    def fmt(x):
        return f"{x:g}"
    label = f"{fmt(S.SCALE)}x_{S.MODEL}"
    if S.MODEL == "fast":
        label += f"_dn{fmt(S.DENOISE_STRENGTH)}"
    if S.ORIGINAL_BLEND:
        label += f"_blend{fmt(S.ORIGINAL_BLEND)}"
    if S.FILM_GRAIN:
        label += f"_grain{fmt(S.FILM_GRAIN)}"
    if S.FACE_RESTORE:
        label += f"_face{fmt(S.FACE_STRENGTH)}"
    return label


# -----------------------------------------------------------------------------
#  ffmpeg helpers
# -----------------------------------------------------------------------------
_FFMPEG_INFO = {}


def ffmpeg_info():
    if _FFMPEG_INFO:
        return _FFMPEG_INFO
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise UpscalerError("ffmpeg (the video tool) is not installed. Run the CHECK cell first - "
                            "it installs it automatically if needed.")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout
    m = re.search(r"version\s+n?(\d+)", out)
    major = int(m.group(1)) if m else 4
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    _FFMPEG_INFO.update(major=major, zscale=" zscale " in filters, tonemap=" tonemap " in filters)
    return _FFMPEG_INFO


def passthrough_flag():
    return ["-fps_mode", "passthrough"] if ffmpeg_info()["major"] >= 5 else ["-vsync", "passthrough"]


def parse_rate(text):
    try:
        fr = Fraction(str(text))
        return fr if fr > 0 else None
    except Exception:
        return None


def probe_video(path):
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise UpscalerError(f"File not found: {path}")
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_streams", "-show_format", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise UpscalerError(f"Could not read '{name}'. It may be damaged or not a real video.\n"
                            f"ffprobe said: {r.stderr.strip()[-400:]}")
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        raise UpscalerError(f"'{name}' has no video picture inside it (maybe it is audio only).")
    vs = vids[0]
    fmt = data.get("format", {})
    w, h = int(vs.get("width", 0)), int(vs.get("height", 0))
    if w <= 0 or h <= 0:
        raise UpscalerError(f"Could not find the picture size of '{name}'.")

    rotation = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if not rotation and "rotate" in vs.get("tags", {}):
        rotation = int(float(vs["tags"]["rotate"]))
    if abs(rotation) % 180 == 90:   # phone videos filmed upright: ffmpeg turns them for us
        w, h = h, w

    r_rate, a_rate = parse_rate(vs.get("r_frame_rate")), parse_rate(vs.get("avg_frame_rate"))
    if r_rate and r_rate <= 120 and (not a_rate or abs(float(r_rate) - float(a_rate)) / float(r_rate) < 0.01):
        fps = r_rate
    elif a_rate and a_rate <= 240:
        fps = a_rate
    elif r_rate and r_rate <= 240:
        fps = r_rate
    else:
        fps = Fraction(30, 1)
    fps = fps.limit_denominator(1001)

    duration = 0.0
    for d in (vs.get("duration"), fmt.get("duration")):
        try:
            duration = float(d)
            if duration > 0:
                break
        except (TypeError, ValueError):
            pass
    nb = int(vs.get("nb_frames") or 0)
    est_frames = nb if (nb > 0 and fps == r_rate) else int(round(duration * float(fps)))
    est_frames = max(est_frames, 1)

    audio = [s for s in streams if s.get("codec_type") == "audio"]
    space = (vs.get("color_space") or "").lower()
    transfer = (vs.get("color_transfer") or "").lower()
    if space in ("bt709",):
        matrix = "bt709"
    elif space in ("smpte170m", "bt470bg", "fcc"):
        matrix = "bt601"
    elif space.startswith("bt2020"):
        matrix = "bt2020"
    else:
        matrix = "bt709" if w * h >= 1280 * 720 else "bt601"

    return dict(path=path, name=name, stem=os.path.splitext(name)[0], width=w, height=h,
                fps=fps, fps_str=f"{fps.numerator}/{fps.denominator}", duration=duration,
                est_frames=est_frames, has_audio=bool(audio),
                audio_codec=(audio[0].get("codec_name") if audio else None),
                matrix=matrix, hdr=transfer in ("smpte2084", "arib-std-b67"),
                vfr=bool(r_rate and a_rate and abs(float(r_rate) - float(a_rate)) / float(r_rate) >= 0.01),
                pix_fmt=vs.get("pix_fmt", ""), codec=vs.get("codec_name", ""))


def output_size(info, scale):
    def even(x):
        return max(2, int(round(x / 2.0)) * 2)
    return even(info["width"] * scale), even(info["height"] * scale)


def decode_command(info, start_frame=0, seek_seconds=None, length_seconds=None):
    """ffmpeg command that turns the video into raw RGB pictures, one after another."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error"]
    if seek_seconds is not None:
        cmd += ["-ss", f"{seek_seconds:.3f}"]
    if length_seconds is not None:
        cmd += ["-t", f"{length_seconds:.3f}"]
    cmd += ["-i", info["path"], "-map", "0:v:0", "-an", "-sn", "-dn"]
    vf = [f"fps={info['fps_str']}"]
    if start_frame > 0:
        vf += [f"select=gte(n\\,{int(start_frame)})", "setpts=N/FRAME_RATE/TB"]
    ff = ffmpeg_info()
    if info["hdr"] and ff["zscale"] and ff["tonemap"]:
        vf += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
               "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=rgb24"]
    else:
        vf += [f"scale=in_color_matrix={info['matrix']}:flags=bicubic+accurate_rnd+full_chroma_int",
               "format=rgb24"]
    cmd += ["-vf", ",".join(vf)] + passthrough_flag() + ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    return cmd


def log_tail(path, n=600):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""


class FrameReader:
    """Reads pictures from ffmpeg in the background so the GPU never waits."""

    def __init__(self, cmd, width, height, log_path, prefetch=24):
        self.w, self.h = width, height
        self.frame_bytes = width * height * 3
        self.log_path = log_path
        self.log = open(log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.q = queue.Queue(maxsize=prefetch)
        self.pending = deque()
        self.eof = False
        self.stop = False
        self.frames_read = 0
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _read_exact(self):
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        got = 0
        while got < self.frame_bytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return buf

    def _pump(self):
        try:
            while not self.stop:
                buf = self._read_exact()
                if buf is None:
                    break
                self.q.put(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3))
        finally:
            self.q.put(None)

    def read(self):
        if self.pending:
            return self.pending.popleft()
        if self.eof:
            return None
        frame = self.q.get()
        if frame is None:
            self.eof = True
            return None
        self.frames_read += 1
        return frame

    def read_n(self, n):
        out = []
        for _ in range(int(n)):
            f = self.read()
            if f is None:
                break
            out.append(f)
        return out

    def peek_n(self, n):
        frames = self.read_n(n)
        self.pending.extendleft(reversed(frames))
        return frames

    def check_error(self):
        """Call after reaching the end: complains if ffmpeg failed."""
        rc = self.proc.wait()
        if rc != 0 and not self.stop:
            if self.frames_read == 0:
                raise UpscalerError("ffmpeg could not read the video.\n" + log_tail(self.log_path))
            print("\n⚠️  ffmpeg reported a problem near the end of the video (the file may be slightly "
                  "damaged). Everything it could read was upscaled.\n" + log_tail(self.log_path, 300))

    def close(self):
        self.stop = True
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        while self.thread.is_alive():
            try:
                self.q.get(timeout=0.1)
            except queue.Empty:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.log.close()


class VideoWriter:
    """Sends finished pictures to ffmpeg, which packs them into a H.264 MP4 file."""

    def __init__(self, path, width, height, fps_str, S, log_path):
        self.path = path
        self.log_path = log_path
        tune = "grain" if S.FILM_GRAIN >= 0.02 else "film"
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-framerate", fps_str, "-i", "pipe:0",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p",
               "-c:v", "libx264", "-preset", S.ENCODER_SPEED, "-crf", str(S.VIDEO_QUALITY), "-tune", tune,
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", path]
        self.log = open(log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.q = queue.Queue(maxsize=8)
        self.error = None
        self.count = 0
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            if self.error:
                continue
            try:
                self.proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
            except Exception as e:  # ffmpeg died (disk full, etc.)
                self.error = e

    def write(self, frame):
        if self.error:
            raise UpscalerError("The video encoder (ffmpeg) stopped unexpectedly. Is the disk full?\n"
                                + log_tail(self.log_path))
        self.q.put(frame)
        self.count += 1

    def close(self):
        self.q.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        self.log.close()
        if rc != 0 or self.error:
            raise UpscalerError("The video encoder (ffmpeg) failed. Is the disk full?\n" + log_tail(self.log_path))

    def abort(self):
        try:
            self.q.put(None)
            self.thread.join(timeout=5)
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        try:
            self.log.close()
        except Exception:
            pass
        if os.path.exists(self.path):
            os.remove(self.path)


# -----------------------------------------------------------------------------
#  Model downloads
# -----------------------------------------------------------------------------
WEIGHT_SOURCES = {
    "general": ("realesr-general-x4v3.pth",
                ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"]),
    "general_wdn": ("realesr-general-wdn-x4v3.pth",
                    ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"]),
    "realbasicvsr": ("realbasicvsr_x4.pth",
                     ["https://download.openmmlab.com/mmediting/restorers/real_basicvsr/"
                      "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth"]),
    "gfpgan": ("GFPGANv1.4.pth",
               ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"]),
}


def get_weights(key, manual_path=""):
    """Returns the path of a model file, downloading it the first time."""
    if manual_path:
        if os.path.isfile(manual_path):
            return manual_path
        raise UpscalerError(f"STEADY_WEIGHTS_FILE points to a file that does not exist:\n{manual_path}")
    fname, urls = WEIGHT_SOURCES[key]
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, fname)
    if os.path.isfile(dest) and os.path.getsize(dest) > 100_000:
        return dest
    if key == "realbasicvsr":   # maybe you uploaded it yourself as a dataset
        for p in glob.glob("/kaggle/input/**/*.pth", recursive=True):
            if "basicvsr" in os.path.basename(p).lower():
                print(f"Using the steady-model file you uploaded: {p}")
                return p
    last_err = None
    for url in urls:
        for attempt in range(4):
            try:
                print(f"Downloading {fname} ...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                tmp = dest + ".part"
                with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1024 * 1024)
                if os.path.getsize(tmp) < 100_000:
                    raise IOError("downloaded file is too small (probably an error page)")
                os.replace(tmp, dest)
                return dest
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
    hint = ("Turn on Internet: in the notebook's right-hand panel open 'Session options' "
            "(or 'Settings') and switch 'Internet' ON. Your Kaggle account must be phone-verified for this.")
    if key == "realbasicvsr":
        hint += ("\nOr download the RealBasicVSR .pth file yourself, upload it to Kaggle as a dataset, and "
                 "put its path in STEADY_WEIGHTS_FILE. Or simply use MODEL = \"fast\".")
    raise UpscalerError(f"Could not download the model file {fname}.\nReason: {last_err}\n{hint}")


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Official checkpoints from trusted sources (Real-ESRGAN / OpenMMLab) only.
        return torch.load(path, map_location="cpu", weights_only=False)


# -----------------------------------------------------------------------------
#  Neural networks (written out here so no fragile extra packages are needed)
# -----------------------------------------------------------------------------
class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN 'realesr-general-x4v3' network (same layout as the official one)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class ResidualBlockNoBN(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_ch, out_ch=64, num_blocks=30):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_ch) for _ in range(num_blocks)]))

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_ch, out_ch, scale, kernel=3):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel, padding=(kernel - 1) // 2)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class ConvModule(nn.Module):
    def __init__(self, cin, cout, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x) if self.relu else x


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            ConvModule(8, 32), ConvModule(32, 64), ConvModule(64, 32), ConvModule(32, 16),
            ConvModule(16, 2, relu=False))

    def forward(self, x):
        return self.basic_module(x)


def flow_warp(x, flow, padding_mode="zeros"):
    """Moves the pixels of x along the motion 'flow' (N,H,W,2)."""
    _, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                            torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
    grid = torch.stack((gx, gy), 2) + flow.float()
    gfx = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    gfy = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    out = F.grid_sample(x.float(), torch.stack((gfx, gfy), dim=3), mode="bilinear",
                        padding_mode=padding_mode, align_corners=True)
    return out.to(x.dtype)


class SPyNet(nn.Module):
    """Small network that estimates how things move between two frames."""

    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        n, _, h, w = ref.shape
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], 2, 2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], 2, 2, count_include_pad=False))
        ref, supp = ref[::-1], supp[::-1]
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            flow_up = flow if level == 0 else F.interpolate(
                flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border"),
                flow_up.to(ref[level].dtype)], 1))
        return flow

    def forward(self, ref, supp):
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp).float(), size=(h, w), mode="bilinear",
                             align_corners=False)
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow


class BasicVSRNet(nn.Module):
    def __init__(self, ch=64, num_blocks=20):
        super().__init__()
        self.mid_channels = ch
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(ch + 3, ch, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(ch + 3, ch, num_blocks)
        self.fusion = nn.Conv2d(ch * 2, ch, 1, 1, 0)
        self.upsample1 = PixelShufflePack(ch, ch, 2)
        self.upsample2 = PixelShufflePack(ch, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)


class RealBasicVSRNet(nn.Module):
    """RealBasicVSR: looks at neighbouring frames, so details stay steady over time."""

    def __init__(self, ch=64, num_prop_blocks=20, num_clean_blocks=20):
        super().__init__()
        self.image_cleaning = nn.Sequential(ResidualBlocksWithInputConv(3, ch, num_clean_blocks),
                                            nn.Conv2d(ch, 3, 3, 1, 1))
        self.basicvsr = BasicVSRNet(ch, num_prop_blocks)

    def clean(self, lqs, passes):
        """Removes noise/blockiness before upscaling (one frame at a time to save memory)."""
        for _ in range(passes):
            lqs = torch.cat([lqs[i:i + 1] + self.image_cleaning(lqs[i:i + 1]).to(lqs.dtype)
                             for i in range(lqs.shape[0])])
        return lqs

    def decide_cleaning_passes(self, lqs, threshold=5 / 255.0):
        """Same rule as the official test setting: clean up to 3 times while it still changes a lot."""
        passes = 0
        for _ in range(3):
            res = [self.image_cleaning(lqs[i:i + 1]).float() for i in range(lqs.shape[0])]
            lqs = torch.cat([lqs[i:i + 1] + r.to(lqs.dtype) for i, r in enumerate(res)])
            passes += 1
            if torch.stack([r.abs().mean() for r in res]).mean().item() < threshold:
                break
        return passes

    def run(self, lqs, keep_from, keep_count, passes, emit, keep_feats_on_gpu=True):
        """lqs: (T,3,H,W) in [0,1]. Calls emit(k, sr) for frames keep_from..keep_from+keep_count-1."""
        bv = self.basicvsr
        lqs = self.clean(lqs, passes)
        t, _, h, w = lqs.shape
        last = keep_from + keep_count - 1
        store = lqs.device if keep_feats_on_gpu else torch.device("cpu")

        back = {}
        feat = lqs.new_zeros(1, bv.mid_channels, h, w)
        for i in range(t - 1, keep_from - 1, -1):          # look backwards in time
            if i < t - 1:
                flow = bv.spynet(lqs[i:i + 1], lqs[i + 1:i + 2])
                feat = flow_warp(feat, flow.permute(0, 2, 3, 1))
            feat = bv.backward_resblocks(torch.cat([lqs[i:i + 1], feat.to(lqs.dtype)], 1))
            if i <= last:
                back[i] = feat.to(store, non_blocking=True)

        feat = torch.zeros_like(feat)
        for i in range(0, last + 1):                        # look forwards in time
            if i > 0:
                flow = bv.spynet(lqs[i:i + 1], lqs[i - 1:i])
                feat = flow_warp(feat, flow.permute(0, 2, 3, 1))
            feat = bv.forward_resblocks(torch.cat([lqs[i:i + 1], feat.to(lqs.dtype)], 1))
            if i >= keep_from:
                out = torch.cat([back.pop(i).to(feat.device), feat], 1)
                out = bv.lrelu(bv.fusion(out))
                out = bv.lrelu(bv.upsample1(out))
                out = bv.lrelu(bv.upsample2(out))
                out = bv.lrelu(bv.conv_hr(out))
                out = bv.conv_last(out)
                out = out + F.interpolate(lqs[i:i + 1], scale_factor=4, mode="bilinear",
                                          align_corners=False).to(out.dtype)
                emit(i - keep_from, out)
                del out


# -----------------------------------------------------------------------------
#  Upscaling engines
# -----------------------------------------------------------------------------
class NonFiniteOutput(Exception):
    pass


def is_memory_error(e):
    if isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(e)
    return any(s in msg for s in ("out of memory", "CUDNN_STATUS", "CUBLAS_STATUS_ALLOC",
                                  "canUse32BitIndexMath", "cuDNN error", "valid cuDNN algorithm",
                                  "DefaultCPUAllocator: can't allocate"))


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


TILE_STEPS = [768, 512, 384, 256, 192, 128]


class Finisher:
    """Turns the model's 4x picture into the final picture: resize, blend, grain."""

    def __init__(self, out_w, out_h, S, device):
        self.out_w, self.out_h = out_w, out_h
        self.blend = float(S.ORIGINAL_BLEND)
        self.grain = float(S.FILM_GRAIN)
        self.device = device
        k = torch.tensor([0.25, 0.5, 0.25], device=device)
        self.grain_kernel = (k[:, None] * k[None, :]).view(1, 1, 3, 3)

    def original_upscaled(self, lr):
        return F.interpolate(lr.float(), size=(self.out_h, self.out_w), mode="bicubic",
                             align_corners=False).clamp(0, 1)

    def __call__(self, sr4, lr):
        sr = sr4.to(self.device).float()
        if not torch.isfinite(sr).all():
            raise NonFiniteOutput()
        if sr.shape[-2:] != (self.out_h, self.out_w):
            sr = F.interpolate(sr, size=(self.out_h, self.out_w), mode="bicubic",
                               align_corners=False, antialias=True)
        if self.blend > 0:
            sr = sr * (1 - self.blend) + self.original_upscaled(lr) * self.blend
        sr = sr.clamp(0, 1)
        if self.grain > 0:
            noise = torch.randn(1, 1, self.out_h, self.out_w, device=sr.device)
            noise = F.conv2d(F.pad(noise, (1, 1, 1, 1), mode="reflect"), self.grain_kernel)
            noise = noise / noise.std().clamp_min(1e-6)       # soft, film-like grain clumps
            luma = (0.2126 * sr[:, 0:1] + 0.7152 * sr[:, 1:2] + 0.0722 * sr[:, 2:3])
            weight = 0.35 + 2.6 * luma * (1 - luma)           # grain shows most in mid-tones
            sr = (sr + noise * weight * self.grain).clamp(0, 1)
        return (sr[0] * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


def frames_to_tensor(frames, device):
    arr = torch.from_numpy(np.stack(frames))                 # T,H,W,3 uint8
    return arr.to(device).permute(0, 3, 1, 2).float().div_(255.0)


def tiled_forward(fn, x, tile, pad, scale=4, canvas_device=None):
    """Runs fn on overlapping squares of x and stitches the results (saves GPU memory)."""
    b, c, h, w = x.shape
    canvas = torch.empty(b, 3, h * scale, w * scale, dtype=torch.float16,
                         device=canvas_device or x.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
            py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
            out = fn(x[:, :, py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            canvas[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                out[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale].to(canvas.dtype)
            del out
    return canvas


class FastEngine:
    """Real-ESRGAN realesr-general-x4v3, one picture at a time (with denoise control)."""
    context = 0

    def __init__(self, S, device, out_w, out_h):
        self.device = device
        self.half = S.USE_HALF_PRECISION and device.type == "cuda"
        self.finish = Finisher(out_w, out_h, S, device)
        a = torch_load(get_weights("general"))
        b = torch_load(get_weights("general_wdn"))
        a = a.get("params_ema", a.get("params", a))
        b = b.get("params_ema", b.get("params", b))
        d = float(S.DENOISE_STRENGTH)
        # Mix the "strong denoise" and "keep noise" models (official Real-ESRGAN trick)
        mixed = {k: d * a[k].float() + (1 - d) * b[k].float() for k in a}
        self.net = SRVGGNetCompact()
        self.net.load_state_dict(mixed, strict=True)
        self.net.eval().to(device)
        self.batch = 4 if device.type == "cuda" else 1
        self.tile = None
        self.step = 8

    def _forward(self, x):
        pad = 8
        h, w = x.shape[-2:]
        mode = "reflect" if min(h, w) > pad else "replicate"
        xp = F.pad(x, (pad, pad, pad, pad), mode=mode)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                             enabled=self.half):
            out = self.net(xp)
        return out[:, :, pad * 4:pad * 4 + h * 4, pad * 4:pad * 4 + w * 4]

    def _process(self, frames):
        x = frames_to_tensor(frames, self.device)
        with torch.no_grad():
            if self.tile:
                sr = tiled_forward(self._forward, x, self.tile, pad=40)
            else:
                sr = self._forward(x)
            return [self.finish(sr[j:j + 1], x[j:j + 1]) for j in range(len(frames))]

    def _shrink(self):
        if self.batch > 1:
            self.batch //= 2
            return
        smaller = [t for t in TILE_STEPS if self.tile is None or t < self.tile]
        if not smaller:
            raise UpscalerError("Ran out of GPU memory even with the smallest tiles. "
                                "Try a smaller SCALE or a lower-resolution video.")
        self.tile = smaller[0]
        print(f"\n(GPU memory is tight - switching to tiles of {self.tile} px. This is automatic.)")

    def run(self, before, core, after):
        out, i = [], 0
        while i < len(core):
            chunk = core[i:i + self.batch]
            try:
                out.extend(self._process(chunk))
                i += len(chunk)
            except NonFiniteOutput:
                if not self.half:
                    raise UpscalerError("The model produced broken pixels. Please report this video.")
                print("\n(Half precision gave bad pixels - switching to full precision. Automatic.)")
                self.half = False
            except Exception as e:
                if not is_memory_error(e):
                    raise
                free_memory()
                self._shrink()
            if self.tile:
                free_memory()
        return out


class SteadyEngine:
    """RealBasicVSR: uses neighbouring frames, so there is much less flicker."""

    def __init__(self, S, device, out_w, out_h):
        self.device = device
        self.half = S.USE_HALF_PRECISION and device.type == "cuda"
        self.finish = Finisher(out_w, out_h, S, device)
        self.context = 5
        self.chunk = int(S.STEADY_CHUNK_FRAMES)
        self.step = self.chunk
        self.tile = None
        self.passes = None
        path = get_weights("realbasicvsr", S.STEADY_WEIGHTS_FILE)
        ck = torch_load(path)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        for prefix in ("generator_ema.", "generator.", ""):
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if any(k.startswith("basicvsr.") for k in sub):
                break
        self.net = RealBasicVSRNet()
        missing, unexpected = self.net.load_state_dict(sub, strict=False)
        missing = [k for k in missing if not k.endswith((".mean", ".std"))]
        if missing or unexpected:
            raise UpscalerError("The steady-model file does not match what this notebook expects.\n"
                                f"Missing: {missing[:5]}  Unexpected: {unexpected[:5]}\n"
                                'Please use MODEL = "fast" for now.')
        self.net.eval().to(device)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half)

    def calibrate(self, info):
        """Decides once per video how much 'cleaning' the model does, so it never jumps mid-video."""
        frames = []
        n = 6
        for k in range(n):
            t = info["duration"] * (k + 0.5) / n if info["duration"] > 0 else 0
            cmd = decode_command(info, seek_seconds=t)
            cmd = cmd[:-1] + ["-frames:v", "1", "pipe:1"]
            r = subprocess.run(cmd, capture_output=True)
            if len(r.stdout) == info["width"] * info["height"] * 3:
                frames.append(np.frombuffer(r.stdout, np.uint8).reshape(info["height"], info["width"], 3))
        if not frames:
            self.passes = 1
            return
        x = frames_to_tensor(frames, self.device)
        side = max(x.shape[-2:])
        if side > 1280:  # keep this quick check light on memory
            f = 1280 / side
            x = F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
        with torch.no_grad(), self.autocast():
            self.passes = self.net.decide_cleaning_passes(x)
        free_memory()

    def _window(self, frames, keep_from, keep_count):
        x = frames_to_tensor(frames, self.device)
        h, w = x.shape[-2:]
        results = [None] * keep_count
        feat_bytes = keep_count * 64 * h * w * (2 if self.half else 4)
        on_gpu = self.device.type == "cuda" and feat_bytes < 2.5 * 1024 ** 3
        with torch.no_grad(), self.autocast():
            if not self.tile:
                def emit(k, sr):
                    results[k] = self.finish(sr, x[keep_from + k:keep_from + k + 1])
                self.net.run(x, keep_from, keep_count, self.passes, emit, on_gpu)
            else:
                canvases = [torch.empty(1, 3, h * 4, w * 4, dtype=torch.float16) for _ in range(keep_count)]
                tile, pad = self.tile, 48
                for y0 in range(0, h, tile):
                    for x0 in range(0, w, tile):
                        y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                        py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                        py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
                        oy, ox = (y0 - py0) * 4, (x0 - px0) * 4

                        def emit(k, sr):
                            canvases[k][:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] = \
                                sr[:, :, oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4].to("cpu", torch.float16)
                        self.net.run(x[:, :, py0:py1, px0:px1].contiguous(), keep_from, keep_count,
                                     self.passes, emit, on_gpu)
                        free_memory()
                for k in range(keep_count):
                    results[k] = self.finish(canvases[k], x[keep_from + k:keep_from + k + 1])
                    canvases[k] = None
        return results

    def run(self, before, core, after):
        ctx = self.context
        before, after = list(before[-ctx:]) if ctx else [], list(after[:ctx])
        if len(core) > self.chunk:
            out = []
            for s in range(0, len(core), self.chunk):
                b = (list(before) + list(core[:s]))[-ctx:]
                a = (list(core[s + self.chunk:]) + after)[:ctx]
                out += self.run(b, core[s:s + self.chunk], a)
            return out
        try:
            return self._window(before + list(core) + after, len(before), len(core))
        except NonFiniteOutput:
            if not self.half:
                raise UpscalerError("The model produced broken pixels. Please report this video.")
            print("\n(Half precision gave bad pixels - switching to full precision. Automatic.)")
            self.half = False
        except Exception as e:
            if not is_memory_error(e):
                raise
            free_memory()
            if len(core) > 6:
                self.chunk = max(6, len(core) // 2)
                print(f"\n(GPU memory is tight - now doing {self.chunk} frames at a time. Automatic.)")
            else:
                smaller = [t for t in TILE_STEPS if self.tile is None or t < self.tile]
                if not smaller:
                    raise UpscalerError("Ran out of GPU memory even with the smallest settings. "
                                        "Try a smaller SCALE, a lower-resolution video, or MODEL = \"fast\".")
                self.tile = smaller[0]
                print(f"\n(GPU memory is tight - switching to tiles of {self.tile} px. Automatic.)")
        free_memory()
        return self.run(before, core, after)


class FaceRestorer:
    """Optional GFPGAN face fixing. Off by default because it can make faces look fake."""

    def __init__(self, S, device):
        self.strength = float(S.FACE_STRENGTH)
        ensure_gfpgan_installed()
        from gfpgan import GFPGANer
        weights = get_weights("gfpgan")
        here = os.getcwd()
        os.chdir(MODEL_DIR)   # its helper files download into the current folder - keep them out of Output
        try:
            self.restorer = GFPGANer(model_path=weights, upscale=1, arch="clean",
                                     channel_multiplier=2, bg_upsampler=None, device=device)
        finally:
            os.chdir(here)

    def __call__(self, rgb):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        try:
            _, _, restored = self.restorer.enhance(bgr, has_aligned=False, only_center_face=False,
                                                   paste_back=True)
        except Exception as e:
            if is_memory_error(e):
                free_memory()
                return rgb
            raise
        if restored is None:
            return rgb
        restored = restored[:, :, ::-1]
        if restored.shape != rgb.shape:
            return rgb
        s = self.strength
        return np.clip(rgb.astype(np.float32) * (1 - s) + restored.astype(np.float32) * s + 0.5,
                       0, 255).astype(np.uint8)


def ensure_gfpgan_installed():
    try:
        import gfpgan  # noqa: F401
        return
    except Exception:
        pass
    import importlib, tarfile, tempfile
    print("Installing the face-restoration add-on (GFPGAN). This takes 1-3 minutes, only once per session ...")
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    try:
        # basicsr 1.4.2 (needed by GFPGAN) has two known bugs with newer Python/torchvision.
        # We download its source, fix those two lines, and install the fixed copy.
        meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/basicsr/1.4.2/json", timeout=60))
        url = [u["url"] for u in meta["urls"] if u["filename"].endswith(".tar.gz")][0]
        tmp = tempfile.mkdtemp()
        tgz = os.path.join(tmp, "basicsr.tar.gz")
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        src_dir = os.path.join(tmp, "basicsr-1.4.2")
        setup_py = os.path.join(src_dir, "setup.py")
        s = open(setup_py).read()
        s = s.replace("return locals()['__version__']", "return '1.4.2'")
        s = s.replace("setup_requires=['cython', 'numpy', 'torch'],", "")
        open(setup_py, "w").write(s)
        deg = os.path.join(src_dir, "basicsr", "data", "degradations.py")
        d = open(deg).read().replace("torchvision.transforms.functional_tensor",
                                     "torchvision.transforms.functional")
        open(deg, "w").write(d)
    except Exception as e:
        raise UpscalerError(f"Could not download GFPGAN's helper package. Is Internet turned on?\n{e}\n"
                            "You can set FACE_RESTORE = False and continue.")
    steps = [pip + ["addict", "yapf", "lmdb", "filterpy", "scikit-image"],
             pip + ["--no-deps", "--no-build-isolation", src_dir],
             pip + ["--no-deps", "facexlib==0.3.0", "gfpgan==1.3.8"]]
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise UpscalerError("Could not install GFPGAN (face restoration). Is Internet turned on?\n"
                                + r.stderr[-600:] + "\nYou can set FACE_RESTORE = False and continue.")
    importlib.invalidate_caches()
    try:
        import gfpgan  # noqa: F401
    except Exception as e:
        raise UpscalerError(f"GFPGAN installed but would not start: {e}\n"
                            "You can set FACE_RESTORE = False and continue.")


def get_device(allow_cpu=False):
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if allow_cpu:
        return torch.device("cpu")
    raise UpscalerError(
        "No GPU found. The GPU is switched off.\n"
        "Fix: in the right-hand panel of the notebook open 'Session options' (or 'Settings'),\n"
        "set ACCELERATOR to 'GPU T4 x2', confirm, then run all cells again from the top.\n"
        "(If you don't see the option, verify your phone number in your Kaggle account settings.)")


_ENGINE_CACHE = {}


def build_engine(S, device, info):
    out_w, out_h = output_size(info, S.SCALE)
    key = (S.MODEL, S.DENOISE_STRENGTH, S.USE_HALF_PRECISION, S.STEADY_CHUNK_FRAMES, S.STEADY_WEIGHTS_FILE, str(device))
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        _ENGINE_CACHE.clear()
        free_memory()
        engine = (FastEngine if S.MODEL == "fast" else SteadyEngine)(S, device, out_w, out_h)
        _ENGINE_CACHE[key] = engine
    engine.finish = Finisher(out_w, out_h, S, device)   # new video size / blend / grain
    if S.MODEL == "steady":
        engine.calibrate(info)
    return engine


_FACE_CACHE = {}


def build_face(S, device):
    if not S.FACE_RESTORE:
        return None
    if "face" not in _FACE_CACHE:
        _FACE_CACHE["face"] = FaceRestorer(S, device)
    _FACE_CACHE["face"].strength = float(S.FACE_STRENGTH)
    return _FACE_CACHE["face"]


# -----------------------------------------------------------------------------
#  Finding videos, disk space, time limits
# -----------------------------------------------------------------------------
def find_videos(S, quiet=False):
    root = str(S.INPUT_FOLDER)
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        listing = ""
        if os.path.isdir("/kaggle/input"):
            found = sorted(glob.glob("/kaggle/input/*/") + glob.glob("/kaggle/input/*/*/"))[:20]
            listing = "\nFolders that DO exist:\n" + "\n".join("  " + f for f in found)
        raise UpscalerError(f"The INPUT_FOLDER does not exist:\n  {root}{listing}\n"
                            "Tip: leave INPUT_FOLDER = \"/kaggle/input\" and the notebook searches everywhere.")
    skip_dirs = {WORK_FOLDER_NAME, PREVIEW_FOLDER_NAME}
    videos = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS and "_upscaled_" not in f:
                videos.append(os.path.join(dirpath, f))
    videos.sort()
    if S.ONLY_THESE_FILES:
        wanted = {w.strip().lower() for w in S.ONLY_THESE_FILES}
        chosen = [v for v in videos if os.path.basename(v).lower() in wanted]
        missing = wanted - {os.path.basename(v).lower() for v in chosen}
        if missing and not quiet:
            print("⚠️  These names in ONLY_THESE_FILES were not found: " + ", ".join(sorted(missing)))
        videos = chosen
    if not videos and S.ONLY_THESE_FILES:
        raise UpscalerError("None of the names in ONLY_THESE_FILES were found.\n"
                            "Write the exact file name (e.g. \"holiday.mp4\"), or use ONLY_THESE_FILES = [] for all videos.")
    if not videos:
        sample = []
        for dirpath, _, filenames in os.walk(root):
            sample += [os.path.join(dirpath, f) for f in filenames]
            if len(sample) > 15:
                break
        raise UpscalerError(f"No videos found in {root}.\n"
                            "Did you add your video dataset to this notebook? (right panel → 'Add Input')\n"
                            + ("Files that ARE there:\n" + "\n".join("  " + s for s in sample[:15]) if sample
                               else "That folder is empty."))
    return videos


def folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def free_space(path):
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    if ON_KAGGLE and os.path.abspath(path).startswith("/kaggle/working"):
        free = min(free, int(KAGGLE_OUTPUT_LIMIT_GB * 1024 ** 3) - folder_size("/kaggle/working"))
    return max(0, free)


def session_start_time():
    try:
        import psutil
        return min(psutil.Process(1).create_time(), psutil.Process().create_time())
    except Exception:
        return _ENGINE_LOADED_AT


_ENGINE_LOADED_AT = time.time()


def estimate_mb_per_second(out_w, out_h, S):
    # Rough guide for H.264 at these quality settings (real size depends on the video)
    base = out_w * out_h / (1920 * 1080) * 1.6        # MB/s at CRF 17
    return base * (2 ** ((17 - S.VIDEO_QUALITY) / 6.0)) * (1.6 if S.FILM_GRAIN >= 0.02 else 1.0)


# -----------------------------------------------------------------------------
#  CHECK cell
# -----------------------------------------------------------------------------
@friendly
def check_setup(source=None):
    S = settings(source if source is not None else globals())
    print("1) GPU")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"   ✅ {torch.cuda.device_count()} GPU(s) found: {p.name}, {p.total_memory / 1024 ** 3:.0f} GB memory")
    else:
        get_device()  # raises the friendly "turn on the GPU" message

    print("2) ffmpeg (the video tool)")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("   Not found - installing it (about 1 minute) ...")
        subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not shutil.which("ffmpeg"):
        raise UpscalerError("Could not install ffmpeg. Is Internet turned on? (Session options → Internet ON)")
    ff = ffmpeg_info()
    print(f"   ✅ ffmpeg version {ff['major']} ready")

    print("3) Internet + model files")
    needed = ["general", "general_wdn"] if S.MODEL == "fast" else ["realbasicvsr"]
    if S.FACE_RESTORE:
        needed.append("gfpgan")
    for key in needed:
        path = get_weights(key, S.STEADY_WEIGHTS_FILE if key == "realbasicvsr" else "")
        print(f"   ✅ {os.path.basename(path)} ready")

    print("4) Disk space")
    free = free_space(S.OUTPUT_FOLDER)
    print(f"   ✅ about {human_size(free)} free for results (Kaggle keeps up to ~20 GB in /kaggle/working)")

    print("5) Your videos")
    videos = find_videos(S)
    for v in videos:
        info = probe_video(v)
        ow, oh = output_size(info, S.SCALE)
        est_gb = estimate_mb_per_second(ow, oh, S) * info["duration"] / 1024
        print(f"   🎬 {info['name']}: {info['width']}x{info['height']}, {float(info['fps']):.2f} fps, "
              f"{human_time(info['duration'])}, ~{info['est_frames']} frames")
        print(f"      → will become {ow}x{oh}. Expected file size roughly {est_gb:.1f} GB "
              f"(needs about twice that free while finishing).")
        if ow * oh > 8192 * 4320:
            print("      ❌ That is bigger than H.264 allows (8K). Lower SCALE for this video.")
        elif ow > 3840 or oh > 3840:
            print("      ⚠️  Output is bigger than 4K - slow, huge files. Consider a smaller SCALE.")
        if info["hdr"]:
            print("      ⚠️  This is an HDR video (e.g. iPhone). It will be converted to normal (SDR) colours.")
        if info["vfr"]:
            print("      ℹ️  Variable frame rate (common on phones) - it will be made a steady "
                  f"{float(info['fps']):.2f} fps.")
        if est_gb * 2 > free / 1024 ** 3:
            print("      ⚠️  Might not fit in the free space. Do one video at a time (ONLY_THESE_FILES), "
                  "or raise VIDEO_QUALITY to e.g. 20 for smaller files.")
    say("✅ Everything looks ready.", "Next: run the PREVIEW cell to test your settings on a short clip.")


# -----------------------------------------------------------------------------
#  PREVIEW cell
# -----------------------------------------------------------------------------
def put_label(img, text):
    try:
        import cv2
        scale = max(0.6, img.shape[0] / 900)
        th = max(1, int(scale * 2))
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        cv2.rectangle(img, (8, 8), (8 + tw + 16, 8 + tht + 20), (0, 0, 0), -1)
        cv2.putText(img, text, (16, 8 + tht + 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th,
                    cv2.LINE_AA)
    except Exception:
        pass
    return img


def side_by_side(orig_big, upscaled):
    """Same middle area of both pictures, next to each other, at 100% size."""
    h, w = upscaled.shape[:2]
    half = (w // 2) // 2 * 2
    x0 = (w - half) // 2
    left = np.ascontiguousarray(orig_big[:, x0:x0 + half])
    right = np.ascontiguousarray(upscaled[:, x0:x0 + half])
    put_label(left, "BEFORE (just stretched)")
    put_label(right, "AFTER (upscaled)")
    out = np.concatenate([left, right], axis=1)
    out[:, half - 1:half + 1] = 255
    return out


def mux_preview_audio(video_only, info, start, length, final):
    if not info["has_audio"]:
        os.replace(video_only, final)
        return
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", video_only,
           "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", info["path"],
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", final]
    if subprocess.run(cmd, capture_output=True).returncode == 0:
        os.remove(video_only)
    else:
        os.replace(video_only, final)


@friendly
def run_preview(source=None, allow_cpu=False):
    S = settings(source if source is not None else globals())
    if S.SKIP_PREVIEW_IN_SAVE_VERSION and os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() == "batch":
        print("⏭  Preview skipped (this is a 'Save Version' background run). Going on to the full upscale.")
        return
    device = get_device(allow_cpu)
    videos = find_videos(S)
    if S.PREVIEW_VIDEO:
        match = [v for v in videos if os.path.basename(v).lower() == str(S.PREVIEW_VIDEO).strip().lower()]
        if not match:
            raise UpscalerError(f"PREVIEW_VIDEO '{S.PREVIEW_VIDEO}' not found. Videos found:\n"
                                + "\n".join("  " + os.path.basename(v) for v in videos))
        video = match[0]
    else:
        video = videos[0]
    info = probe_video(video)
    start = min(float(S.PREVIEW_START_SECONDS), max(0.0, info["duration"] - 1))
    length = min(float(S.PREVIEW_LENGTH_SECONDS), max(1.0, info["duration"] - start)) if info["duration"] else float(S.PREVIEW_LENGTH_SECONDS)
    out_w, out_h = output_size(info, S.SCALE)
    if out_w * out_h > 8192 * 4320:
        raise UpscalerError(f"{out_w}x{out_h} is too big for H.264 video. Lower SCALE.")

    print(f"Preview of '{info['name']}' from {start:g}s, {length:g}s long, "
          f"{info['width']}x{info['height']} → {out_w}x{out_h}, settings: {settings_label(S)}")
    engine = build_engine(S, device, info)
    face = build_face(S, device)

    pdir = os.path.join(S.OUTPUT_FOLDER, PREVIEW_FOLDER_NAME)
    os.makedirs(pdir, exist_ok=True)
    base = os.path.join(pdir, f"{info['stem']}_{settings_label(S)}_at{int(start)}s")
    log = base + ".log"
    reader = FrameReader(decode_command(info, seek_seconds=start, length_seconds=length),
                         info["width"], info["height"], log)
    w_up = VideoWriter(base + "_tmp_up.mp4", out_w, out_h, info["fps_str"], S, log)
    cmp_w = (out_w // 2) // 2 * 2 * 2
    w_cmp = VideoWriter(base + "_tmp_cmp.mp4", cmp_w, out_h, info["fps_str"], S, log)

    expected = max(1, int(round(length * float(info["fps"]))))
    middle = expected // 2
    still = None
    done = 0
    before = []
    t0 = time.time()
    bar = tqdm(total=expected, unit="frame", desc="Preview")
    try:
        while True:
            core = reader.read_n(engine.step)
            if not core:
                break
            after = reader.peek_n(engine.context) if engine.context else []
            outs = engine.run(before, core, after)
            for lr, up in zip(core, outs):
                if face:
                    up = face(up)
                big = engine.finish.original_upscaled(frames_to_tensor([lr], device))
                big = (big[0] * 255 + 0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                cmp_frame = side_by_side(big, up)
                w_up.write(up)
                w_cmp.write(cmp_frame)
                if done == middle or still is None:
                    still = cmp_frame
                done += 1
                bar.update(1)
            before = (before + core)[-engine.context:] if engine.context else []
        reader.check_error()
        w_up.close()
        w_cmp.close()
    except BaseException:
        w_up.abort()
        w_cmp.abort()
        raise
    finally:
        bar.close()
        reader.close()
    if done == 0:
        raise UpscalerError("No frames could be read for the preview. Is PREVIEW_START_SECONDS past the end "
                            f"of the video? (video length: {info['duration']:.1f} s)")
    took = time.time() - t0
    speed = done / max(took, 1e-6)

    mux_preview_audio(base + "_tmp_up.mp4", info, start, length, base + "_upscaled.mp4")
    os.replace(base + "_tmp_cmp.mp4", base + "_compare.mp4")
    png = base + "_compare.png"
    try:
        import cv2
        cv2.imwrite(png, still[:, :, ::-1])
    except Exception:
        png = None
    if os.path.exists(log) and os.path.getsize(log) == 0:
        os.remove(log)

    full_time = info["est_frames"] / speed
    say("✅ Preview finished",
        f"Speed: {speed:.2f} frames per second  (took {human_time(took)} for {done} frames)",
        f"Estimated time for the WHOLE video '{info['name']}': about {human_time(full_time)}",
        ("⚠️  That is longer than one Kaggle session - no problem, it will stop safely and you continue "
         "in a new session (see the guide)." if full_time > S.MAX_RUNTIME_HOURS * 3600 else ""),
        "Files (right panel → Output → /kaggle/working/previews):",
        f"  • {os.path.basename(base)}_upscaled.mp4  - the upscaled clip",
        f"  • {os.path.basename(base)}_compare.mp4   - before | after, the middle part, at full size",
        f"  • {os.path.basename(base)}_compare.png   - one still frame of the same (shown below)",
        "Look at skin, hair, grass, text, and dark areas. Too smooth/plastic → lower DENOISE_STRENGTH",
        "or raise ORIGINAL_BLEND. Too noisy → raise DENOISE_STRENGTH. Flicker → try MODEL = \"steady\".")
    if png:
        try:
            from IPython.display import Image, display
            display(Image(filename=png, width=1100))
        except Exception:
            pass


# -----------------------------------------------------------------------------
#  UPSCALE cell (full videos, with saved progress)
# -----------------------------------------------------------------------------
def load_progress(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_progress(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def bring_back_earlier_work(work_dir, final_name, out_dir):
    """If you attached the Output of an earlier run as Input, copy its saved progress back."""
    if not os.path.isdir("/kaggle/input"):
        return False
    for p in glob.glob(f"/kaggle/input/**/{final_name}", recursive=True):
        if os.path.getsize(p) > 0:
            print(f"Found the finished video from an earlier run: {p} - copying it to the output.")
            shutil.copy2(p, os.path.join(out_dir, final_name))
            return True
    name = os.path.basename(work_dir)
    mine = load_progress(os.path.join(work_dir, "progress.json")) or {}
    best = None
    for p in glob.glob(f"/kaggle/input/**/{WORK_FOLDER_NAME}/{name}/progress.json", recursive=True):
        data = load_progress(p) or {}
        if data.get("frames_done", 0) > max(mine.get("frames_done", 0), (best or ({}, 0))[0].get("frames_done", 0)):
            best = (data, os.path.dirname(p))
    if best:
        data, src = best
        print(f"Found saved progress from an earlier run ({data['frames_done']} frames done). "
              "Copying it so we can continue ...")
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.copytree(src, work_dir)
    return False


def assemble(work_dir, progress, info, final_path):
    segs = [os.path.join(work_dir, s["file"]) for s in progress["segments"]]
    lst = os.path.join(work_dir, "segments.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.basename(s)}'\n")
    tmp = final_path + ".partial.mp4"
    base = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", lst]
    attempts = []
    if info["has_audio"]:
        audio_in = ["-i", info["path"], "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1"]
        attempts.append(audio_in + ["-c:v", "copy", "-c:a", "copy"])
        attempts.append(audio_in + ["-c:v", "copy", "-c:a", "aac", "-b:a", "256k"])
    attempts.append(["-map", "0:v:0", "-c:v", "copy"])
    err = ""
    for i, extra in enumerate(attempts):
        r = subprocess.run(base + extra + ["-movflags", "+faststart", tmp], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            if info["has_audio"] and i == len(attempts) - 1:
                print("⚠️  The sound could not be added - the video was saved WITHOUT sound.")
            os.replace(tmp, final_path)
            return
        err = r.stderr
    raise UpscalerError("Could not join the finished parts into the final video.\n" + err[-600:]
                        + "\nThe parts are still saved - free some disk space and run the cell again.")


def process_video(video, S, device, deadline):
    info = probe_video(video)
    out_w, out_h = output_size(info, S.SCALE)
    if out_w * out_h > 8192 * 4320:
        raise UpscalerError(f"'{info['name']}' would become {out_w}x{out_h}, which is too big for H.264. "
                            "Lower SCALE.")
    out_dir = S.OUTPUT_FOLDER
    os.makedirs(out_dir, exist_ok=True)
    final_name = f"{info['stem']}_upscaled_{settings_label(S)}.mp4"
    final_path = os.path.join(out_dir, final_name)
    if os.path.exists(final_path):
        print(f"✅ '{final_name}' already exists - skipping. (Delete it to redo.)")
        return "done"
    work_root = os.path.join(out_dir, WORK_FOLDER_NAME)
    work = os.path.join(work_root, f"{info['stem']}__{settings_key(S)}")
    if bring_back_earlier_work(work, final_name, out_dir):
        return "done"
    os.makedirs(work, exist_ok=True)
    for other in glob.glob(os.path.join(work_root, f"{info['stem']}__*")):
        if other != work:
            print(f"ℹ️  There is saved progress for '{info['name']}' with DIFFERENT settings "
                  f"({human_size(folder_size(other))}). It is kept; delete it with the CLEAN-UP cell if you "
                  "don't need it.")

    ppath = os.path.join(work, "progress.json")
    progress = load_progress(ppath) or dict(video=info["name"], settings=settings_label(S), frames_done=0,
                                            segments=[], complete=False)
    # forget any part whose file went missing; delete half-written parts
    kept, frames = [], 0
    for s in progress["segments"]:
        if os.path.exists(os.path.join(work, s["file"])):
            kept.append(s)
            frames += s["frames"]
        else:
            break
    progress["segments"], progress["frames_done"] = kept, frames
    known = {s["file"] for s in kept}
    for f in glob.glob(os.path.join(work, "seg_*.mp4")):
        if os.path.basename(f) not in known:
            os.remove(f)

    print(f"\n🎬 {info['name']}: {info['width']}x{info['height']} → {out_w}x{out_h}  ({settings_label(S)})")
    if progress["frames_done"]:
        print(f"   Continuing from saved progress: {progress['frames_done']} of ~{info['est_frames']} frames done.")

    if not progress["complete"]:
        engine = build_engine(S, device, info)
        face = build_face(S, device)
        seg_frames = max(1, int(round(float(S.SAVE_EVERY_SECONDS) * float(info["fps"]))))
        start = progress["frames_done"]
        ctx = engine.context
        rstart = max(0, start - ctx)
        if rstart > 0:
            print("   Skipping ahead to where we stopped (can take a few minutes for long videos) ...")
        log = os.path.join(work, "ffmpeg.log")
        reader = FrameReader(decode_command(info, start_frame=rstart), info["width"], info["height"], log)
        before = reader.read_n(start - rstart)
        bar = tqdm(total=max(info["est_frames"], start + 1), initial=start, unit="frame",
                   desc=info["stem"][:30], smoothing=0.05)
        seg_times = []
        status = "done"
        try:
            while True:
                avg = sum(seg_times[-5:]) / len(seg_times[-5:]) if seg_times else 0
                if time.time() + avg * 1.3 > deadline:
                    status = "paused"
                    break
                need = (estimate_mb_per_second(out_w, out_h, S) * float(S.SAVE_EVERY_SECONDS) * 3
                        * 1024 ** 2 + 300 * 1024 ** 2)
                if free_space(out_dir) < need:
                    status = "disk"
                    break
                seg_name = f"seg_{len(progress['segments']):05d}.mp4"
                writer = VideoWriter(os.path.join(work, seg_name), out_w, out_h, info["fps_str"], S, log)
                t0 = time.time()
                got = 0
                try:
                    while got < seg_frames:
                        k = min(engine.step, seg_frames - got)
                        core = reader.read_n(k)
                        if not core:
                            break
                        after = reader.peek_n(ctx) if ctx else []
                        for up in engine.run(before, core, after):
                            writer.write(face(up) if face else up)
                        bar.update(len(core))
                        before = (before + core)[-ctx:] if ctx else []
                        got += len(core)
                        if len(core) < k:
                            break
                except BaseException:
                    writer.abort()
                    raise
                if got == 0:
                    writer.abort()
                    reader.check_error()
                    progress["complete"] = True
                    break
                writer.close()
                progress["segments"].append(dict(file=seg_name, frames=got))
                progress["frames_done"] += got
                save_progress(ppath, progress)
                seg_times.append(time.time() - t0)
                if got < seg_frames:
                    reader.check_error()
                    progress["complete"] = True
                    break
        finally:
            bar.close()
            reader.close()
        save_progress(ppath, progress)
        if progress["frames_done"] == 0 and progress["complete"]:
            raise UpscalerError(f"No pictures could be read from '{info['name']}'.\n" + log_tail(log))
        if not progress["complete"]:
            return status

    need = folder_size(work) * 1.05 + 200 * 1024 ** 2
    if free_space(out_dir) < need:
        raise UpscalerError(f"Not enough disk space to join the parts (needs about {human_size(need)}).\n"
                            "Download and delete other finished videos (or previews), then run this cell again.\n"
                            "Your progress is saved.")
    print("   Joining the parts and adding the original sound ...")
    assemble(work, progress, info, final_path)
    shutil.rmtree(work, ignore_errors=True)
    if os.path.isdir(work_root) and not os.listdir(work_root):
        os.rmdir(work_root)
    print(f"   ✅ Saved: {final_path}  ({human_size(os.path.getsize(final_path))})")
    return "done"


@friendly
def run_upscale(source=None, allow_cpu=False):
    S = settings(source if source is not None else globals())
    device = get_device(allow_cpu)
    videos = find_videos(S)
    deadline = session_start_time() + float(S.MAX_RUNTIME_HOURS) * 3600 - 10 * 60
    left = deadline - time.time()
    print(f"{len(videos)} video(s) to do. Time left in this session (safe limit): {human_time(left)}")
    results = {}
    for v in videos:
        status = process_video(v, S, device, deadline)
        results[os.path.basename(v)] = status
        if status == "paused":
            say("⏸  Stopped safely before Kaggle's time limit. Progress is SAVED.",
                "To continue: see 'If the time runs out' in the guide. In short: start a new session,",
                "attach this run's Output as Input (if it was a 'Save Version' run), and run all cells again.")
            return
        if status == "disk":
            say("⏸  Stopped because the disk is almost full. Progress is SAVED.",
                "Download finished videos from the Output panel, delete them (or run the CLEAN-UP cell),",
                "then run this cell again. Tip: a higher VIDEO_QUALITY number (e.g. 20) makes smaller files.")
            return
    say("🎉 All done!", "Your videos are in /kaggle/working (right panel → Output). Download them there.",
        "Remember: in a normal (interactive) session, files are DELETED when the session ends,",
        "so download them first - or use 'Save Version' so Kaggle keeps them.")


@friendly
def clean_up(delete_previews=False, delete_progress=False, source=None):
    S = settings(source if source is not None else globals())
    targets = []
    if delete_previews:
        targets.append(os.path.join(S.OUTPUT_FOLDER, PREVIEW_FOLDER_NAME))
    if delete_progress:
        targets.append(os.path.join(S.OUTPUT_FOLDER, WORK_FOLDER_NAME))
    for t in targets:
        if os.path.isdir(t):
            size = folder_size(t)
            shutil.rmtree(t, ignore_errors=True)
            print(f"Deleted {t} ({human_size(size)})")
    print(f"Free space for results now: about {human_size(free_space(S.OUTPUT_FOLDER))}")
    for root, _, files in os.walk(S.OUTPUT_FOLDER):
        for f in sorted(files):
            if f.endswith(".mp4") or f.endswith(".png"):
                p = os.path.join(root, f)
                print(f"  {human_size(os.path.getsize(p)):>9}  {p}")


print("✅ Engine loaded. Next: run the CHECK cell.")
