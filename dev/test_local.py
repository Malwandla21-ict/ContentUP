"""Local test of the engine on CPU with tiny synthetic videos.

Run:  python dev/test_local.py   (needs torch, numpy, opencv, tqdm, ffmpeg)
It checks: frame-exact resume, fast model output, preview files, pause+continue,
steady model (with random weights - only checks the plumbing), tiling, HDR/rotation input.
"""
import os, sys, json, shutil, subprocess, tempfile, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

TMP = os.environ.get("TEST_DIR") or tempfile.mkdtemp(prefix="upscaler_test_")
VID = os.path.join(TMP, "in")
OUT = os.path.join(TMP, "out")
os.makedirs(VID, exist_ok=True)
if os.environ.get("MODEL_DIR"):
    E.MODEL_DIR = os.environ["MODEL_DIR"]


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-print_format", "json", "-show_streams", path],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"]


def decode_all(path):
    info = E.probe_video(path)
    r = subprocess.run(E.decode_command(info), capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, info["height"], info["width"], 3)


def base_settings(**kw):
    s = dict(E.DEFAULTS)
    s.update(INPUT_FOLDER=VID, OUTPUT_FOLDER=OUT, ONLY_THESE_FILES=["clip.mp4"], ENCODER_SPEED="ultrafast")
    s.update(kw)
    return s


def main():
    clip = os.path.join(VID, "clip.mp4")
    ff("-f", "lavfi", "-i", "testsrc2=size=160x96:rate=30000/1001:duration=4",
       "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", "-colorspace", "bt709", "-c:a", "aac", "-shortest", clip)
    rot = os.path.join(VID, "rotated.mp4")
    ff("-display_rotation", "90", "-i", clip, "-c", "copy", rot)
    hdr = os.path.join(VID, "hdr.mp4")
    ff("-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25:duration=1", "-c:v", "libx265", "-x265-params", "log-level=error",
       "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
       "-colorspace", "bt2020nc", hdr)

    # 1) probing: rotation + HDR
    ri = E.probe_video(rot)
    assert (ri["width"], ri["height"]) == (96, 160), ri
    assert decode_all(rot).shape[1:3] == (160, 96)
    hi = E.probe_video(hdr)
    assert hi["hdr"], hi
    hf = decode_all(hdr)
    assert hf.shape[0] == 25 and hf.mean() > 20, (hf.shape, hf.mean())
    print("OK probe/rotation/HDR", ri["fps_str"], hi["fps_str"])

    # 2) frame-exact restart in the middle (what 'continue' relies on)
    info = E.probe_video(clip)
    allf = decode_all(clip)
    assert len(allf) == info["est_frames"], (len(allf), info["est_frames"])
    for k in (1, 37, 64, 110):
        r = subprocess.run(E.decode_command(info, start_frame=k), capture_output=True, check=True)
        part = np.frombuffer(r.stdout, np.uint8).reshape(-1, info["height"], info["width"], 3)
        assert len(part) == len(allf) - k and np.array_equal(part[0], allf[k]) and np.array_equal(part[-1], allf[-1]), k
    print("OK frame-exact restart")

    dev = torch.device("cpu")
    # 3) fast model: weights load, tiles match no tiles, looks sensible
    S = E.settings(base_settings())
    eng = E.build_engine(S, dev, info)
    frames = list(allf[:2])
    a = eng.run([], frames, [])
    eng.tile = 64
    b = eng.run([], frames, [])
    eng.tile = None
    diff = np.abs(a[0].astype(int) - b[0].astype(int)).max()
    assert diff <= 2, diff
    big = torch.nn.functional.interpolate(E.frames_to_tensor(frames[:1], dev), scale_factor=2, mode="bicubic")
    big = (big[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    mad = np.abs(a[0].astype(int) - big.astype(int)).mean()
    assert a[0].shape == (192, 320, 3) and mad < 25, (a[0].shape, mad)
    print(f"OK fast model (tile diff {diff}, mean diff vs bicubic {mad:.1f})")

    # 4) preview
    E.run_preview(base_settings(PREVIEW_START_SECONDS=1, PREVIEW_LENGTH_SECONDS=2), allow_cpu=True)
    pdir = os.path.join(OUT, "previews")
    files = sorted(os.listdir(pdir))
    print(files)
    up = [f for f in files if f.endswith("_upscaled.mp4")][0]
    st = probe(os.path.join(pdir, up))
    v = [s for s in st if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (320, 192) and abs(int(v["nb_read_frames"]) - 60) <= 1, v
    assert any(s["codec_type"] == "audio" for s in st)
    assert any(f.endswith("_compare.png") for f in files) and any(f.endswith("_compare.mp4") for f in files)
    print("OK preview")

    # 5) full run, interrupted by the time limit, then continued; compare with an uninterrupted run
    Sx = E.settings(base_settings(SAVE_EVERY_SECONDS=1))
    st1 = E.process_video(clip, Sx, dev, deadline=time.time() + 1.5)
    work_root = os.path.join(OUT, E.WORK_FOLDER_NAME)
    prog = json.load(open(next(os.path.join(r, f) for r, _, fs in os.walk(work_root) for f in fs if f == "progress.json")))
    print("first run:", st1, prog["frames_done"], "frames,", len(prog["segments"]), "parts")
    assert st1 == "paused" and 0 < prog["frames_done"] < len(allf)
    st2 = E.process_video(clip, Sx, dev, deadline=time.time() + 3600)
    assert st2 == "done"
    final = os.path.join(OUT, f"clip_upscaled_{E.settings_label(Sx)}.mp4")
    st = probe(final)
    v = [s for s in st if s["codec_type"] == "video"][0]
    assert int(v["nb_read_frames"]) == len(allf), v["nb_read_frames"]
    assert v["r_frame_rate"] == "30000/1001", v["r_frame_rate"]
    assert v.get("color_space") == "bt709"
    assert any(s["codec_type"] == "audio" for s in st)
    assert not os.path.exists(work_root), "parts should be cleaned up"
    resumed = decode_all(final)
    os.rename(final, final + ".resumed.mp4")
    E.process_video(clip, Sx, dev, deadline=time.time() + 3600)
    straight = decode_all(final)
    assert np.array_equal(resumed, straight), "resumed result differs from straight result"
    # colour check: upscaled frames should match original colours closely on average
    small = torch.nn.functional.interpolate(torch.from_numpy(straight[:10]).permute(0, 3, 1, 2).float(),
                                            size=(96, 160), mode="area")
    cd = (small - torch.from_numpy(allf[:10]).permute(0, 3, 1, 2).float()).mean(dim=(0, 2, 3))
    assert cd.abs().max() < 3, cd
    print("OK pause + continue gives identical video; colour shift per channel:", cd.numpy().round(2))

    # 6) steady model plumbing with random weights in the official checkpoint layout
    torch.manual_seed(0)
    net = E.RealBasicVSRNet()
    for p in net.parameters():
        p.data.mul_(0.1)
    ck = {"meta": {}, "state_dict": {**{"generator." + k: v for k, v in net.state_dict().items()},
                                      **{"generator_ema." + k: v for k, v in net.state_dict().items()},
                                      "step_counter": torch.zeros(1)}}
    wpath = os.path.join(TMP, "fake_realbasicvsr.pth")
    torch.save(ck, wpath)
    Ss = base_settings(MODEL="steady", STEADY_WEIGHTS_FILE=wpath, STEADY_CHUNK_FRAMES=8, SAVE_EVERY_SECONDS=1,
                       FILM_GRAIN=0.02, ORIGINAL_BLEND=0.2, SCALE=1.5)
    S2 = E.settings(Ss)
    eng = E.build_engine(S2, dev, info)
    print("cleaning passes decided:", eng.passes)
    frames = list(allf[:14])
    o1 = eng.run(frames[:3], frames[3:11], frames[11:14])
    eng.tile = 64
    o2 = eng.run(frames[:3], frames[3:11], frames[11:14])
    eng.tile = None
    assert len(o1) == 8 and o1[0].shape == (144, 240, 3)
    # grain differs per run, so compare loosely
    td = np.abs(np.stack(o1).astype(int) - np.stack(o2).astype(int)).mean()
    assert td < 8, td
    st = E.process_video(clip, S2, dev, deadline=time.time() + 3600)
    final = os.path.join(OUT, f"clip_upscaled_{E.settings_label(S2)}.mp4")
    v = [s for s in probe(final) if s["codec_type"] == "video"][0]
    assert int(v["nb_read_frames"]) == len(allf) and (v["width"], v["height"]) == (240, 144), v
    print("OK steady plumbing (tile mean diff %.2f)" % td)

    # 7) HDR + rotated videos go through the whole pipeline
    E.run_upscale(base_settings(ONLY_THESE_FILES=["hdr.mp4", "rotated.mp4"], ENCODER_SPEED="ultrafast"), allow_cpu=True)
    for name, size in (("hdr", (256, 192)), ("rotated", (192, 320))):
        f = [x for x in os.listdir(OUT) if x.startswith(name + "_upscaled")][0]
        v = [s for s in probe(os.path.join(OUT, f)) if s["codec_type"] == "video"][0]
        assert (v["width"], v["height"]) == size, (name, v["width"], v["height"])
    print("OK HDR + rotated")

    # 8) friendly errors
    E.run_upscale(base_settings(INPUT_FOLDER="/nope"), allow_cpu=True)
    E.run_upscale(base_settings(MODEL="slow"), allow_cpu=True)
    E.run_upscale(base_settings(ONLY_THESE_FILES=["missing.mp4"]), allow_cpu=True)
    print("\nALL TESTS PASSED in", TMP)


if __name__ == "__main__":
    main()
