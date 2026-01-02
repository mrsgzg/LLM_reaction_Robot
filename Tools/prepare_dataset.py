"""
Dataset preparation script for Speaker→Listener AU Prediction
Converts raw video, audio, and AU annotation data into Hugging Face Dataset format.

Task: Predict listener's AU changes from speaker's video + audio

Key assumptions:
- Splits: train/ and val/ directories (val is validation split).
- Speaker and Listener are paired by the same basename in the same session.
- Each sample includes:
  * Input (Speaker): speaker_video_path, speaker_audio_path
  * Target (Listener): listener_au_prob, listener_au_act, listener_video_path, listener_audio_path
- Directory structure:
    <split>/<session>/
    ├── video-face-crop/speaker/*.mp4, listener/*.mp4
    ├── audio/speaker/*.wav, listener/*.wav
    └── AU_Continue(or AU_continue)/speaker/*.csv, listener/*.csv
- File pairing uses basename: <stem>.mp4, <stem>.wav, <stem>_AUs.csv
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from datasets import Dataset  # type: ignore
except Exception:
    Dataset = None  # type: ignore
from tqdm import tqdm


def _exists(p: Optional[Path]) -> bool:
    return p is not None and p.exists()


def _try_cv2_video_meta(video_path: Path) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None, None, None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None, None, None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0:
            return None, None, frame_count if frame_count > 0 else None
        duration = frame_count / fps if frame_count > 0 else None
        return float(fps), float(duration) if duration is not None else None, frame_count if frame_count > 0 else None
    except Exception:
        return None, None, None


def _try_ffprobe_video_meta(video_path: Path) -> Tuple[Optional[float], Optional[float]]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip().splitlines()
        if len(out) >= 2:
            fr = out[0]
            if "/" in fr:
                num, den = fr.split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
            else:
                try:
                    fps = float(fr)
                except ValueError:
                    fps = None
            try:
                duration = float(out[1])
            except ValueError:
                duration = None
            return fps, duration
    except Exception:
        pass
    return None, None


def explore_data_structure(data_root: Path) -> Dict[str, any]:
    """
    Explore and print the actual data structure.
    Returns information about file formats and naming patterns.
    """
    info: Dict[str, any] = {}

    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        return info

    print(f"Data root: {data_root}")
    splits = []
    for name in ["train", "val"]:
        p = data_root / name
        if p.exists():
            splits.append(name)
    print(f"Found splits: {splits}")
    info["splits"] = splits

    modalities = {
        "video": ["video-face-crop"],
        "audio": ["audio"],
        "anno": ["AU_Continue", "AU_continue"],
    }

    for s in splits:
        print(f"\n-- Split: {s}")
        split_root = data_root / s
        video_root = next((split_root / m / "listener" for m in modalities["video"] if (split_root / m / "listener").exists()), None)
        audio_root = next((split_root / m / "listener" for m in modalities["audio"] if (split_root / m / "listener").exists()), None)
        anno_root = None
        for m in modalities["anno"]:
            maybe = split_root / m / "listener"
            if maybe.exists():
                anno_root = maybe
                break

        print(f"video_root: {video_root}")
        print(f"audio_root: {audio_root}")
        print(f"anno_root:  {anno_root}")

        for label, root in [("video", video_root), ("audio", audio_root), ("anno", anno_root)]:
            if not _exists(root):
                continue
            sessions = sorted([p for p in root.iterdir() if p.is_dir()])
            print(f"  {label}: sessions={len(sessions)}")
            if sessions:
                s0 = sessions[0]
                files = sorted(list(s0.iterdir()))[:5]
                print(f"    sample session: {s0.name}, files: {[f.name for f in files]}")
        info[s] = {
            "video_root": str(video_root) if video_root else None,
            "audio_root": str(audio_root) if audio_root else None,
            "anno_root": str(anno_root) if anno_root else None,
        }

    # Infer naming pattern using first available video
    example = None
    for s in splits:
        vr = info[s].get("video_root") if s in info else None
        if not vr:
            continue
        vrp = Path(vr)
        any_session = next((p for p in sorted(vrp.iterdir()) if p.is_dir()), None)
        if any_session:
            vids = sorted(any_session.glob("*.mp4"))
            if vids:
                example = vids[0]
                break
    if example is not None:
        stem = example.stem
        print(f"\nExample basename: {stem}")
        print("Expected pairing:")
        print(f"  video: {stem}.mp4")
        print(f"  audio: {stem}.wav")
        print(f"  anno : {stem}_AUs.csv")
        info["naming"] = {"video_ext": ".mp4", "audio_ext": ".wav", "anno_suffix": "_AUs.csv"}

        for s in splits:
            anno_root = info[s].get("anno_root") if s in info else None
            if anno_root:
                session_dirs = sorted([p for p in Path(anno_root).iterdir() if p.is_dir()])
                for sd in session_dirs:
                    candidate = sd / f"{stem}_AUs.csv"
                    if candidate.exists():
                        print(f"\nPeek annotation CSV: {candidate}")
                        try:
                            with candidate.open("r", encoding="utf-8", errors="ignore") as fh:
                                for i, line in enumerate(fh):
                                    print(line.strip())
                                    if i >= 4:
                                        break
                        except Exception as e:
                            print(f"  [WARN] Could not read CSV: {e}")
                        break
                break
    return info


def _read_au_csv(anno_path: Path) -> Optional[Dict[str, any]]:
    try:
        df = pd.read_csv(anno_path)
    except Exception as e:
        print(f"[WARN] Failed to read annotation CSV {anno_path}: {e}")
        return None

    if df.empty:
        return None

    df.columns = [c.strip() for c in df.columns]
    au_prob_cols = [c for c in df.columns if c.endswith("_prob")]
    au_names = [c[:-5] for c in au_prob_cols]
    frame_idx = df["frame_idx"].astype(int).tolist() if "frame_idx" in df.columns else list(range(len(df)))

    au_prob: Dict[str, List[float]] = {}
    au_act: Dict[str, List[int]] = {}
    for an in au_names:
        pcol = f"{an}_prob"
        acol = f"{an}_act"
        if pcol in df.columns:
            try:
                au_prob[an] = df[pcol].astype(float).tolist()
            except Exception:
                au_prob[an] = df[pcol].tolist()
        if acol in df.columns:
            try:
                au_act[an] = df[acol].astype(int).tolist()
            except Exception:
                au_act[an] = df[acol].tolist()

    return {
        "frame_idx": frame_idx,
        "au_names": au_names,
        "au_prob": au_prob,
        "au_act": au_act,
        "n_frames": len(frame_idx),
    }


def load_sample(
    speaker_video_path: Path,
    speaker_audio_path: Path,
    listener_video_path: Path,
    listener_audio_path: Path,
    listener_anno_path: Path,
) -> Optional[Dict[str, any]]:
    """Load paired speaker and listener data for AU prediction.
    
    Args:
        speaker_video_path: Speaker's face video (INPUT)
        speaker_audio_path: Speaker's audio (INPUT)
        listener_video_path: Listener's face video (TARGET)
        listener_audio_path: Listener's audio (TARGET, for context)
        listener_anno_path: Listener's AU annotation CSV (LABEL)
    
    Returns:
        Dict with speaker inputs and listener AU targets
    """
    # Check all required files exist
    all_paths = [speaker_video_path, speaker_audio_path, listener_video_path, listener_audio_path, listener_anno_path]
    if not all(p.exists() for p in all_paths):
        missing = [p.name for p in all_paths if not p.exists()]
        print(f"[WARN] Missing files: {', '.join(missing)}")
        return None

    # Load listener AU annotations (TARGET LABELS)
    listener_au = _read_au_csv(listener_anno_path)
    if listener_au is None:
        return None

    # Get video metadata from listener video
    fps, duration, vframes = _try_cv2_video_meta(listener_video_path)
    if fps is None or duration is None:
        ffps, fdur = _try_ffprobe_video_meta(listener_video_path)
        if fps is None:
            fps = ffps
        if duration is None:
            duration = fdur

    if vframes is None and fps is not None and listener_au.get("n_frames"):
        try:
            vframes = int(listener_au["n_frames"])
        except Exception:
            vframes = None

    return {
        # Sample identifier
        "id": listener_video_path.stem,
        
        # ===== INPUT: Speaker Modalities =====
        "speaker_video_path": str(speaker_video_path),
        "speaker_audio_path": str(speaker_audio_path),
        
        # ===== TARGET: Listener Modalities (for context/reference) =====
        "listener_video_path": str(listener_video_path),
        "listener_audio_path": str(listener_audio_path),
        
        # ===== LABEL: Listener AU Changes (frame-level) =====
        "listener_au_names": listener_au.get("au_names"),
        "listener_au_prob": listener_au.get("au_prob"),
        "listener_au_act": listener_au.get("au_act"),
        "listener_frame_idx": listener_au.get("frame_idx"),
        
        # Metadata
        "fps": fps,
        "duration": duration,
        "n_frames": listener_au.get("n_frames"),
    }


def _gather_split_candidates(data_root: Path, split: str) -> List[Tuple[Path, Path, Path, Path, Path]]:
    """Gather paired (speaker_video, speaker_audio, listener_video, listener_audio, listener_anno) tuples.
    
    Returns:
        List of 5-tuples: (speaker_video, speaker_audio, listener_video, listener_audio, listener_anno)
    """
    quintuples: List[Tuple[Path, Path, Path, Path, Path]] = []
    split_root = data_root / split
    if not split_root.exists():
        return quintuples

    # Detect anno root name (AU_Continue or AU_continue)
    anno_root_base = split_root / "AU_Continue"
    if not anno_root_base.exists():
        anno_root_base = split_root / "AU_continue"
    if not anno_root_base.exists():
        print(f"[WARN] Missing AU annotation root for split '{split}'. Skipping.")
        return quintuples

    # Get session directories from listener's video folder (reference)
    listener_video_root = split_root / "video-face-crop" / "listener"
    if not listener_video_root.exists():
        print(f"[WARN] Missing listener video root for split '{split}'. Skipping.")
        return quintuples

    listener_audio_root = split_root / "audio" / "listener"
    if not listener_audio_root.exists():
        print(f"[WARN] Missing listener audio root for split '{split}'. Skipping.")
        return quintuples

    speaker_video_root = split_root / "video-face-crop" / "speaker"
    speaker_audio_root = split_root / "audio" / "speaker"
    if not speaker_video_root.exists() or not speaker_audio_root.exists():
        print(f"[WARN] Missing speaker video/audio root for split '{split}'. Skipping.")
        return quintuples

    listener_anno_root = anno_root_base / "listener"
    if not listener_anno_root.exists():
        print(f"[WARN] Missing listener annotation root for split '{split}'. Skipping.")
        return quintuples

    # Iterate through listener sessions (reference for pairing)
    listener_sessions = sorted([p for p in listener_video_root.iterdir() if p.is_dir()])
    for l_vsess in listener_sessions:
        session_name = l_vsess.name
        l_asess = listener_audio_root / session_name
        l_nsess = listener_anno_root / session_name
        s_vsess = speaker_video_root / session_name
        s_asess = speaker_audio_root / session_name

        if not (l_asess.exists() and l_nsess.exists() and s_vsess.exists() and s_asess.exists()):
            print(f"[WARN] Missing session dirs in '{session_name}' for split '{split}'. Skipping session.")
            continue

        # Iterate through listener videos (reference for pairing by basename)
        for l_vfile in sorted(l_vsess.glob("*.mp4")):
            stem = l_vfile.stem
            l_afile = l_asess / f"{stem}.wav"
            l_nfile = l_nsess / f"{stem}_AUs.csv"
            s_vfile = s_vsess / f"{stem}.mp4"
            s_afile = s_asess / f"{stem}.wav"

            # Check all required files exist
            if all(p.exists() for p in [l_afile, l_nfile, s_vfile, s_afile]):
                quintuples.append((s_vfile, s_afile, l_vfile, l_afile, l_nfile))
            else:
                missing = []
                if not l_afile.exists():
                    missing.append("listener_audio")
                if not l_nfile.exists():
                    missing.append("listener_anno")
                if not s_vfile.exists():
                    missing.append("speaker_video")
                if not s_afile.exists():
                    missing.append("speaker_audio")
                print(f"[WARN] Missing {','.join(missing)} for {stem} in session {session_name} ({split}).")
    return quintuples


def _split_indices(n: int) -> Tuple[List[int], List[int], List[int]]:
    # Placeholder retained for compatibility; no split needed here.
    return list(range(n)), [], []


def print_dataset_stats(dataset: Dataset, split_name: str) -> None:
    n = len(dataset)
    nframes = []
    has_fps = 0
    for rec in dataset:
        nf = rec.get("n_frames")
        if isinstance(nf, (int, float)):
            nframes.append(int(nf))
        if rec.get("fps") is not None:
            has_fps += 1
    mean_frames = float(np.mean(nframes)) if nframes else 0.0
    print(f"[{split_name}] samples={n}, mean_frames={mean_frames:.1f}, fps_known={has_fps}/{n}")


def prepare_dataset(
    data_root: Path,
    output_dir: Path,
) -> None:
    if Dataset is None:
        raise RuntimeError("Hugging Face 'datasets' is not installed. Please install via 'pip install datasets'.")

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in ["train", "val"]:
        print(f"\nProcessing split: {split}")
        quintuples = _gather_split_candidates(data_root, split)
        if not quintuples:
            print(f"[WARN] No matched samples found for split '{split}'. Skipping.")
            continue

        records: List[Dict[str, any]] = []
        for s_vfile, s_afile, l_vfile, l_afile, l_nfile in tqdm(quintuples, desc=f"Loading {split}"):
            rec = load_sample(s_vfile, s_afile, l_vfile, l_afile, l_nfile)
            if rec is not None:
                records.append(rec)
        print(f"Loaded {len(records)} samples out of {len(quintuples)} candidates for split '{split}'.")
        if not records:
            continue

        ds = Dataset.from_list(records)

        out_dir = output_dir / split
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ds.save_to_disk(str(out_dir))
        print_dataset_stats(ds, split)
        summaries[split] = len(ds)

    print("\nDone. Saved splits under:")
    for k, v in summaries.items():
        print(f"  {k}: {output_dir / k} ({v} samples)")


def main():
    parser = argparse.ArgumentParser(description="Prepare Speaker→Listener AU Prediction Dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="scratch/LLM_reaction_Robot/Reaction_DataSet",
        help="Path to raw dataset root (contains train/ and val/ with speaker/listener subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scratch/LLM_reaction_Robot/Reaction_DataSet/processed",
        help="Path to save processed datasets (train/val as HF datasets)",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    print("Exploring data structure...")
    info = explore_data_structure(data_root)
    print("\nExploration summary:")
    try:
        print(json.dumps(info, indent=2))
    except Exception:
        print(str(info))

    print("\n" + "="*80)
    print("PREPARING SPEAKER→LISTENER AU PREDICTION DATASETS")
    print("Task: Predict listener AU from speaker's video+audio")
    print("="*80)
    prepare_dataset(data_root, output_dir)


if __name__ == "__main__":
    main()
