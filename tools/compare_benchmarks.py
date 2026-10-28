"""Summarize interval samples and verify fixed-input recognition independently of timing."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def same(left, right, path="frame"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path + " keys"
        for key in left:
            same(left[key], right[key], path + "." + key)
    elif isinstance(left, list):
        assert len(left) == len(right), path + " length"
        for index, (a, b) in enumerate(zip(left, right)):
            same(a, b, path + "[" + str(index) + "]")
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-6), path
    else:
        assert left == right, path


def compare_videos(before, after):
    report = []
    for key in ("rub", "fist", "stretch", "cup", "abduction", "press"):
        old, new = rows(before / (key + ".jsonl")), rows(after / (key + ".jsonl"))
        assert len(old) == len(new) and old, key + " frame count"
        for a, b in zip(old, new):
            for row in (a, b):
                row.pop("processing_ms", None)
                row.pop("drawing_ms", None)
            same(a, b, key + ".frame" + str(a["index"]))
        a, b = (json.loads((directory / (key + "-summary.json")).read_text()) for directory in (before, after))
        for row in (a, b):
            row.pop("elapsed", None)
        same(a, b, key + ".summary")
        report.append({"action": key, "frames": len(old), "equal": True, "count": a["count"], "passed": a["passed"]})
    return report


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not values:
        return None
    position = (len(values) - 1) * .95
    lo, hi = math.floor(position), math.ceil(position)
    return {"samples": len(values), "median": statistics.median(values),
            "p95": values[lo] + (values[hi] - values[lo]) * (position - lo), "min": values[0], "max": values[-1]}


def summarize(root):
    output = []
    for phase in ("before", "after"):
        renderer = root / ("renderer-" + phase + ".log")
        renderer_lines = renderer.read_text(errors="replace").splitlines() if renderer.exists() else []
        for file in sorted((root / ("robot-" + phase)).glob("*.jsonl")):
            if "-system" in file.stem:
                continue
            all_rows = rows(file)
            samples = [row for row in all_rows if row.get("warmup") is False]
            if not samples:
                continue
            # The user shortened the protocol; use the same first 30s after warmup in long runs.
            first_at = min(row["wall_time_ms"] for row in samples)
            samples = [row for row in samples if row["wall_time_ms"] < first_at + 30000]
            start = min(row["wall_time_ms"] for row in samples) / 1000
            end = max(row["wall_time_ms"] for row in samples) / 1000
            metrics = {key: distribution([row.get(key) for row in samples]) for key in samples[0]
                       if key not in ("monotonic_ms", "wall_time_ms", "warmup")}
            saved = next((row["run"] for row in all_rows if row.get("type") == "saved_run"), None)
            if saved and saved.get("sessions"):
                identity = saved["sessions"][0]["id"]
                server = root / ("server-" + phase) / (identity + ".jsonl")
                if server.exists():
                    frames = [row for row in rows(server) if start <= row["sampled_at"] <= end]
                    for key in ("processing_ms", "drawing_ms", "recognition_ms", "server_total_ms"):
                        metrics["server_frame_" + key] = distribution([row.get(key) for row in frames])
                    if len(frames) > 1:
                        dropped = frames[-1]["dropped_input"] - frames[0]["dropped_input"]
                        decoded = frames[-1]["decoded_frames_total"] - frames[0]["decoded_frames_total"]
                        metrics["intentional_input_drop_fraction"] = distribution([dropped / decoded]) if decoded else None
            system = file.with_name(file.stem + "-system.jsonl")
            thermal = {}
            if system.exists():
                for row in rows(system):
                    if not start <= row["wall_time"] <= end:
                        continue
                    lines = row["raw"].strip().splitlines()
                    if not lines:
                        continue
                    try:
                        thermal.setdefault("system_load_1m", []).append(float(lines[0].split()[0]))
                        for index in range(1, len(lines) - 1, 2):
                            thermal.setdefault("temperature_c_" + lines[index], []).append(float(lines[index + 1]) / 1000)
                    except ValueError:
                        pass
                metrics.update({key: distribution(value) for key, value in thermal.items()})
            for view in ("local", "remote"):
                rendered = []
                for line in renderer_lines:
                    if "EglRenderer: training_" + view not in line:
                        continue
                    match = re.search(r"^\s*(\d+\.\d+).* Render fps: (\d*\.\d+)", line)
                    if match and start + 4 <= float(match[1]) <= end:
                        rendered.append(float(match[2]))
                metrics[view + "_render_fps"] = distribution(rendered)
            output.append({"label": file.stem, "phase": phase, "action": samples[0]["action"],
                           "mode": samples[0]["mode"], "samples": len(samples), "metrics": metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = {"effective_window_seconds": 30, "robot_runs": summarize(args.root)}
    if (args.root / "videos-after").exists():
        result["algorithm_comparison"] = compare_videos(args.root / "videos-before", args.root / "videos-after")
    (args.root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    with (args.root / "metrics.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(["label", "metric", "samples", "median", "p95", "min", "max"])
        for run in result["robot_runs"]:
            for key, value in run["metrics"].items():
                if value:
                    writer.writerow([run["label"], key] + [value[k] for k in ("samples", "median", "p95", "min", "max")])
    print(len(result["robot_runs"]), "robot runs summarized")
    if "algorithm_comparison" in result:
        print("All six actions match at fixed frames/timestamps (absolute tolerance 1e-6).")


if __name__ == "__main__":
    assert distribution([1, 2, 3]) == {"samples": 3, "median": 2, "p95": 2.9, "min": 1, "max": 3}
    same({"points": [[0.1, 0.2]]}, {"points": [[0.1, 0.20000001]]})
    main()
