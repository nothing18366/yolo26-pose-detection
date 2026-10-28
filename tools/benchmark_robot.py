"""Drive opt-in Android instrumentation against a disposable server (port 18121)."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--device", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--actions", nargs="+", default=["rub", "fist"])
    parser.add_argument("--modes", nargs="+", help="Retest only these modes; defaults to all modes for the phase")
    parser.add_argument("--sample-seconds", type=int, default=30)
    args = parser.parse_args()
    assert args.server.endswith(":18121"), "Use the isolated benchmark port"
    assert 15 <= args.sample_seconds <= 60
    args.output.mkdir(parents=True, exist_ok=True)
    adb = [args.adb, "-s", args.device]
    modes = ("local", "remote") if args.phase == "before" else ("local", "skeleton", "diagnostic")
    if args.modes:
        assert set(args.modes) <= set(modes), "Unsupported modes for this phase"
        modes = args.modes
    print("Benchmark PID", os.getpid(), flush=True)
    for action in args.actions:
        for repeat in range(1, 4):
            for mode in modes:
                prefix = args.phase + ("-retest" if args.modes else "")
                label = f"{prefix}-{action}-{repeat}-{mode}"
                if (args.output / (label + ".jsonl")).exists():
                    continue
                print(time.strftime("%H:%M:%S"), "START", label, f"15s warmup + {args.sample_seconds}s sample", flush=True)
                command = adb + ["shell", "am", "instrument", "-w", "-r", "-e", "class", "com.robint.TrainingPerformanceTest",
                    "-e", "benchmarkUrl", args.server, "-e", "action", action, "-e", "mode", mode, "-e", "label", label,
                    "-e", "sampleSeconds", str(args.sample_seconds),
                    "com.robint.sdk.demo.test/androidx.test.runner.AndroidJUnitRunner"]
                stop = threading.Event()

                def sample_system():
                    with (args.output / (label + "-system.jsonl")).open("a") as output:
                        while not stop.is_set():
                            snapshot = subprocess.run(adb + ["shell", "cat /proc/loadavg; for z in /sys/class/thermal/thermal_zone*; do cat $z/type $z/temp 2>/dev/null; done"], capture_output=True, text=True, timeout=10)
                            output.write(json.dumps({"wall_time": time.time(), "raw": snapshot.stdout}) + "\n")
                            output.flush()
                            stop.wait(5)

                sampler = threading.Thread(target=sample_system, daemon=True)
                sampler.start()
                try:
                    with (args.output / (label + "-test.log")).open("a") as log:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=140)
                finally:
                    stop.set()
                    sampler.join(timeout=12)
                result = (args.output / (label + "-test.log")).read_text()
                # Only the synthetic benchmark directory is read; no production record is exported.
                sample = subprocess.check_output(adb + ["exec-out", "run-as", "com.robint.sdk.demo", "cat", "files/benchmarks/" + label + ".jsonl"])
                rows = [json.loads(line) for line in sample.splitlines()]
                assert sum(not row.get("warmup", True) for row in rows) >= args.sample_seconds - 2, label
                junit_ok = completed.returncode == 0 and "OK (1 test)" in result and "FAILURES" not in result
                if not junit_ok:
                    saved = next((row["run"] for row in rows if row.get("type") == "saved_run"), {})
                    exit_info = subprocess.check_output(adb + ["shell", "dumpsys", "activity", "exit-info", "com.robint.sdk.demo"], text=True)
                    latest_exit = exit_info.split("ApplicationExitInfo #1:")[0]
                    timestamp = re.search(r"timestamp=([\d-]+ [\d:.]+)", latest_exit)
                    stopped_at = datetime.fromisoformat(timestamp[1]).replace(tzinfo=timezone(timedelta(hours=8))).timestamp() if timestamp else 0
                    last_sample = max(row.get("wall_time_ms", 0) for row in rows) / 1000
                    external_stop = ("shortMsg=Process crashed." in result and "INSTRUMENTATION_STATUS_CODE: 0" in result
                        and "FAILURES" not in result and "reason=10 (USER REQUESTED)" in latest_exit
                        and 0 <= stopped_at - last_sample < 15 and saved.get("status") == "completed")
                    (args.output / (label + "-runner.json")).write_text(json.dumps({"junit_ok": False,
                        "external_stop_after_saved": external_stop, "exit_info": exit_info}, ensure_ascii=False, indent=2))
                    if not external_stop:
                        raise RuntimeError(f"{label} failed; inspect the test log before continuing")
                    print("RUNNER WARNING", label, "external stop after complete samples and saved result; not a JUnit pass", flush=True)
                (args.output / (label + ".jsonl")).write_bytes(sample)
                with (args.output / (label + "-system.txt")).open("wb") as output:
                    subprocess.run(adb + ["shell", "cat /proc/loadavg; for z in /sys/class/thermal/thermal_zone*; do cat $z/type $z/temp 2>/dev/null; done"], stdout=output, stderr=subprocess.STDOUT)
                print(time.strftime("%H:%M:%S"), "DONE", label, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
