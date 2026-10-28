"""Drive opt-in Android instrumentation against a disposable server (port 18121)."""

import argparse
import json
import os
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
    parser.add_argument("--sample-seconds", type=int, default=30)
    args = parser.parse_args()
    assert args.server.endswith(":18121"), "Use the isolated benchmark port"
    assert 15 <= args.sample_seconds <= 60
    args.output.mkdir(parents=True, exist_ok=True)
    adb = [args.adb, "-s", args.device]
    modes = ("local", "remote") if args.phase == "before" else ("local", "skeleton", "diagnostic")
    print("Benchmark PID", os.getpid(), flush=True)
    for action in args.actions:
        for repeat in range(1, 4):
            for mode in modes:
                label = f"{args.phase}-{action}-{repeat}-{mode}"
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
                if completed.returncode or "FAILURES" in result or "OK (1 test)" not in result:
                    raise RuntimeError(f"{label} failed; inspect the test log before continuing")
                # Only the synthetic benchmark directory is read; no production record is exported.
                sample = subprocess.check_output(adb + ["exec-out", "run-as", "com.robint.sdk.demo", "cat", "files/benchmarks/" + label + ".jsonl"])
                rows = [json.loads(line) for line in sample.splitlines()]
                assert sum(not row.get("warmup", True) for row in rows) >= args.sample_seconds - 2, label
                (args.output / (label + ".jsonl")).write_bytes(sample)
                with (args.output / (label + "-system.txt")).open("wb") as output:
                    subprocess.run(adb + ["shell", "cat /proc/loadavg; for z in /sys/class/thermal/thermal_zone*; do cat $z/type $z/temp 2>/dev/null; done"], stdout=output, stderr=subprocess.STDOUT)
                print(time.strftime("%H:%M:%S"), "DONE", label, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
