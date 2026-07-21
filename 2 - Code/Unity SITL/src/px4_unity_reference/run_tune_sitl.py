#!/usr/bin/env python3


import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project_config import load_px4_tuning_config


def wait_for_port(host, port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    config = load_px4_tuning_config()
    print(f"[config] {config.source}")
    build = Path(config.build)
    calib = Path(config.calib)
    param_file = Path(config.param_file)
    journal = Path(config.journal)
    study_name = config.study

    if not build.is_file():
        raise SystemExit(f"BUILD not found: {build}")
    if not calib.is_file():
        raise SystemExit(f"calibration {calib} missing — run "
                         "quadsim-px4-thrust-sweep first (stage 2 gates tuning)")
    if not param_file.is_file():
        raise SystemExit(f"PX4 parameter file not found: {param_file}")
    if config.launch_px4 and config.workers != 1:
        raise SystemExit("automatic PX4 launch currently supports --workers 1 only")

    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.exists() and not config.resume:
        journal.unlink()

    sims, workers, logs = [], [], []
    px4_proc = None
    try:
        # 1. PX4 instance (optional — normally started in its own terminal)
        if config.launch_px4:
            px4_directory = Path(config.px4_directory)
            if not px4_directory.is_dir():
                raise SystemExit(f"PX4 directory not found: {px4_directory}")
            cmd = ["make", "px4_sitl", "none_iris"]
            px4_log = open(REPOSITORY_ROOT / "px4_instances.log", "w")
            logs.append(px4_log)
            px4_proc = subprocess.Popen(cmd, stdout=px4_log,
                                        stderr=subprocess.STDOUT,
                                        cwd=px4_directory)
            print(f"[px4] launcher pid={px4_proc.pid} -> px4_instances.log")

        # 2. One headless Unity per worker
        for i in range(config.workers):
            worker = config.worker(i)
            cmd_port = worker.quadsim_port
            log_path = REPOSITORY_ROOT / f"sim_{i}.log"
            log = open(log_path, "w")
            logs.append(log)
            proc = subprocess.Popen(
                [str(build), "-batchmode", "-nographics",
                 "-rpcPort", str(cmd_port),
                 "-telemetryPort", str(cmd_port + 1)],
                stdout=log, stderr=subprocess.STDOUT)
            sims.append(proc)
            print(f"[sim {i}] pid={proc.pid} rpc={cmd_port} -> {log_path}")
        for i in range(config.workers):
            worker = config.worker(i)
            if not wait_for_port(worker.quadsim_host, worker.quadsim_port):
                raise SystemExit(f"sim {i} never bound — check sim_{i}.log")
        print("[sims bound]")

        # 3. Workers — each runs trials against its own plant, all appending
        #    to the shared journal. Ctrl+C here interrupts all of them.
        for i in range(config.workers):
            cmd = [sys.executable, str(SCRIPT_DIRECTORY / "tune_gains_sitl.py")]
            worker_env = os.environ.copy()
            worker_env["PX4_TUNING_WORKER_INDEX"] = str(i)
            worker_log_path = REPOSITORY_ROOT / f"worker_{i}.log"
            wlog = open(worker_log_path, "w")
            logs.append(wlog)
            workers.append(subprocess.Popen(cmd, stdout=wlog,
                                            stderr=subprocess.STDOUT,
                                            cwd=REPOSITORY_ROOT,
                                            env=worker_env))
            print(f"[worker {i}] pid={workers[-1].pid} -> {worker_log_path}")

        for i, w in enumerate(workers):
            rc = w.wait()
            print(f"[worker {i}] exited rc={rc}")

        # 4. Report from the shared journal
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import (JournalFileBackend,
                                             JournalFileOpenLock)
        storage = JournalStorage(JournalFileBackend(
            str(journal), lock_obj=JournalFileOpenLock(str(journal))))
        study = optuna.load_study(study_name=study_name, storage=storage)
        best = study.best_trial
        print("\n" + "=" * 60)
        print(f"{len(study.trials)} trials total; best {best.number}: "
              f"cost={best.value:.0f}")
        for k, v in best.params.items():
            print(f"    {k}: {v:.4f}")
    finally:
        print("\n[teardown]")
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for s in sims:
            s.terminate()
            try:
                s.wait(timeout=5)
            except subprocess.TimeoutExpired:
                s.kill()
        if px4_proc is not None and px4_proc.poll() is None:
            px4_proc.terminate()
        for lg in logs:
            lg.close()
        print("[done]")


if __name__ == "__main__":
    main()
