"""Launch K-ablation experiments for SVPG paper."""
import subprocess
import sys
import os
import glob
import time

PYTHON = r"D:\python\python.exe"
WORK_DIR = r"d:\claudego\强化学习2.0"

K_VALUES = [1, 5, 10, 50]
SEEDS = [42, 123, 456]
N_AGENTS = 5
TOTAL_STEPS = 200000

EXPERIMENTS = [
    (k, seed) for k in K_VALUES for seed in SEEDS
]


def find_latest_checkpoint(log_dir):
    pt_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not pt_files:
        return None
    pt_files.sort(key=lambda x: int(x.split("model_")[-1].split(".")[0]), reverse=True)
    return pt_files[0]


def launch(k, seed):
    exp_name = f"svpg_key_lock_n{N_AGENTS}_s{seed}_K{k}"
    log_dir = os.path.join(WORK_DIR, "logs", exp_name)
    log_file = os.path.join(log_dir, "run.log")
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        PYTHON, "-u", os.path.join(WORK_DIR, "experiments", "train.py"),
        "--algo", "svpg",
        "--env", "key_lock",
        "--n_agents", str(N_AGENTS),
        "--seed", str(seed),
        "--total_steps", str(TOTAL_STEPS),
        "--K", str(k),
        "--device", "cpu",
        "--exp_name", f"K{k}",
    ]

    ckpt = find_latest_checkpoint(log_dir)
    if ckpt:
        ckpt_name = os.path.basename(ckpt)
        cmd.extend(["--resume", ckpt_name])
        resume_note = f" (resuming from {ckpt_name})"
    else:
        resume_note = ""

    with open(log_file, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Starting {exp_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}{resume_note}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")

    proc = subprocess.Popen(
        cmd,
        cwd=WORK_DIR,
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )
    return exp_name, proc.pid


def main():
    processes = []
    skipped = 0
    for i, (k, seed) in enumerate(EXPERIMENTS):
        exp_name = f"svpg_key_lock_n{N_AGENTS}_s{seed}_K{k}"
        log_dir = os.path.join(WORK_DIR, "logs", exp_name)

        final_ckpt = os.path.join(log_dir, f"model_{TOTAL_STEPS}.pt")
        if os.path.exists(final_ckpt):
            print(f"[{i+1:2d}/{len(EXPERIMENTS)}] SKIP {exp_name} — already complete")
            skipped += 1
            continue

        name, pid = launch(k, seed)
        processes.append((name, pid))
        print(f"[{i+1:2d}/{len(EXPERIMENTS)}] {name}  pid={pid}")
        time.sleep(1)

    print(f"\nLaunched {len(processes)} ablation experiments (skipped {skipped} already complete).")


if __name__ == "__main__":
    main()
