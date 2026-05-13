"""Launch all SVPG paper experiments as independent background processes.
Automatically resumes from saved checkpoints if available."""
import subprocess
import sys
import os
import glob
import time

PYTHON = r"D:\python\python.exe"
WORK_DIR = r"d:\claudego\强化学习2.0"

EXPERIMENTS = [
    # Key-Lock (N=5, 200k steps)
    ("svpg",  "key_lock",  5, 42,  200000),
    ("svpg",  "key_lock",  5, 123, 200000),
    ("svpg",  "key_lock",  5, 456, 200000),
    ("coma",  "key_lock",  5, 42,  200000),
    ("coma",  "key_lock",  5, 123, 200000),
    ("coma",  "key_lock",  5, 456, 200000),
    ("cmapg", "key_lock",  5, 42,  200000),
    ("cmapg", "key_lock",  5, 123, 200000),
    ("cmapg", "key_lock",  5, 456, 200000),
    # Transport (N=3, 200k steps)
    ("svpg",  "transport", 3, 42,  200000),
    ("svpg",  "transport", 3, 123, 200000),
    ("svpg",  "transport", 3, 456, 200000),
    ("mappo", "transport", 3, 42,  200000),
    ("mappo", "transport", 3, 123, 200000),
    ("mappo", "transport", 3, 456, 200000),
    # MPE (N=3, 200k steps)
    ("svpg",  "mpe",       3, 42,  200000),
    ("svpg",  "mpe",       3, 123, 200000),
    ("svpg",  "mpe",       3, 456, 200000),
    ("mappo", "mpe",       3, 42,  200000),
    ("mappo", "mpe",       3, 123, 200000),
    ("mappo", "mpe",       3, 456, 200000),
]


def find_latest_checkpoint(log_dir):
    """Find the latest model checkpoint in log_dir."""
    pt_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not pt_files:
        return None
    pt_files.sort(key=lambda x: int(x.split("model_")[-1].split(".")[0]), reverse=True)
    return pt_files[0]


def launch(algo, env, n_agents, seed, total_steps):
    exp_name = f"{algo}_{env}_n{n_agents}_s{seed}"
    log_dir = os.path.join(WORK_DIR, "logs", exp_name)
    log_file = os.path.join(log_dir, "run.log")
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        PYTHON, "-u", os.path.join(WORK_DIR, "experiments", "train.py"),
        "--algo", algo,
        "--env", env,
        "--n_agents", str(n_agents),
        "--seed", str(seed),
        "--total_steps", str(total_steps),
        "--device", "cpu",
    ]

    # Check for checkpoint to resume from
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
    for i, (algo, env, n_agents, seed, steps) in enumerate(EXPERIMENTS):
        exp_name = f"{algo}_{env}_n{n_agents}_s{seed}"
        log_dir = os.path.join(WORK_DIR, "logs", exp_name)

        # Skip if already completed (has final checkpoint)
        final_ckpt = os.path.join(log_dir, f"model_{steps}.pt")
        if os.path.exists(final_ckpt):
            print(f"[{i+1:2d}/{len(EXPERIMENTS)}] SKIP {exp_name} — already complete")
            skipped += 1
            continue

        name, pid = launch(algo, env, n_agents, seed, steps)
        processes.append((name, pid))
        print(f"[{i+1:2d}/{len(EXPERIMENTS)}] {name}  pid={pid}")
        time.sleep(1)

    print(f"\nLaunched {len(processes)} experiments (skipped {skipped} already complete).")


if __name__ == "__main__":
    main()
