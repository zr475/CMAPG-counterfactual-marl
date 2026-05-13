"""Recover metrics.json from run.log for all experiments."""
import re, json, os, sys

LOG_DIR = r"d:\claudego\强化学习2.0\logs"


def recover_from_log(log_dir):
    """Parse run.log and rebuild metrics.json."""
    log_file = os.path.join(log_dir, "run.log")
    if not os.path.exists(log_file):
        return False

    metrics = {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Step  X | key: value  (training metrics)
            m = re.search(r"Step\s+(\d+)\s+\|\s+(\w+):\s+([-\d.]+)", line)
            if m:
                key = m.group(2)
                val = float(m.group(3))
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)

            # Step  X | Return: Y ± Z  (eval)
            m2 = re.search(r"Step\s+(\d+)\s+\|\s+Return:\s+([-\d.]+)", line)
            if m2:
                step = int(m2.group(1))
                ret = float(m2.group(2))
                if "eval_step" not in metrics:
                    metrics["eval_step"] = []
                    metrics["eval_return_mean"] = []
                    metrics["eval_return_std"] = []
                # Avoid duplicate entries for same step
                if not metrics["eval_step"] or metrics["eval_step"][-1] != step:
                    metrics["eval_step"].append(step)
                    metrics["eval_return_mean"].append(ret)
                    metrics["eval_return_std"].append(0.0)

            # eval_ca_accuracy: X
            m3 = re.search(r"eval_ca_accuracy:\s+([-\d.]+)", line)
            if m3:
                ca = float(m3.group(1))
                key = "eval_ca_accuracy"
                if key not in metrics:
                    metrics[key] = []
                # Only add if there's a corresponding eval step
                if metrics.get("eval_step") and len(metrics[key]) < len(metrics["eval_step"]):
                    metrics[key].append(ca)

    if not metrics:
        return False

    # Save
    mf = os.path.join(log_dir, "metrics.json")
    # Backup old if exists and different
    if os.path.exists(mf):
        bk = mf + ".bak"
        with open(mf) as f:
            old = json.load(f)
        old_evals = len(old.get("eval_return_mean", []))
        new_evals = len(metrics.get("eval_return_mean", []))
        if new_evals > old_evals:
            with open(bk, "w") as f:
                json.dump(old, f, indent=2)
            print(f"  Backed up old metrics ({old_evals} evals) -> {bk}")

    with open(mf, "w") as f:
        json.dump(metrics, f, indent=2)
    n_evals = len(metrics.get("eval_return_mean", []))
    print(f"  Recovered: {len(metrics)} metric keys, {n_evals} evals -> {mf}")
    return True


def main():
    recovered = 0
    for d in sorted(os.listdir(LOG_DIR)):
        p = os.path.join(LOG_DIR, d)
        if not os.path.isdir(p):
            continue
        mf = os.path.join(p, "metrics.json")
        if os.path.exists(mf):
            with open(mf) as f:
                m = json.load(f)
            n = len(m.get("eval_return_mean", []))
        else:
            n = 0

        if n >= 15:  # Seems complete
            print(f"{d}: {n} evals — OK, skipping")
            continue

        print(f"{d}: {n} evals — recovering...")
        if recover_from_log(p):
            recovered += 1

    print(f"\nRecovered {recovered} experiments.")


if __name__ == "__main__":
    main()
