#!/bin/bash
# Run all SVPG paper experiments
# Usage: bash run_experiments.sh

PYTHON="D:/python/python.exe"
cd "/d/claudego/强化学习2.0"

# Helper: run experiment and create symlink-like marker
run_exp() {
    local algo=$1 env=$2 n_agents=$3 seed=$4 total_steps=$5
    local exp_name="${algo}_${env}_n${n_agents}_s${seed}"
    local log_dir="logs/${exp_name}"
    local log_file="${log_dir}/run.log"

    mkdir -p "$log_dir"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${exp_name}..."

    $PYTHON experiments/train.py \
        --algo "$algo" \
        --env "$env" \
        --n_agents "$n_agents" \
        --seed "$seed" \
        --total_steps "$total_steps" \
        --device cpu \
        >> "$log_file" 2>&1

    local rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${exp_name} finished (exit=$rc)"
    return $rc
}

# ========== Key-Lock (N=5, 200k steps) ==========
# SVPG
run_exp svpg key_lock 5 42   200000 &
run_exp svpg key_lock 5 123 200000 &
run_exp svpg key_lock 5 456 200000 &

# QMIX
run_exp qmix key_lock 5 42   200000 &
run_exp qmix key_lock 5 123 200000 &
run_exp qmix key_lock 5 456 200000 &

# CMAPG
run_exp cmapg key_lock 5 42   200000 &
run_exp cmapg key_lock 5 123 200000 &
run_exp cmapg key_lock 5 456 200000 &

# ========== Transport (N=3, 200k steps) ==========
run_exp svpg  transport 3 42   200000 &
run_exp svpg  transport 3 123 200000 &
run_exp svpg  transport 3 456 200000 &

run_exp mappo transport 3 42   200000 &
run_exp mappo transport 3 123 200000 &
run_exp mappo transport 3 456 200000 &

# ========== MPE (N=3, 200k steps) ==========
run_exp svpg  mpe 3 42   200000 &
run_exp svpg  mpe 3 123 200000 &
run_exp svpg  mpe 3 456 200000 &

run_exp mappo mpe 3 42   200000 &
run_exp mappo mpe 3 123 200000 &
run_exp mappo mpe 3 456 200000 &

# Wait for all background jobs
echo "All experiments launched. Waiting for completion..."
wait
echo "All experiments done!"
