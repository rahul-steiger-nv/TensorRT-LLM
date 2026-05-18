#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -J cosmos3-h100-1024gb-smoke
#SBATCH -p h100-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=256
#SBATCH --mem=2000G
#SBATCH -t 02:00:00
#SBATCH -o cosmos3-h100-2304gb-smoke-%j.out
#SBATCH -e cosmos3-h100-2304gb-smoke-%j.err

##############################################################################
# Cosmos3 H100-NVL 8-GPU / 2TB+ smoke test for the x86_64 TensorRT-LLM workspace.
#
# Verified target partitions:
#   h100-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb
#   h200-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb
#
# Submit:
#   export COSMOS3_MODEL_PATH=/path/to/cosmos3-diffusers-checkpoint
#   sbatch examples/visual_gen/sbatch_cosmos3_h100_1024gb_smoke.sh
#
# If your Slurm setup does not export the caller environment by default:
#   sbatch --export=ALL,COSMOS3_MODEL_PATH=/path/to/cosmos3-diffusers-checkpoint \
#       examples/visual_gen/sbatch_cosmos3_h100_1024gb_smoke.sh
#
# If your cluster requires an account, submit with:
#   sbatch -A <account> examples/visual_gen/sbatch_cosmos3_h100_1024gb_smoke.sh
#
# To use another 8-GPU 2TB+ partition, override at submit time:
#   sbatch -p h200-nvl@qs1/genoa2d24g2l/8gpu-256cpu-2304gb ...
# To target a currently idle node explicitly, add:
#   sbatch -w 4u8g-gen-0289 ...
#
# This script runs directly in the x86_64 uv virtualenv by default, not a
# container. Override PYTHON_BIN or PROJECT_DIR if you need a different checkout.
# It launches distributed workers with torchrun rather than direct multi-rank
# srun, so it does not require Open MPI to be built with Slurm PMI support.
##############################################################################

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scratch.rsteiger_other/TensorRT-LLM-x86_64}"
COSMOS3_MODEL_PATH="${COSMOS3_MODEL_PATH:?Set COSMOS3_MODEL_PATH to your Cosmos3 checkpoint path}"

if [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: this script is for x86_64 nodes only; got $(uname -m)" >&2
    exit 1
fi

DEFAULT_PYTHON_BIN="/home/scratch.rsteiger_other/.local/uv/venvs/tensorrt-llm-x86_64/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

PROMPT="${PROMPT:-A cat walking on a beach}"
OUTPUT_PATH="${OUTPUT_PATH:-cosmos3_h100_1024gb_smoke.avi}"

# Keep the smoke test focused on load/offload viability.
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-61}"
NUM_STEPS="${NUM_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
FPS="${FPS:-24.0}"
SEED="${SEED:-42}"

ATTENTION_BACKEND="${ATTENTION_BACKEND:-VANILLA}"
CFG_SIZE="${CFG_SIZE:-1}"
ULYSSES_SIZE="${ULYSSES_SIZE:-8}"
PROFILE_MEMORY="${PROFILE_MEMORY:-1}"

ENABLE_OFFLOADING="${ENABLE_OFFLOADING:-1}"
OFFLOAD_SHARE_MEMORY="${OFFLOAD_SHARE_MEMORY:-1}"
OFFLOAD_SHARED_MEMORY_PATH="${OFFLOAD_SHARED_MEMORY_PATH:-}"
OFFLOAD_SHARED_MEMORY_SCOPE="${OFFLOAD_SHARED_MEMORY_SCOPE:-global}"
OFFLOAD_GUARDRAILS="${OFFLOAD_GUARDRAILS:-0}"
DISABLE_GUARDRAILS="${DISABLE_GUARDRAILS:-1}"
DISABLE_PARALLEL_VAE="${DISABLE_PARALLEL_VAE:-1}"
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-1}"
ENABLE_NUMA_BIND="${ENABLE_NUMA_BIND:-1}"
NUMA_MEM_POLICY="${NUMA_MEM_POLICY:-membind}" # one of: membind, preferred, none
NUMA_VERBOSE="${NUMA_VERBOSE:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

prepend_path() {
    local var_name="$1"
    local path_value="$2"
    local current_value="${!var_name:-}"

    [ -n "${path_value}" ] || return 0
    [ -d "${path_value}" ] || return 0

    case ":${current_value}:" in
        *":${path_value}:"*) ;;
        *)
            if [ -n "${current_value}" ]; then
                export "${var_name}=${path_value}:${current_value}"
            else
                export "${var_name}=${path_value}"
            fi
            ;;
    esac
}

if [ ! -d "${PROJECT_DIR}/tensorrt_llm" ]; then
    echo "ERROR: PROJECT_DIR does not look like a TensorRT-LLM checkout: ${PROJECT_DIR}" >&2
    exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ "${OFFLOAD_SHARE_MEMORY}" = "1" ] && [ "${ENABLE_OFFLOADING}" != "1" ]; then
    echo "ERROR: OFFLOAD_SHARE_MEMORY=1 requires ENABLE_OFFLOADING=1" >&2
    exit 1
fi

if [ "${OFFLOAD_SHARE_MEMORY}" = "1" ] && [ "${OFFLOAD_GUARDRAILS}" = "1" ]; then
    echo "ERROR: shared offload memory does not support OFFLOAD_GUARDRAILS=1 yet" >&2
    exit 1
fi

PYTHON_DIR="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.executable).resolve().parent)
PY
)"
SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
TORCH_LIB="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"

if [ -z "${CUDA_HOME:-}" ]; then
    for candidate in \
        "${SITE_PACKAGES}/nvidia/cu13" \
        "${SITE_PACKAGES}/nvidia/cu12" \
        "/usr/local/cuda"; do
        if [ -d "${candidate}" ] && { [ -x "${candidate}/bin/nvcc" ] || [ -f "${candidate}/include/cuda.h" ]; }; then
            export CUDA_HOME="${candidate}"
            break
        fi
    done
fi

if [ -z "${CUDA_HOME:-}" ]; then
    echo "ERROR: CUDA_HOME is unset and no CUDA toolkit was found from ${PYTHON_BIN}" >&2
    exit 1
fi

export PYTHON_BIN
export CUDA_PATH="${CUDA_HOME}"
export TRTLLM_DISABLE_COSMOS3_GUARDRAILS="${TRTLLM_DISABLE_COSMOS3_GUARDRAILS:-${DISABLE_GUARDRAILS}}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
prepend_path PATH "${PYTHON_DIR}"
prepend_path PATH "${CUDA_HOME}/bin"
prepend_path LD_LIBRARY_PATH "${CUDA_HOME}/lib"
prepend_path LD_LIBRARY_PATH "${TORCH_LIB}"

if [ "${RUN_PREFLIGHT}" = "1" ]; then
    "${PYTHON_BIN}" - <<'PY'
from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
print("preflight_ok=visual_gen_imports")
PY
fi

NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${GPUS_PER_NODE:-4}}"
NUM_GPUS=$(( NNODES * GPUS_PER_NODE ))
EXPECTED_GPUS=$(( CFG_SIZE * ULYSSES_SIZE ))
if [ "${NUM_GPUS}" -ne "${EXPECTED_GPUS}" ]; then
    echo "ERROR: allocated GPUs (${NUM_GPUS}) != CFG_SIZE(${CFG_SIZE}) * ULYSSES_SIZE(${ULYSSES_SIZE}) = ${EXPECTED_GPUS}" >&2
    exit 1
fi

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
else
    MASTER_ADDR="$(hostname)"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

CPUS_PER_RANK="${CPUS_PER_RANK:-$(( ${SLURM_CPUS_PER_TASK:-64} / GPUS_PER_NODE ))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPUS_PER_RANK}}"
export NUMA_MEM_POLICY
export NUMA_VERBOSE

echo "=== Slurm allocation ==="
echo "job_id=${SLURM_JOB_ID:-local-dry-run}"
echo "nodelist=${SLURM_JOB_NODELIST:-$(hostname)}"
echo "partition=${SLURM_JOB_PARTITION:-local-dry-run}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "project_dir=${PROJECT_DIR}"
echo "model=${COSMOS3_MODEL_PATH}"
echo "python=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "torch_lib=${TORCH_LIB}"
echo "TRTLLM_DISABLE_COSMOS3_GUARDRAILS=${TRTLLM_DISABLE_COSMOS3_GUARDRAILS}"
echo "ENABLE_OFFLOADING=${ENABLE_OFFLOADING}"
echo "OFFLOAD_SHARE_MEMORY=${OFFLOAD_SHARE_MEMORY}"
echo "OFFLOAD_SHARED_MEMORY_PATH=${OFFLOAD_SHARED_MEMORY_PATH}"
echo "OFFLOAD_SHARED_MEMORY_SCOPE=${OFFLOAD_SHARED_MEMORY_SCOPE}"
echo "OFFLOAD_GUARDRAILS=${OFFLOAD_GUARDRAILS}"
echo "DISABLE_PARALLEL_VAE=${DISABLE_PARALLEL_VAE}"
echo "ENABLE_NUMA_BIND=${ENABLE_NUMA_BIND}"
echo "NUMA_MEM_POLICY=${NUMA_MEM_POLICY}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "nnodes=${NNODES}"
echo "gpus_per_node=${GPUS_PER_NODE}"

echo "=== Node memory ==="
free -h || true
numactl --hardware || true
nvidia-smi || true

RUN_CMD=(
    examples/visual_gen/visual_gen_cosmos3.py
    --model_path "${COSMOS3_MODEL_PATH}"
    --prompt "${PROMPT}"
    --output_path "${OUTPUT_PATH}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --steps "${NUM_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --fps "${FPS}"
    --seed "${SEED}"
    --attention_backend "${ATTENTION_BACKEND}"
    --cfg_size "${CFG_SIZE}"
    --ulysses_size "${ULYSSES_SIZE}"
    --profile_memory "${PROFILE_MEMORY}"
)

if [ "${ENABLE_OFFLOADING}" = "1" ]; then
    RUN_CMD+=(--enable_offloading)
fi
if [ "${OFFLOAD_SHARE_MEMORY}" = "1" ]; then
    RUN_CMD+=(--offload_share_memory)
    RUN_CMD+=(--offload_shared_memory_scope "${OFFLOAD_SHARED_MEMORY_SCOPE}")
fi
if [ -n "${OFFLOAD_SHARED_MEMORY_PATH}" ]; then
    RUN_CMD+=(--offload_shared_memory_path "${OFFLOAD_SHARED_MEMORY_PATH}")
fi
if [ "${OFFLOAD_GUARDRAILS}" = "1" ]; then
    RUN_CMD+=(--offload_guardrails)
fi
if [ "${DISABLE_GUARDRAILS}" = "1" ]; then
    RUN_CMD+=(--disable_guardrails)
fi
if [ "${DISABLE_PARALLEL_VAE}" = "1" ]; then
    RUN_CMD+=(--disable_parallel_vae)
fi
if [ "${DISABLE_TORCH_COMPILE}" = "1" ]; then
    RUN_CMD+=(--disable_torch_compile)
fi

TORCH_ENTRY=("${RUN_CMD[@]}")
if [ "${ENABLE_NUMA_BIND}" = "1" ]; then
    NUMA_WRAPPER="${NUMA_WRAPPER:-/tmp/cosmos3_numa_wrapper_${SLURM_JOB_ID:-$$}.py}"
    cat >"${NUMA_WRAPPER}" <<'PY'
#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _visible_gpu_for_rank(local_rank: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible != "NoDevFiles":
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        if local_rank < len(devices):
            return devices[local_rank]
    return str(local_rank)


def _gpu_bus_id(gpu: str) -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else None


def _sysfs_pci_id(bus_id: str) -> str:
    # nvidia-smi reports 00000000:01:00.0; sysfs uses 0000:01:00.0.
    domain, bus, device = bus_id.split(":", 2)
    return f"{domain[-4:]}:{bus}:{device}".lower()


def _parse_node_list(node_list: str) -> set[int]:
    nodes = set()
    for item in node_list.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            nodes.update(range(int(start), int(end) + 1))
        else:
            nodes.add(int(item))
    return nodes


def _single_visible_numa_node() -> int | None:
    for node_list_path in (
        Path("/sys/devices/system/node/online"),
        Path("/sys/devices/system/node/possible"),
    ):
        try:
            nodes = _parse_node_list(node_list_path.read_text().strip())
        except (OSError, ValueError):
            continue
        if len(nodes) == 1:
            return next(iter(nodes))
    return None


def _numa_node_for_gpu(gpu: str) -> int | None:
    bus_id = _gpu_bus_id(gpu)
    if bus_id is None:
        return None
    numa_path = Path("/sys/bus/pci/devices") / _sysfs_pci_id(bus_id) / "numa_node"
    try:
        numa_node = int(numa_path.read_text().strip())
    except (OSError, ValueError):
        return None
    if numa_node >= 0:
        return numa_node
    return _single_visible_numa_node()


def _exec_without_numa(reason: str) -> None:
    if os.environ.get("NUMA_VERBOSE", "1") == "1":
        print(f"[numa-bind] rank={os.environ.get('RANK', '?')} disabled: {reason}", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


def main() -> None:
    if not sys.argv[1:]:
        raise SystemExit("numa wrapper requires a Python script argument")

    if os.environ.get("COSMOS3_NUMA_WRAPPED") == "1":
        _exec_without_numa("already wrapped")

    if shutil.which("numactl") is None:
        _exec_without_numa("numactl not found")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu = _visible_gpu_for_rank(local_rank)
    numa_node = _numa_node_for_gpu(gpu)
    if numa_node is None:
        _exec_without_numa(f"could not resolve NUMA node for GPU {gpu}")

    mem_policy = os.environ.get("NUMA_MEM_POLICY", "membind")
    cmd = ["numactl", f"--cpunodebind={numa_node}"]
    if mem_policy == "membind":
        cmd.append(f"--membind={numa_node}")
    elif mem_policy == "preferred":
        cmd.append(f"--preferred={numa_node}")
    elif mem_policy == "none":
        pass
    else:
        _exec_without_numa(f"invalid NUMA_MEM_POLICY={mem_policy!r}")

    cmd.extend([sys.executable, *sys.argv[1:]])
    env = os.environ.copy()
    env["COSMOS3_NUMA_WRAPPED"] = "1"

    if env.get("NUMA_VERBOSE", "1") == "1":
        print(
            f"[numa-bind] rank={env.get('RANK', '?')} local_rank={local_rank} "
            f"gpu={gpu} numa_node={numa_node} mem_policy={mem_policy}",
            flush=True,
        )

    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
PY
    chmod +x "${NUMA_WRAPPER}"
    TORCH_ENTRY=("${NUMA_WRAPPER}" "${RUN_CMD[@]}")
fi

TORCHRUN_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nnodes "${NNODES}"
    --nproc_per_node "${GPUS_PER_NODE}"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --max_restarts 0
    "${TORCH_ENTRY[@]}"
)

printf '=== Run command ===\n'
printf '%q ' "${TORCHRUN_CMD[@]}"
printf '\n'

cd "${PROJECT_DIR}"
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1, skipping torchrun launch"
    exit 0
fi

"${TORCHRUN_CMD[@]}"
