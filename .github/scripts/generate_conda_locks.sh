#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly LOCK_DIR="${REPO_ROOT}/.github/conda-lock"
readonly TEMP_ENV_FILE="${REPO_ROOT}/env_tmp.yml"
readonly PINNED_CONDA_LOCK_VERSION="4.0.0"
readonly PYTHON_VERSIONS=("3.11" "3.12" "3.13" "3.14")

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

conda_lock_version() {
  conda-lock --version | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1
}

CONDARC_DIR=""

cleanup() {
  rm -f "${TEMP_ENV_FILE}"
  if [[ -n "${CONDARC_DIR}" && -d "${CONDARC_DIR}" ]]; then
    rm -rf "${CONDARC_DIR}"
  fi
}

write_isolated_condarc() {
  local condarc_file="$1"
  cat > "${condarc_file}" <<'EOF'
channels:
  - conda-forge
  - nodefaults
channel_priority: strict
EOF
}

main() {
  cd "${REPO_ROOT}"

  require_command conda-lock
  require_command python3

  local detected_version
  detected_version="$(conda_lock_version)"
  if [[ "${detected_version}" != "${PINNED_CONDA_LOCK_VERSION}" ]]; then
    echo "Expected conda-lock ${PINNED_CONDA_LOCK_VERSION}, found ${detected_version}." >&2
    echo "Install the pinned toolchain version before generating lockfiles." >&2
    exit 1
  fi

  mkdir -p "${LOCK_DIR}"

  CONDARC_DIR="$(mktemp -d)"
  trap cleanup EXIT

  local condarc_file="${CONDARC_DIR}/condarc"
  write_isolated_condarc "${condarc_file}"
  export CONDARC="${condarc_file}"

  local python_version
  for python_version in "${PYTHON_VERSIONS[@]}"; do
    echo "Generating unified lockfile for python ${python_version}"
    python3 .github/update_ci.py "${python_version}"

    if [[ ! -f "${TEMP_ENV_FILE}" ]]; then
      echo "Missing ${TEMP_ENV_FILE} after running update_ci.py" >&2
      exit 1
    fi

    conda-lock lock \
      --file "${TEMP_ENV_FILE}" \
      --platform linux-64 \
      --platform osx-arm64 \
      --lockfile "${LOCK_DIR}/py${python_version}.conda-lock.yml"
  done

  echo "Generated lockfiles in ${LOCK_DIR}"
}

main "$@"
