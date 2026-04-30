#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly LOCK_DIR="${REPO_ROOT}/.github/conda-lock"
readonly TEMP_ENV_FILE="${REPO_ROOT}/env_tmp.yml"
readonly PINNED_CONDA_LOCK_VERSION="2.5.7"
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

probe_conda_lock() {
  local env_file="$1"
  local python_version="$2"
  local probe_dir="$3"
  local probe_template="${probe_dir}/probe-${python_version}-{platform}.lock"

  if conda-lock lock \
    --file "${env_file}" \
    --kind explicit \
    --platform linux-64 \
    --filename-template "${probe_template}" >/dev/null 2>&1; then
    rm -f "${probe_dir}/probe-${python_version}-linux-64.lock"
    return 0
  fi

  return 1
}

prepare_lock_input() {
  local python_version="$1"
  local probe_dir="$2"

  if probe_conda_lock "${TEMP_ENV_FILE}" "${python_version}" "${probe_dir}"; then
    echo "${TEMP_ENV_FILE}"
    return 0
  fi

  local named_env_file="${probe_dir}/env_tmp_named_${python_version}.yml"
  {
    echo "name: firecrown_lock_py${python_version//./}"
    cat "${TEMP_ENV_FILE}"
  } > "${named_env_file}"

  if probe_conda_lock "${named_env_file}" "${python_version}" "${probe_dir}"; then
    echo "${named_env_file}"
    return 0
  fi

  echo "conda-lock could not process env_tmp.yml for python ${python_version}" >&2
  return 1
}

cleanup() {
  rm -f "${TEMP_ENV_FILE}"
  if [[ -n "${PROBE_DIR:-}" && -d "${PROBE_DIR}" ]]; then
    rm -rf "${PROBE_DIR}"
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

  PROBE_DIR="$(mktemp -d)"
  trap cleanup EXIT

  local condarc_file
  condarc_file="${PROBE_DIR}/condarc"
  write_isolated_condarc "${condarc_file}"
  export CONDARC="${condarc_file}"

  local python_version
  for python_version in "${PYTHON_VERSIONS[@]}"; do
    echo "Generating lockfiles for python ${python_version}"
    python3 .github/update_ci.py "${python_version}"

    if [[ ! -f "${TEMP_ENV_FILE}" ]]; then
      echo "Missing ${TEMP_ENV_FILE} after running update_ci.py" >&2
      exit 1
    fi

    local lock_input
    lock_input="$(prepare_lock_input "${python_version}" "${PROBE_DIR}")"

    conda-lock lock \
      --file "${lock_input}" \
      --kind explicit \
      --platform linux-64 \
      --platform osx-arm64 \
      --filename-template "${LOCK_DIR}/conda-{platform}-py${python_version}.lock"
  done

  echo "Generated lockfiles in ${LOCK_DIR}"
}

main "$@"
