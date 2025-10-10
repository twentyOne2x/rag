#!/usr/bin/env bash
#
# Downloads and installs the Google Cloud SDK locally, then configures the
# default project to just-skyline-474622-e1. Supports Linux and macOS.
#
# Usage:
#   bash scripts/install_gcloud_sdk.sh
#

set -euo pipefail

PROJECT_ID="just-skyline-474622-e1"
SDK_VERSION="542.0.0"

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "${os}" in
    Linux) os="linux" ;;
    Darwin) os="darwin" ;;
    *)
      echo "Unsupported operating system: ${os}" >&2
      exit 1
      ;;
  esac

  case "${arch}" in
    x86_64) arch="x86_64" ;;
    amd64) arch="x86_64" ;;
    arm64) arch="arm" ;; # Google distributes arm builds as 'arm'
    aarch64) arch="arm" ;;
    *)
      echo "Unsupported architecture: ${arch}" >&2
      exit 1
      ;;
  esac

  echo "${os}" "${arch}"
}

main() {
  read -r os arch < <(detect_platform)

  local sdk_archive="google-cloud-cli-${SDK_VERSION}-${os}-${arch}.tar.gz"
  local sdk_url="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${sdk_archive}"
  local install_dir="${HOME}/google-cloud-sdk"

  if command -v gcloud >/dev/null 2>&1; then
    echo "gcloud already installed; skipping download."
  else
    echo "Downloading Google Cloud SDK ${SDK_VERSION} for ${os}/${arch}..."
    curl -# -L "${sdk_url}" -o "${sdk_archive}"
    tar -xf "${sdk_archive}"
    rm -f "${sdk_archive}"

    echo "Installing Google Cloud SDK..."
    "./google-cloud-sdk/install.sh" --quiet
  fi

  if [ -d "${install_dir}" ]; then
    if [ -f "${install_dir}/path.bash.inc" ] && [ -n "${BASH:-}" ]; then
      # shellcheck source=/dev/null
      source "${install_dir}/path.bash.inc"
    elif [ -f "${install_dir}/path.zsh.inc" ] && [ -n "${ZSH_VERSION:-}" ]; then
      # shellcheck source=/dev/null
      source "${install_dir}/path.zsh.inc"
    else
      export PATH="${install_dir}/bin:${PATH}"
    fi
  fi

  echo "Authenticating with Google Cloud (browser window will open)..."
  gcloud auth login

  echo "Setting default project to ${PROJECT_ID}..."
  gcloud config set project "${PROJECT_ID}"

  echo "Installation complete. Verify with: gcloud info"
}

main "$@"
