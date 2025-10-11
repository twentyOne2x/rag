#!/usr/bin/env bash
#
# Provision end-to-end CI/CD for deploying this project to Google Cloud Run via Cloud Build.
# The script relies entirely on gcloud / bash so you can stay in the terminal.
#
# Usage (defaults assume the current gcloud project/region):
#   ./scripts/setup_cloud_run_cicd.sh \
#       --project_id just-skyline-474622-e1 \
#       --region us-east1 \
#       --service_name rag-service \
#       --repo_owner twentyOne2x \
#       --repo_name rag \
#       --branch main
#
# Follows the Cloud Build configs checked into the repo:
#   - cloudbuild_app.yaml (build + push)
#   - cloudbuild_deploy_app.yaml (Cloud Run deploy)
#
# Requirements:
#   * gcloud >= 420.0.0 with application-default credentials that have owner/admin rights
#   * (Optional) Artifact Registry API will be enabled even though the sample uses GCR
#   * For GitHub triggers, create the GitHub App connection once via:
#       gcloud builds connections create github --project $PROJECT_ID --location global --installation-id <id> ...

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-east1}"
SERVICE_NAME="rag-service"
SERVICE_ACCOUNT_NAME="cloud-run-backend"
REPO_OWNER="twentyOne2x"
REPO_NAME="rag"
BRANCH_PATTERN="main"
TRIGGER_BUILD_NAME="rag-build"
TRIGGER_DEPLOY_NAME="rag-deploy"
GITHUB_CONNECTION=""
IMAGE_REGISTRY="us-east1-docker.pkg.dev"
IMAGE_REPOSITORY="rag"
IMAGE_NAME="rag"
BUILD_CONFIG="cloudbuild_app.yaml"
DEPLOY_CONFIG="cloudbuild_deploy_app.yaml"
SECRET_IDS=(
  "OPENAI_API_KEY"
  "PINECONE_API_KEY"
  "PINECONE_API_ENVIRONMENT"
)

AVAILABLE_SECRETS=()
CONNECTION_REGION=""
CONNECTION_ID=""
REPOSITORY_RESOURCE=""

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --project_id <id>         GCP project (defaults to current gcloud config)
  --region <region>         Cloud Run region (default: ${REGION})
  --service_name <name>     Cloud Run service name (default: ${SERVICE_NAME})
  --service_account <name>  Service account name (default: ${SERVICE_ACCOUNT_NAME})
  --repo_owner <owner>      GitHub organisation/user (required for triggers)
  --repo_name <name>        GitHub repository name (required for triggers)
  --branch <pattern>        Branch regex for triggers (default: ${BRANCH_PATTERN})
  --github_connection <id>  gcloud GitHub connection ID (run 'gcloud builds connections list')
  --image_repo <host>       Container registry host (default: ${IMAGE_REGISTRY})
  --image_repository <name> Artifact Registry repository (default: ${IMAGE_REPOSITORY})
  --image_name <name>       Container image name (default: ${IMAGE_NAME})
  --build_trigger_name <n>  Cloud Build trigger name for builds (default: ${TRIGGER_BUILD_NAME})
  --deploy_trigger_name <n> Cloud Build trigger name for deploy (default: ${TRIGGER_DEPLOY_NAME})
  -h, --help                Show this message

Examples:
  $0 --project_id just-skyline-474622-e1 --repo_owner your-org --repo_name rag --github_connection projects/my-project/locations/global/connections/my-conn
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project_id) PROJECT_ID="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service_name) SERVICE_NAME="$2"; shift 2 ;;
    --service_account) SERVICE_ACCOUNT_NAME="$2"; shift 2 ;;
    --repo_owner) REPO_OWNER="$2"; shift 2 ;;
    --repo_name) REPO_NAME="$2"; shift 2 ;;
    --branch) BRANCH_PATTERN="$2"; shift 2 ;;
    --github_connection) GITHUB_CONNECTION="$2"; shift 2 ;;
    --image_repo) IMAGE_REGISTRY="$2"; shift 2 ;;
    --image_repository) IMAGE_REPOSITORY="$2"; shift 2 ;;
    --image_name) IMAGE_NAME="$2"; shift 2 ;;
    --build_trigger_name) TRIGGER_BUILD_NAME="$2"; shift 2 ;;
    --deploy_trigger_name) TRIGGER_DEPLOY_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: --project_id or gcloud config project is required." >&2
  exit 1
fi

IMAGE_PATH="${IMAGE_REGISTRY}/${PROJECT_ID}/${IMAGE_REPOSITORY}/${IMAGE_NAME}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

banner() {
  printf "\n\033[1;34m%s\033[0m\n" "$1"
}

ensure_api() {
  local api="$1"
  gcloud services enable "${api}" --project "${PROJECT_ID}" >/dev/null
}

create_service_account() {
  if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Service account ${SERVICE_ACCOUNT_EMAIL} already exists."
  else
    gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
      --project "${PROJECT_ID}" \
      --display-name "Cloud Run runtime for ${SERVICE_NAME}"
  fi
}

grant_iam_roles() {
  local cb_sa
  cb_sa="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

  # Runtime SA permissions (Secrets + Cloud Logging / Trace + Pub/Sub (optional))
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/secretmanager.secretAccessor" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/logging.logWriter" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "roles/run.invoker" >/dev/null

  # Cloud Build SA permissions (build/deploy)
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${cb_sa}" \
    --role "roles/run.admin" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${cb_sa}" \
    --role "roles/iam.serviceAccountUser" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${cb_sa}" \
    --role "roles/storage.admin" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${cb_sa}" \
    --role "roles/secretmanager.secretAccessor" >/dev/null
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${cb_sa}" \
    --role "roles/artifactregistry.writer" >/dev/null 2>/dev/null || true
}

create_secrets() {
  if [[ -f .env ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
  fi
  local missing=()
  for secret_id in "${SECRET_IDS[@]}"; do
    if gcloud secrets describe "${secret_id}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
      echo "Secret ${secret_id} already exists."
      AVAILABLE_SECRETS+=("${secret_id}")
    else
      local value="${!secret_id:-}"
      if [[ -z "${value}" ]]; then
        missing+=("${secret_id}")
        continue
      fi
      echo "Creating secret ${secret_id}."
      gcloud secrets create "${secret_id}" --project "${PROJECT_ID}" --replication-policy="automatic" >/dev/null
      printf "%s" "${value}" | gcloud secrets versions add "${secret_id}" --data-file=- --project "${PROJECT_ID}" >/dev/null
      AVAILABLE_SECRETS+=("${secret_id}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing values for the following secrets in .env (or environment):" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "Populate them and rerun the script." >&2
    exit 1
  fi
}

ensure_repository() {
  if [[ -z "${GITHUB_CONNECTION}" || -z "${REPO_OWNER}" || -z "${REPO_NAME}" ]]; then
    echo "Skipping repository creation; repo_owner/repo_name/github_connection not set."
    return
  fi
  local repo_id="${REPO_NAME}"
  if gcloud alpha builds repositories describe "${repo_id}" \
       --connection="${CONNECTION_ID}" \
       --project "${PROJECT_ID}" \
       --region "${CONNECTION_REGION}" >/dev/null 2>&1; then
    REPOSITORY_RESOURCE="projects/${PROJECT_ID}/locations/${CONNECTION_REGION}/connections/${CONNECTION_ID}/repositories/${repo_id}"
    echo "Repository ${repo_id} already exists."
    return
  fi
  echo "Creating Cloud Build repository ${repo_id}."
  gcloud alpha builds repositories create "${repo_id}" \
    --connection="${CONNECTION_ID}" \
    --remote-uri="https://github.com/${REPO_OWNER}/${REPO_NAME}.git" \
    --project "${PROJECT_ID}" \
    --region "${CONNECTION_REGION}" >/dev/null
  REPOSITORY_RESOURCE="projects/${PROJECT_ID}/locations/${CONNECTION_REGION}/connections/${CONNECTION_ID}/repositories/${repo_id}"
}

build_update_secrets_csv() {
  if (( ${#AVAILABLE_SECRETS[@]} == 0 )); then
    echo ""
    return
  fi
  local parts=()
  for name in "${AVAILABLE_SECRETS[@]}"; do
    parts+=("${name}=${name}:latest")
  done
  local old_ifs="${IFS}"
  IFS=','
  local joined="${parts[*]}"
  IFS="${old_ifs}"
  echo "${joined}"
}

deploy_bootstrap_revision() {
  local secrets_csv
  secrets_csv="$(build_update_secrets_csv)"
  local deploy_args=(
    gcloud run deploy "${SERVICE_NAME}"
    --project "${PROJECT_ID}"
    --region "${REGION}"
    --image "${IMAGE_PATH}:latest"
    --allow-unauthenticated
    --service-account "${SERVICE_ACCOUNT_EMAIL}"
    --set-env-vars "PYTHONUNBUFFERED=1"
  )
  if [[ -n "${secrets_csv}" ]]; then
    deploy_args+=(--update-secrets "${secrets_csv}")
  fi
  deploy_args+=(--port "8080")
  "${deploy_args[@]}"
}

create_trigger() {
  local name="$1"
  local config="$2"
  if [[ -z "${REPOSITORY_RESOURCE}" ]]; then
    echo "Skipping trigger ${name}; repository resource not available."
    return
  fi
  if gcloud builds triggers describe "${name}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "Trigger ${name} already exists."
    return
  fi
  gcloud builds triggers create github \
    --name "${name}" \
    --project "${PROJECT_ID}" \
    --region "${CONNECTION_REGION}" \
    --repository "${REPOSITORY_RESOURCE}" \
    --branch-pattern "${BRANCH_PATTERN}" \
    --build-config "${config}" \
    --substitutions "_SERVICE_NAME=${SERVICE_NAME},_REGION=${REGION}"
}

main() {
  if [[ -n "${GITHUB_CONNECTION}" ]]; then
    CONNECTION_REGION="$(echo "${GITHUB_CONNECTION}" | awk -F'/' '{for(i=1;i<=NF;i++) if($i=="locations") {print $(i+1); exit}}')"
    CONNECTION_ID="$(echo "${GITHUB_CONNECTION}" | awk -F'/' '{print $NF}')"
  fi
  if [[ -z "${CONNECTION_REGION}" ]]; then
    CONNECTION_REGION="global"
  fi
  banner "Getting the project ready (using gcloud config set project ${PROJECT_ID})"
  gcloud config set project "${PROJECT_ID}" >/dev/null

  banner "Turning on the services the app needs (enabling Cloud Run, Cloud Build, Secret Manager, Artifact Registry, Compute)"
  ensure_api run.googleapis.com
  ensure_api cloudbuild.googleapis.com
  ensure_api secretmanager.googleapis.com
  ensure_api artifactregistry.googleapis.com
  ensure_api compute.googleapis.com

  banner "Creating or reusing the runtime service account (iam.serviceAccounts)"
  create_service_account

  banner "Giving build and runtime accounts the right permissions (binding IAM roles)"
  grant_iam_roles

  banner "Storing required API keys and config (Secret Manager secrets)"
  create_secrets

  banner "Preparing a container registry home (Artifact Registry docker repo)"
  gcloud artifacts repositories describe "${IMAGE_REPOSITORY}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "${IMAGE_REPOSITORY}" \
      --repository-format docker \
      --location "${REGION}" \
      --description "Docker repo for ${SERVICE_NAME}" \
      --project "${PROJECT_ID}" || true

  banner "Rolling out an initial Cloud Run revision (gcloud run deploy)"
  deploy_bootstrap_revision || echo "Initial deploy skipped (image may not exist yet)."

  banner "Setting up Cloud Build repository (GitHub connection)"
  ensure_repository

  banner "Hooking Cloud Build to your repo (GitHub triggers, cloudbuild.yaml)"
  create_trigger "${TRIGGER_BUILD_NAME}" "${BUILD_CONFIG}"
  create_trigger "${TRIGGER_DEPLOY_NAME}" "${DEPLOY_CONFIG}"

  banner "All done setting up the pipeline (CI/CD ready)"
  cat <<EOF

Next steps:
  1. Push a commit to ${BRANCH_PATTERN} to trigger ${TRIGGER_BUILD_NAME} and ${TRIGGER_DEPLOY_NAME}.
  2. Monitor Cloud Build: https://console.cloud.google.com/cloud-build/builds?project=${PROJECT_ID}
  3. Cloud Run service URL: $(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)' 2>/dev/null || echo "<pending>")

EOF
}

main "$@"
