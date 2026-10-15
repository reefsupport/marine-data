#!/usr/bin/env bash
# marine-data release VM: build one release flavour, gate it, print the publish plan (dry run).
#
# Runs once, as root, from cloud-init user-data (create.sh prepends the non-secret settings as
# `export` lines) or by hand: RELEASE_ID=v1.1 FLAVOUR=open MARINEDATA_REF=<sha> bash bootstrap.sh
# (run it as a file, not piped into bash).
#
# Set-up (idempotent): docker + git, the volume at /data, a clone of MARINEDATA_REF, the image,
# and /usr/local/bin/marinedata-run (runs a command in the image with this VM's mounts).
# Stages (STAGES="gate plan" runs only those):
#   lock    the lock and the frozen split map are in the checkout; print their sha256
#   build   marinedata release build   -> /data/releases/$RELEASE_ID/$FLAVOUR
#   export  hf_export + hf_card        -> /data/export/$RELEASE_ID/$FLAVOUR (the folder to upload)
#   gate    marinedata release gate    -> GATES.json in the export folder
#   plan    marinedata release publish -> the DRY RUN plan; this script never passes --go
#
# Secrets never travel in user-data: it is stored in plain text with the server and on the VM.
# The operator copies $SECRETS_FILE (MARINEDATA_S3_* lines, mode 600) to the VM and this script
# waits for it. HF_TOKEN is not in that file: it is exported by hand for `publish --go` only.
#
# Settings (environment). Required: MARINEDATA_REF (a full commit sha; a tag can move),
# RELEASE_ID, FLAVOUR (open|nc). Optional: REPO_URL, LOCK_FILE, SPLIT_MAP_FILE,
# RELEASE_BUILD_ARGS (extra `release build` flags), EXPORT_ARGS (extra hf_export flags),
# SECRETS_FILE, SECRETS_WAIT_SECONDS, STAGES.

set -Eeuo pipefail

REPO_URL=${REPO_URL:-https://github.com/reefsupport/marine-data.git}
REPO_DIR=${REPO_DIR:-/opt/marine-data}
LOCK_FILE=${LOCK_FILE:-release.lock.yaml}
SPLIT_MAP_FILE=${SPLIT_MAP_FILE:-registry/SPLIT_MAP.json}
RELEASE_BUILD_ARGS=${RELEASE_BUILD_ARGS:-}
EXPORT_ARGS=${EXPORT_ARGS:-}
SECRETS_FILE=${SECRETS_FILE:-/root/marinedata.env}
SECRETS_WAIT_SECONDS=${SECRETS_WAIT_SECONDS:-3600}
STAGES=${STAGES:-lock build export gate plan}
DATA_DIR=/data
LOG_DIR=$DATA_DIR/logs
RUN_WRAPPER=/usr/local/bin/marinedata-run

say() { printf 'bootstrap: %s\n' "$*"; }
die() {
  printf 'bootstrap: %s\n' "$*" >&2
  exit 1
}

validate_settings() {
  [[ -n ${MARINEDATA_REF:-} ]] || die "MARINEDATA_REF is required (a full commit sha is best)"
  [[ -n ${RELEASE_ID:-} ]] || die "RELEASE_ID is required (e.g. v1.1)"
  [[ ${FLAVOUR:-} == open || ${FLAVOUR:-} == nc ]] || die "FLAVOUR must be open or nc, got '${FLAVOUR:-}'"
  [[ $MARINEDATA_REF =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || die "MARINEDATA_REF '$MARINEDATA_REF' is not a plain git ref"
  [[ $RELEASE_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RELEASE_ID '$RELEASE_ID' may only hold letters, digits, '.', '_' and '-'"
  [[ $LOCK_FILE != /* && $LOCK_FILE != *..* ]] || die "LOCK_FILE '$LOCK_FILE' must be a path inside the repo"
  [[ $SPLIT_MAP_FILE != /* && $SPLIT_MAP_FILE != *..* ]] || die "SPLIT_MAP_FILE '$SPLIT_MAP_FILE' must be a path inside the repo"
  [[ $REPO_DIR =~ ^/[A-Za-z0-9._/-]+$ ]] || die "REPO_DIR '$REPO_DIR' must be a plain absolute path"
  [[ $SECRETS_FILE =~ ^/[A-Za-z0-9._/-]+$ ]] || die "SECRETS_FILE '$SECRETS_FILE' must be a plain absolute path"
  [[ $REPO_URL != -* ]] || die "REPO_URL '$REPO_URL' must not start with '-'"
  [[ $SECRETS_WAIT_SECONDS =~ ^[0-9]+$ ]] || die "SECRETS_WAIT_SECONDS must be a whole number"
  local stage
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  for stage in $STAGES; do
    case $stage in
      lock | build | export | gate | plan) ;;
      *) die "unknown stage '$stage' in STAGES (lock build export gate plan)" ;;
    esac
  done
}

derive_paths() {
  REL_DIR=$DATA_DIR/releases/$RELEASE_ID/$FLAVOUR
  EXPORT_DIR=$DATA_DIR/export/$RELEASE_ID/$FLAVOUR
  SUMMARY=$DATA_DIR/export/$RELEASE_ID/$FLAVOUR.summary.json
}

install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq docker.io git ca-certificates
  apt-get install -y -qq docker-buildx || say "docker-buildx is not available; the classic builder will do"
  systemctl enable --now docker
}

# The Hetzner volume shows up as /dev/disk/by-id/scsi-0HC_Volume_<id>. A blank one is formatted;
# one that already has a filesystem is mounted as it is and never reformatted.
attach_volume() {
  if mountpoint -q "$DATA_DIR"; then
    return 0
  fi
  local matches waited=0 dev fs
  while ((waited < 120)); do
    matches=(/dev/disk/by-id/scsi-0HC_Volume_*)
    if [[ -e ${matches[0]} ]]; then
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  [[ -e ${matches[0]} ]] || die "no Hetzner volume attached (/dev/disk/by-id/scsi-0HC_Volume_*)"
  ((${#matches[@]} == 1)) || die "more than one volume attached (${matches[*]}); detach the others"
  dev=${matches[0]}
  fs=$(blkid -o value -s TYPE "$dev" || true)
  if [[ -z $fs ]]; then
    say "formatting the blank volume $dev as ext4"
    mkfs.ext4 -q -F "$dev"
  fi
  mkdir -p "$DATA_DIR"
  mount -o discard,defaults "$dev" "$DATA_DIR"
  if ! grep -qF " $DATA_DIR " /etc/fstab; then
    printf '%s %s ext4 discard,nofail,defaults 0 0\n' "$dev" "$DATA_DIR" >>/etc/fstab
  fi
}

# uid 10001 is the non-root user of the image (Dockerfile).
prepare_data_dir() {
  chown 10001:10001 "$DATA_DIR"
  install -d -o 10001 -g 10001 "$DATA_DIR/cache" "$DATA_DIR/hf" "$DATA_DIR/releases" "$DATA_DIR/export"
  install -d "$LOG_DIR"
}

clone_repo() {
  export GIT_TERMINAL_PROMPT=0
  if [[ ! -d $REPO_DIR/.git ]]; then
    git clone --quiet --no-checkout "$REPO_URL" "$REPO_DIR"
  fi
  git -C "$REPO_DIR" fetch --quiet --tags origin
  if ! git -C "$REPO_DIR" checkout --quiet --detach "$MARINEDATA_REF" 2>/dev/null; then
    git -C "$REPO_DIR" fetch --quiet origin "$MARINEDATA_REF"
    git -C "$REPO_DIR" checkout --quiet --detach FETCH_HEAD
  fi
  COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
  say "checked out $MARINEDATA_REF = $COMMIT"
  if [[ ! $MARINEDATA_REF =~ ^[0-9a-f]{40}$ ]]; then
    say "warning: $MARINEDATA_REF is not a full commit sha; a tag or branch can move"
  fi
}

build_image() {
  IMAGE="marinedata:${RELEASE_ID}-${COMMIT:0:12}"
  say "docker build $IMAGE"
  docker build --tag "$IMAGE" "$REPO_DIR"
}

secrets_ready() {
  [[ -s $SECRETS_FILE ]] || return 1
  local key
  for key in MARINEDATA_S3_ACCESS_KEY MARINEDATA_S3_SECRET_KEY; do
    grep -q "^${key}=." "$SECRETS_FILE" || return 1
  done
}

# Bounded wait for the operator's secrets file, then refuse a file that is readable by others,
# owned by someone else, or that carries the Hub token. Values are never printed.
wait_for_secrets() {
  local waited=0 key
  until secrets_ready; do
    ((waited < SECRETS_WAIT_SECONDS)) || die "no complete $SECRETS_FILE after ${SECRETS_WAIT_SECONDS}s: copy it (README step 4), then re-run with STAGES=\"$STAGES\""
    if ((waited % 300 == 0)); then
      say "waiting for $SECRETS_FILE (${waited}s of ${SECRETS_WAIT_SECONDS}s)"
    fi
    sleep 10
    waited=$((waited + 10))
  done
  if [[ -n $(find "$SECRETS_FILE" -maxdepth 0 \( -perm -g=r -o -perm -g=w -o -perm -g=x -o -perm -o=r -o -perm -o=w -o -perm -o=x \)) ]]; then
    die "$SECRETS_FILE is readable by group or others: chmod 600 $SECRETS_FILE"
  fi
  [[ -z $(find "$SECRETS_FILE" -maxdepth 0 ! -user "$(id -un)") ]] || die "$SECRETS_FILE must be owned by $(id -un)"
  if grep -q '^HF_TOKEN=' "$SECRETS_FILE"; then
    die "HF_TOKEN must not be in $SECRETS_FILE: the build stages would see it. Export it by hand for publish --go (README step 6)"
  fi
  for key in MARINEDATA_S3_ENDPOINT MARINEDATA_S3_REGION; do
    if ! grep -q "^${key}=." "$SECRETS_FILE"; then
      say "note: $key is not set in $SECRETS_FILE (fine only if the bucket client has a default for it)"
    fi
  done
}

# marinedata-run [--py] ARGS...: run marinedata (or python with --py) in the image, with /data
# and the checkout's data/ (task labels, read-only) mounted. HF_TOKEN is passed through only
# when the calling shell has it; the S3 env file is added only while it exists.
write_wrapper() {
  cat >"$RUN_WRAPPER" <<EOF
#!/usr/bin/env bash
# Written by bootstrap.sh. Usage: marinedata-run release publish ... | marinedata-run --py -m MODULE ...
set -euo pipefail
args=(--rm -e HF_TOKEN -v ${DATA_DIR}:${DATA_DIR} -v ${REPO_DIR}/data:/app/data:ro --workdir ${DATA_DIR})
if [[ -r ${SECRETS_FILE} ]]; then
  args+=(--env-file ${SECRETS_FILE})
fi
if [[ \${1:-} == --py ]]; then
  shift
  args+=(--entrypoint python)
fi
exec docker run "\${args[@]}" ${IMAGE} "\$@"
EOF
  chmod 755 "$RUN_WRAPPER"
}

# Each stage returns non-zero on failure; run_stage logs it and stops the run. A stage runs in a
# pipeline subshell, so every command that is not the last one carries its own `|| return 1`.
run_stage() {
  local name=$1 rc=0
  shift
  say "stage $name: start"
  "$@" 2>&1 | tee -a "$LOG_DIR/$name.log" || rc=$?
  if ((rc == 0)); then
    printf '%s %s ok\n' "$(date -u +%FT%TZ)" "$name" >>"$LOG_DIR/STATUS"
    say "stage $name: ok"
  else
    printf '%s %s FAILED(%s)\n' "$(date -u +%FT%TZ)" "$name" "$rc" >>"$LOG_DIR/STATUS"
    die "stage $name failed (exit $rc); see $LOG_DIR/$name.log"
  fi
}

stage_lock() {
  local f
  for f in "$LOCK_FILE" "$SPLIT_MAP_FILE"; do
    [[ -s $REPO_DIR/$f ]] || die "$f is not in $MARINEDATA_REF: commit the release lock and the frozen split map first"
  done
  (cd "$REPO_DIR" && sha256sum "$LOCK_FILE" "$SPLIT_MAP_FILE")
}

stage_build() {
  local flags=()
  # shellcheck disable=SC2206  # the extra flags are a deliberate word list
  flags=($RELEASE_BUILD_ARGS)
  "$RUN_WRAPPER" release build --release "$RELEASE_ID" --flavour "$FLAVOUR" \
    --lock "/app/$LOCK_FILE" --split-map "/app/$SPLIT_MAP_FILE" --out "$DATA_DIR" \
    ${flags[@]+"${flags[@]}"}
}

stage_export() {
  local flags=()
  # shellcheck disable=SC2206  # the extra flags are a deliberate word list
  flags=($EXPORT_ARGS)
  if [[ -n $(ls -A "$EXPORT_DIR" 2>/dev/null) ]]; then
    die "$EXPORT_DIR is not empty: remove it to export again"
  fi
  install -d -o 10001 -g 10001 "$(dirname "$EXPORT_DIR")" || return 1
  "$RUN_WRAPPER" --py -m marinedata.hf_export --release-dir "$REL_DIR" --out "$EXPORT_DIR" \
    --summary "$SUMMARY" --flavour "$FLAVOUR" ${flags[@]+"${flags[@]}"} || return 1
  "$RUN_WRAPPER" --py -m marinedata.hf_card --release-dir "$REL_DIR" --summary "$SUMMARY" \
    --out "$EXPORT_DIR" --flavour "$FLAVOUR"
}

# The gate suite (--all, --require-all) arrives with the gates branch; before that only the
# masks gate exists, and 'release publish' then refuses for the missing required gates.
stage_gate() {
  local help
  help=$("$RUN_WRAPPER" release gate --help) || return 1
  if [[ $help == *--require-all* ]]; then
    "$RUN_WRAPPER" release gate "$EXPORT_DIR" --all --require-all --flavour "$FLAVOUR" --release-dir "$REL_DIR"
  else
    say "WARNING: this ref's 'release gate' has no --all/--require-all; only the masks gate runs"
    "$RUN_WRAPPER" release gate --masks "$EXPORT_DIR"
  fi
}

stage_plan() {
  "$RUN_WRAPPER" release publish --release-id "$RELEASE_ID" --flavour "$FLAVOUR" \
    --dir "$EXPORT_DIR" --release-dir "$REL_DIR" --lock "/app/$LOCK_FILE"
}

print_next_steps() {
  cat <<EOF

bootstrap: dry run finished for $RELEASE_ID ($FLAVOUR) at $COMMIT.
  plan   : $LOG_DIR/plan.log     stage outcomes: $LOG_DIR/STATUS
  Read the plan and $EXPORT_DIR/GATES.json, copy the evidence back (README step 7), then publish by hand (README step 6):
    rm -f $SECRETS_FILE
    read -rs HF_TOKEN && export HF_TOKEN
    marinedata-run release publish --release-id $RELEASE_ID --flavour $FLAVOUR \\
      --dir $EXPORT_DIR --release-dir $REL_DIR --lock /app/$LOCK_FILE --go
EOF
}

main() {
  [[ $(id -u) -eq 0 ]] || die "run as root"
  validate_settings
  derive_paths
  install_packages
  attach_volume
  prepare_data_dir
  exec > >(tee -a "$LOG_DIR/bootstrap.log") 2>&1
  trap 'printf "bootstrap: FAILED near line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
  say "release $RELEASE_ID ($FLAVOUR) from $MARINEDATA_REF"
  clone_repo
  build_image
  local stage needs_secrets=0
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  for stage in $STAGES; do
    if [[ $stage == build || $stage == export ]]; then
      needs_secrets=1
    fi
  done
  if [[ " $STAGES " == *" lock "* ]]; then
    run_stage lock stage_lock
  fi
  if ((needs_secrets)); then
    wait_for_secrets
  fi
  write_wrapper
  # shellcheck disable=SC2086
  for stage in $STAGES; do
    if [[ $stage != lock ]]; then
      run_stage "$stage" "stage_$stage"
    fi
  done
  print_next_steps
}

if [[ ${BOOTSTRAP_SOURCE_ONLY:-0} != 1 ]]; then
  main "$@"
fi
