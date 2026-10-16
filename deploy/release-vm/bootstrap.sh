#!/usr/bin/env bash
# marine-data release VM: build one release flavour, gate it, print the publish plan (dry run).
#
# Runs once, as root, from cloud-init user-data (create.sh prepends the non-secret settings as
# `export` lines) or by hand: RELEASE_ID=v1.1 FLAVOUR=open MARINEDATA_REF=<sha> bash bootstrap.sh
# (run it as a file, not piped into bash). On the Proxmox VM the settings are in
# /etc/marinedata/settings.env: `set -a; . /etc/marinedata/settings.env; set +a` first.
#
# Set-up (idempotent): docker + git, the data disk at /data, a clone of MARINEDATA_REF, the image,
# and /usr/local/bin/marinedata-run (runs a command in the image with this VM's mounts).
# STAGES="" (set but empty) stops after the set-up: no stage, no wait for the secrets file.
# Re-run on a VM set up earlier with another ref: give it the new MARINEDATA_REF (export it after
# sourcing settings.env, which keeps the first-boot ref) and the STAGES you want. The checkout is
# fetched and moved with `git checkout --detach`: never `reset --hard`, never `clean`, and local
# changes to tracked files stop the move. The image is rebuilt under a tag with the new commit, /data
# keeps its contents (a disk with a filesystem is mounted, never formatted) and only those STAGES run.
# Run it with BOOTSTRAP_DRY_RUN=1 first: it shows where the checkout is and where it would go.
# Stages (STAGES="gate plan" runs only those; unset runs all of them, in this order):
#   lock      the lock and the frozen split map are in the checkout; print their sha256
#   build     marinedata release build   -> /data/releases/$RELEASE_ID/$FLAVOUR
#   export    hf_export + hf_card        -> /data/export/$RELEASE_ID/$FLAVOUR (the folder to upload)
#   croissant marinedata release croissant -> croissant.json in the export folder
#   gate      marinedata release gate    -> GATES.json in the export folder
#   plan      marinedata release publish -> the DRY RUN plan; this script never passes --go
#
# The data disk is DATA_DEV (environment, else the DATA_DEV= line of /etc/marinedata/settings.env),
# else the Hetzner volume /dev/disk/by-id/scsi-0HC_Volume_*, else the Proxmox disk
# /dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_drive-scsi1. A blank disk is formatted and one that holds a
# filesystem is mounted as it is. One with a partition table, partitions, a mount or the root
# filesystem is refused.
#
# Secrets never travel in user-data: it is stored in plain text with the server and on the VM.
# The operator copies $SECRETS_FILE (MARINEDATA_S3_* lines, mode 600) to the VM and this script
# waits for it. HF_TOKEN is not in that file: it is exported by hand for `publish --go` only.
#
# Settings (environment). Required: MARINEDATA_REF (a full commit sha; a tag can move),
# RELEASE_ID, FLAVOUR (open|nc). Optional: REPO_URL, LOCK_FILE, SPLIT_MAP_FILE, DATA_DEV,
# RELEASE_BUILD_ARGS (extra `release build` flags), EXPORT_ARGS (extra hf_export flags),
# SECRETS_FILE, SECRETS_WAIT_SECONDS, STAGES.
# BOOTSTRAP_DRY_RUN=1 prints the stage list and what would happen to the data disk, then exits
# without changing anything (exit 1 if the disk would be refused). It needs no root.

set -Eeuo pipefail

REPO_URL=${REPO_URL:-https://github.com/reefsupport/marine-data.git}
REPO_DIR=${REPO_DIR:-/opt/marine-data}
LOCK_FILE=${LOCK_FILE:-release.lock.yaml}
SPLIT_MAP_FILE=${SPLIT_MAP_FILE:-registry/SPLIT_MAP.json}
RELEASE_BUILD_ARGS=${RELEASE_BUILD_ARGS:-}
EXPORT_ARGS=${EXPORT_ARGS:-}
# 'release gate --v11' options (docs/GATES.md), a plain word list: --v1-dir DIR (the frozen v1.0
# release dir, G2), --hidden-test MANIFEST, --benchmark MANIFEST, --nc-id ID.
GATE_V11_ARGS=${GATE_V11_ARGS:-}
# Extra 'release publish' options for the plan, e.g. "--v1-dir /data/v1.0-repo" (the delta against v1.0).
PLAN_ARGS=${PLAN_ARGS:-}
# The privacy gates (G8) run real YuNet: the pinned, sha256-verified model baked into the image.
# The wrapper passes this path; the gates never download a model.
YUNET_MODEL=${YUNET_MODEL:-/opt/models/face_detection_yunet_2023mar.onnx}
SECRETS_FILE=${SECRETS_FILE:-/root/marinedata.env}
SECRETS_WAIT_SECONDS=${SECRETS_WAIT_SECONDS:-3600}
# `-` not `:-`: STAGES set but empty means set-up only; only an unset STAGES runs every stage.
STAGES=${STAGES-lock build export croissant gate plan}
DATA_DIR=/data
DATA_DEV=${DATA_DEV:-}
SETTINGS_ENV=${SETTINGS_ENV:-/etc/marinedata/settings.env}
HETZNER_DEV_GLOB='/dev/disk/by-id/scsi-0HC_Volume_*'
PVE_DATA_DEV=/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_drive-scsi1
LOG_DIR=$DATA_DIR/logs
RUN_WRAPPER=/usr/local/bin/marinedata-run

say() { printf 'bootstrap: %s\n' "$*"; }
die() {
  printf 'bootstrap: %s\n' "$*" >&2
  exit 1
}

# True for an empty value or a plain /dev path (no spaces, no '..').
plain_dev() {
  [[ -z $1 ]] || [[ $1 =~ ^/dev/[A-Za-z0-9._:+/-]+$ && $1 != *..* ]]
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
  plain_dev "$DATA_DEV" || die "DATA_DEV '$DATA_DEV' must be a plain /dev path (for example /dev/disk/by-id/...)"
  local stage
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  for stage in $STAGES; do
    case $stage in
      lock | build | export | croissant | gate | plan) ;;
      *) die "unknown stage '$stage' in STAGES (lock build export croissant gate plan)" ;;
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

# Is /data already a mount point? (mountpoint is Linux-only; a missing one counts as "no".)
data_mounted() {
  command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$DATA_DIR"
}

# The DATA_DEV= line of $SETTINGS_ENV, read and never sourced; empty when there is none.
settings_data_dev() {
  local line
  [[ -r $SETTINGS_ENV ]] || return 0
  line=$(sed -n 's/^DATA_DEV=//p' "$SETTINGS_ENV" | tail -n 1)
  line=${line#[\"\']}
  printf '%s' "${line%[\"\']}"
}

# Sets DATA_DEV and waits up to $1 s for the device: DATA_DEV from the environment or $SETTINGS_ENV
# wins; else the Hetzner volume (/dev/disk/by-id/scsi-0HC_Volume_*); else the Proxmox scsi1 disk.
resolve_data_dev() {
  local max_wait=$1 waited=0 matches
  if [[ -z $DATA_DEV ]]; then
    DATA_DEV=$(settings_data_dev)
    plain_dev "$DATA_DEV" || die "DATA_DEV '$DATA_DEV' in $SETTINGS_ENV must be a plain /dev path"
  fi
  while :; do
    if [[ -n $DATA_DEV ]]; then
      if [[ -e $DATA_DEV ]]; then
        return 0
      fi
    else
      matches=(/dev/disk/by-id/scsi-0HC_Volume_*)
      if [[ -e ${matches[0]} ]]; then
        ((${#matches[@]} == 1)) || die "more than one volume attached (${matches[*]}); detach the others or set DATA_DEV"
        DATA_DEV=${matches[0]}
        return 0
      fi
      if [[ -e $PVE_DATA_DEV ]]; then
        DATA_DEV=$PVE_DATA_DEV
        return 0
      fi
    fi
    ((waited < max_wait)) || break
    sleep 5
    waited=$((waited + 5))
  done
  if [[ -n $DATA_DEV ]]; then
    die "DATA_DEV $DATA_DEV does not exist"
  fi
  die "no data disk found: set DATA_DEV, or attach the Hetzner volume ($HETZNER_DEV_GLOB) or the Proxmox scsi1 disk ($PVE_DATA_DEV)"
}

# Prints why $1 must not be formatted and returns 1; prints nothing and returns 0 for a blank,
# unused disk. A partitioned whole disk has no blkid TYPE, so the blkid test alone would format it.
check_blank_disk() {
  local dev=$1 pttype kids mounts
  pttype=$(lsblk -no PTTYPE "$dev" 2>/dev/null) || { printf 'lsblk cannot read it'; return 1; }
  kids=$(lsblk -nro TYPE "$dev" 2>/dev/null) || { printf 'lsblk cannot read it'; return 1; }
  mounts=$(lsblk -nro MOUNTPOINTS "$dev" 2>/dev/null) || { printf 'lsblk cannot read it'; return 1; }
  pttype=$(printf '%s\n' "$pttype" | awk 'NF { print $1; exit }')
  if [[ -n $pttype ]]; then
    printf 'it has a partition table (%s)' "$pttype"
    return 1
  fi
  if (($(printf '%s\n' "$kids" | grep -c .) > 1)); then
    printf 'it has partitions or other child devices'
    return 1
  fi
  if printf '%s\n' "$mounts" | grep -qx /; then
    printf 'it backs /'
    return 1
  fi
  if printf '%s\n' "$mounts" | grep -q .; then
    printf 'it or a child is mounted'
    return 1
  fi
  return 0
}

# A blank disk is formatted (-m 0: the container runs as uid 10001, so the 5 % root reserve would
# be unusable); one that already has a filesystem is mounted as it is and never reformatted.
attach_volume() {
  if data_mounted; then
    return 0
  fi
  local dev fs why uuid
  resolve_data_dev 120
  dev=$DATA_DEV
  fs=$(blkid -o value -s TYPE "$dev" || true)
  if [[ -z $fs ]]; then
    if ! why=$(check_blank_disk "$dev"); then
      die "refusing to format $dev: $why"
    fi
    say "formatting the blank disk $dev as ext4"
    mkfs.ext4 -q -F -m 0 "$dev"
  fi
  mkdir -p "$DATA_DIR"
  mount -o discard,defaults "$dev" "$DATA_DIR"
  if ! grep -qF " $DATA_DIR " /etc/fstab; then
    uuid=$(blkid -p -o value -s UUID "$dev")
    [[ -n $uuid ]] || die "no UUID on $dev after mounting it"
    printf 'UUID=%s %s ext4 discard,nofail,defaults 0 0\n' "$uuid" "$DATA_DIR" >>/etc/fstab
  fi
}

# uid 10001 is the non-root user of the image (Dockerfile).
prepare_data_dir() {
  chown 10001:10001 "$DATA_DIR"
  install -d -o 10001 -g 10001 "$DATA_DIR/cache" "$DATA_DIR/hf" "$DATA_DIR/releases" "$DATA_DIR/export" \
    "$DATA_DIR/tmp" "$DATA_DIR/torch"
  install -d "$LOG_DIR"
}

# True when a tracked file differs from HEAD. A clone made with --no-checkout has no index yet and
# counts as clean.
checkout_is_dirty() {
  [[ -e $REPO_DIR/.git/index ]] || return 1
  [[ -n $(git -C "$REPO_DIR" status --porcelain --untracked-files=no 2>/dev/null) ]]
}

# Safe to repeat with another MARINEDATA_REF: fetch, then move the checkout with a plain
# `git checkout --detach`. Never `reset --hard`, never `clean`. Local changes to tracked files stop a
# move to a different commit (git would carry them into the new ref and from there into the image);
# untracked files are left as they are.
clone_repo() {
  export GIT_TERMINAL_PROMPT=0
  if [[ ! -d $REPO_DIR/.git ]]; then
    git clone --quiet --no-checkout "$REPO_URL" "$REPO_DIR"
  fi
  local before target
  before=$(git -C "$REPO_DIR" rev-parse --verify --quiet HEAD || true)
  git -C "$REPO_DIR" fetch --quiet --tags origin
  if ! target=$(git -C "$REPO_DIR" rev-parse --verify --quiet "${MARINEDATA_REF}^{commit}"); then
    git -C "$REPO_DIR" fetch --quiet origin "$MARINEDATA_REF"
    target=$(git -C "$REPO_DIR" rev-parse --verify "FETCH_HEAD^{commit}")
  fi
  if [[ -n $before && $before != "$target" ]] && checkout_is_dirty; then
    die "$REPO_DIR has local changes to tracked files and will not move from ${before:0:12} to ${target:0:12}: bootstrap never resets or cleans. Commit, stash or revert them (git -C $REPO_DIR status), then run it again"
  fi
  git -C "$REPO_DIR" checkout --quiet --detach "$target"
  COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
  if [[ -n $before && $before != "$COMMIT" ]]; then
    say "moved $REPO_DIR from ${before:0:12} to ${COMMIT:0:12}"
  fi
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

# marinedata-run [--py] ARGS...: run marinedata (or python with --py) in the image, with /data,
# the checkout's data/ (task labels, read-only) and, when the ref has one, its releases/ (read-only
# at /app/releases: .dockerignore keeps it out of the image, so the lock and split map committed at
# the ref are the ones LOCK_FILE and SPLIT_MAP_FILE name). HF_TOKEN is passed through only
# when the calling shell has it; the S3 env file is added only while it exists. TMPDIR and
# TORCH_HOME sit on /data (the root disk is 32 GiB and also holds /var/lib/docker); --memory 14g
# makes an overrun kill the container, not sshd.
write_wrapper() {
  cat >"$RUN_WRAPPER" <<EOF
#!/usr/bin/env bash
# Written by bootstrap.sh. Usage: marinedata-run release publish ... | marinedata-run --py -m MODULE ...
set -euo pipefail
args=(--rm -e HF_TOKEN -e MARINEDATA_YUNET_MODEL=${YUNET_MODEL} -e TMPDIR=${DATA_DIR}/tmp -e TORCH_HOME=${DATA_DIR}/torch --memory 14g -v ${DATA_DIR}:${DATA_DIR} -v ${REPO_DIR}/data:/app/data:ro --workdir ${DATA_DIR})
if [[ -d ${REPO_DIR}/releases ]]; then
  args+=(-v ${REPO_DIR}/releases:/app/releases:ro)
fi
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
  stage_card
}

# hf_export writes the shards and the summary only. This stage is what renders README.md and writes
# LICENSE, NOTICE and CITATION.cff into the export folder, so the export holds no CITATION.cff yet and
# the card cannot cite one. It cites the repo's CITATION.cff inside the image, the same input as
# stage_croissant. --citation is absolute: the wrapper's workdir is /data, not /app.
stage_card() {
  "$RUN_WRAPPER" --py -m marinedata.hf_card --release-dir "$REL_DIR" --summary "$SUMMARY" \
    --out "$EXPORT_DIR" --flavour "$FLAVOUR" --citation /app/CITATION.cff
}

# The Hub repo id of $FLAVOUR, read from the checkout's registry/flavours.yaml (the host has no yaml tool).
flavour_repo_id() {
  awk -v want="$FLAVOUR" '
    /^flavours:/ { in_flavours = 1; next }
    /^[^ #]/ { in_flavours = 0 }
    in_flavours && /^  [^ #]/ { cur = $1; sub(/:$/, "", cur) }
    in_flavours && cur == want && $1 == "repo_id:" { v = $2; gsub(/\047|"/, "", v); print v; exit }
  ' "$REPO_DIR/registry/flavours.yaml"
}

# Croissant metadata for the export folder: gates/croissant.py skips without croissant.json and
# --require-all cannot pass. --citation is absolute: the wrapper's workdir is /data, not /app.
stage_croissant() {
  local repo_id
  repo_id=$(flavour_repo_id) || return 1
  [[ -n $repo_id ]] || die "no repo_id for flavour $FLAVOUR in $REPO_DIR/registry/flavours.yaml"
  "$RUN_WRAPPER" release croissant "$EXPORT_DIR" --release-dir "$REL_DIR" \
    --citation /app/CITATION.cff --repo-id "$repo_id"
}

# Both gate sets, against the same release dir: --all (the v1.0 gates, --require-all) and then
# --v11 (G1..G9). --v11 does not chain the v1.0 gates: it merges its entries into GATES.json, so
# 'release publish' stays BLOCKED until both ran. Both run even when the first fails, so GATES.json
# lists every failure. The privacy gate (G8, and the v1.0 one) loads the pinned YuNet model that
# the wrapper passes in MARINEDATA_YUNET_MODEL; the gates never download it.
stage_gate() {
  local help rc=0 v11=()
  # shellcheck disable=SC2206  # the extra flags are a deliberate word list
  v11=($GATE_V11_ARGS)
  help=$("$RUN_WRAPPER" release gate --help) || return 1
  if [[ $help == *--require-all* ]]; then
    "$RUN_WRAPPER" release gate "$EXPORT_DIR" --all --require-all --flavour "$FLAVOUR" \
      --release-dir "$REL_DIR" || rc=1
  else
    say "WARNING: this ref's 'release gate' has no --all/--require-all; only the masks gate runs"
    "$RUN_WRAPPER" release gate --masks "$EXPORT_DIR" || rc=1
  fi
  if [[ $help == *--v11* ]]; then
    "$RUN_WRAPPER" release gate "$EXPORT_DIR" --v11 --flavour "$FLAVOUR" \
      --release-dir "$REL_DIR" ${v11[@]+"${v11[@]}"} || rc=1
  else
    say "WARNING: this ref's 'release gate' has no --v11: G1..G9 are not written, a v1.1.0 publish stays BLOCKED"
  fi
  return "$rc"
}

stage_plan() {
  local flags=()
  # shellcheck disable=SC2206  # the extra flags are a deliberate word list
  flags=($PLAN_ARGS)
  "$RUN_WRAPPER" release publish --release-id "$RELEASE_ID" --flavour "$FLAVOUR" \
    --dir "$EXPORT_DIR" --release-dir "$REL_DIR" --lock "/app/$LOCK_FILE" ${flags[@]+"${flags[@]}"}
}

print_next_steps() {
  cat <<EOF

bootstrap: dry run finished for $RELEASE_ID ($FLAVOUR) at $COMMIT.
  plan   : $LOG_DIR/plan.log     stage outcomes: $LOG_DIR/STATUS
  Read the plan and $EXPORT_DIR/GATES.json, copy the evidence back (README step 7), then publish by hand (README step 6),
  one part per call (data:<config>, provenance, card, tag; the tag lands on the card commit). A failed part
  stops the loop: fix it and run the loop again, finished data parts upload nothing new:
    rm -f $SECRETS_FILE
    read -rs HF_TOKEN && export HF_TOKEN
    for part in \$(marinedata-run release publish --release-id $RELEASE_ID --flavour $FLAVOUR \\
        --dir $EXPORT_DIR --release-dir $REL_DIR --lock /app/$LOCK_FILE --list-parts); do
      marinedata-run release publish --release-id $RELEASE_ID --flavour $FLAVOUR \\
        --dir $EXPORT_DIR --release-dir $REL_DIR --lock /app/$LOCK_FILE --go --part "\$part" || break
    done
EOF
}

print_setup_done() {
  cat <<EOF

bootstrap: set-up finished for $RELEASE_ID ($FLAVOUR) at $COMMIT; no stage ran.
  $DATA_DIR is mounted, image $IMAGE is built, $RUN_WRAPPER is written.
  Run the stages (any subset, in order; build and export wait for $SECRETS_FILE, README step 4):
    STAGES="lock build export croissant gate plan" bash $REPO_DIR/deploy/release-vm/bootstrap.sh
EOF
}

# True when STAGES holds a stage that reads the bucket (build, export): only those wait for the secrets file.
stages_need_secrets() {
  local stage
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  for stage in $STAGES; do
    if [[ $stage == build || $stage == export ]]; then
      return 0
    fi
  done
  return 1
}

# What BOOTSTRAP_DRY_RUN=1 says about the data disk. Read-only: lsblk and blkid, no mount, no mkfs.
dry_run_device() {
  local fs why
  if data_mounted; then
    say "dry run: $DATA_DIR is already a mount point: the data disk is left alone"
    return 0
  fi
  resolve_data_dev 0
  say "dry run: data disk $DATA_DEV"
  if [[ $(id -u) -ne 0 ]]; then
    say "dry run: not root, so blkid may not see a filesystem on it"
  fi
  fs=$(blkid -o value -s TYPE "$DATA_DEV" 2>/dev/null || true)
  if [[ -n $fs ]]; then
    say "dry run: $DATA_DEV holds a $fs filesystem: it would be mounted at $DATA_DIR as it is, never reformatted"
  elif why=$(check_blank_disk "$DATA_DEV"); then
    say "dry run: $DATA_DEV is blank and unused: it would be formatted ext4 (-m 0) and mounted at $DATA_DIR"
  else
    die "refusing to format $DATA_DEV: $why"
  fi
}

# What BOOTSTRAP_DRY_RUN=1 says about the checkout. Read-only: rev-parse and status, no fetch.
dry_run_checkout() {
  local before
  if [[ ! -d $REPO_DIR/.git ]]; then
    say "dry run: no checkout at $REPO_DIR: it would be cloned from $REPO_URL and checked out at $MARINEDATA_REF"
    return 0
  fi
  before=$(git -C "$REPO_DIR" rev-parse --verify --quiet HEAD || true)
  if [[ $before == "$MARINEDATA_REF" ]]; then
    say "dry run: $REPO_DIR is already at $MARINEDATA_REF"
    return 0
  fi
  say "dry run: $REPO_DIR is at ${before:-no commit}: it would be fetched and checked out at $MARINEDATA_REF (checkout --detach; no reset, no clean) and the image rebuilt"
  if checkout_is_dirty; then
    die "refusing to move $REPO_DIR: it has local changes to tracked files (git -C $REPO_DIR status). Commit, stash or revert them first"
  fi
}

dry_run() {
  validate_settings
  derive_paths
  say "dry run: nothing is installed, mounted, formatted, cloned or built"
  say "dry run: release $RELEASE_ID ($FLAVOUR) from $MARINEDATA_REF"
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  set -- $STAGES
  say "dry run: stages=[$*]"
  if (($# == 0)); then
    say "dry run: STAGES is empty: set-up only, no stage runs and nothing waits for $SECRETS_FILE"
  elif stages_need_secrets; then
    say "dry run: build or export is in STAGES: the run waits for $SECRETS_FILE"
  else
    say "dry run: no build or export stage: the run does not wait for $SECRETS_FILE"
  fi
  dry_run_device
  dry_run_checkout
}

main() {
  if [[ ${BOOTSTRAP_DRY_RUN:-0} == 1 ]]; then
    dry_run
    return 0
  fi
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
  local stage
  if [[ " $STAGES " == *" lock "* ]]; then
    run_stage lock stage_lock
  fi
  if stages_need_secrets; then
    wait_for_secrets
  fi
  write_wrapper
  # shellcheck disable=SC2086  # STAGES is a deliberate word list
  for stage in $STAGES; do
    if [[ $stage != lock ]]; then
      run_stage "$stage" "stage_$stage"
    fi
  done
  if [[ -n ${STAGES//[[:space:]]/} ]]; then
    print_next_steps
  else
    print_setup_done
  fi
}

if [[ ${BOOTSTRAP_SOURCE_ONLY:-0} != 1 ]]; then
  main "$@"
fi
