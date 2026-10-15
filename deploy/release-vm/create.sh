#!/usr/bin/env bash
# create.sh: print (default) or run the Hetzner commands for one release-build VM.
#
#   create.sh --release-id v1.1 --flavour open --ref <full commit sha> [--ssh-key NAME] [options]
#
# --print-only (the default) writes the user-data file and prints the exact hcloud commands. It
# needs no token and makes no network call. --run executes those same commands: it needs
# HCLOUD_TOKEN in the environment (a fresh project token; the earlier Hetzner token is
# compromised), the hcloud CLI, --ssh-key, and a typed confirmation (or --yes). No bucket or Hub
# secret passes through here; the S3 secrets file is copied to the VM by hand (README step 4).

set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=bootstrap.sh
BOOTSTRAP_SOURCE_ONLY=1 . "$HERE/bootstrap.sh"

die() {
  printf 'create.sh: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: create.sh --release-id ID --flavour open|nc --ref SHA [options]

  --ssh-key NAME         hcloud ssh key installed for root (required with --run)
  --name NAME            server name (default marinedata-<id>-<flavour>); the volume is NAME-data
  --out-dir DIR          where user-data.sh is written (default $TMPDIR/marinedata-release-vm/NAME)
  --location L           default hel1         --server-type T   default cx32
  --volume-size GB       default 200          --os-image I      default ubuntu-24.04
  --lock-file PATH       lock inside the repo (default release.lock.yaml)
  --split-map-file PATH  frozen split map inside the repo (default registry/SPLIT_MAP.json)
  --build-args "FLAGS"   extra flags for 'release build', e.g. "--decon"
  --export-args "FLAGS"  extra flags for hf_export, e.g. "--quality /data/quality.parquet"
  --repo-url URL         default https://github.com/reefsupport/marine-data.git
  --print-only           write user-data and print the commands (default)
  --run                  run them: needs HCLOUD_TOKEN, --ssh-key, hcloud and a confirmation
  --yes                  skip the typed confirmation (still needs HCLOUD_TOKEN)
EOF
}

RELEASE_ID="" FLAVOUR="" REF="" SSH_KEY="" NAME="" OUT_DIR=""
LOCATION=hel1 SERVER_TYPE=cx32 VOLUME_GB=200 OS_IMAGE=ubuntu-24.04
OPT_LOCK_FILE="" OPT_SPLIT_MAP_FILE="" OPT_REPO_URL=""
MODE=print YES=0

while (($#)); do
  case $1 in
    --release-id) RELEASE_ID=${2:?--release-id needs a value}; shift 2 ;;
    --flavour) FLAVOUR=${2:?--flavour needs a value}; shift 2 ;;
    --ref) REF=${2:?--ref needs a value}; shift 2 ;;
    --ssh-key) SSH_KEY=${2:?--ssh-key needs a value}; shift 2 ;;
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --out-dir) OUT_DIR=${2:?--out-dir needs a value}; shift 2 ;;
    --location) LOCATION=${2:?--location needs a value}; shift 2 ;;
    --server-type) SERVER_TYPE=${2:?--server-type needs a value}; shift 2 ;;
    --volume-size) VOLUME_GB=${2:?--volume-size needs a value}; shift 2 ;;
    --os-image) OS_IMAGE=${2:?--os-image needs a value}; shift 2 ;;
    --lock-file) OPT_LOCK_FILE=${2:?--lock-file needs a value}; shift 2 ;;
    --split-map-file) OPT_SPLIT_MAP_FILE=${2:?--split-map-file needs a value}; shift 2 ;;
    --build-args) RELEASE_BUILD_ARGS=${2:?--build-args needs a value}; shift 2 ;;
    --export-args) EXPORT_ARGS=${2:?--export-args needs a value}; shift 2 ;;
    --repo-url) OPT_REPO_URL=${2:?--repo-url needs a value}; shift 2 ;;
    --print-only) MODE=print; shift ;;
    --run) MODE=run; shift ;;
    --yes) YES=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

MARINEDATA_REF=$REF
if [[ -n $OPT_LOCK_FILE ]]; then LOCK_FILE=$OPT_LOCK_FILE; fi
if [[ -n $OPT_SPLIT_MAP_FILE ]]; then SPLIT_MAP_FILE=$OPT_SPLIT_MAP_FILE; fi
if [[ -n $OPT_REPO_URL ]]; then REPO_URL=$OPT_REPO_URL; fi
validate_settings

need() { [[ $1 =~ $2 ]] || die "$3 '$1' is not valid"; }
need "$LOCATION" '^[a-z0-9-]+$' "--location"
need "$SERVER_TYPE" '^[a-z0-9-]+$' "--server-type"
need "$OS_IMAGE" '^[a-z0-9._-]+$' "--os-image"
need "$VOLUME_GB" '^[0-9]+$' "--volume-size"
((VOLUME_GB >= 10)) || die "--volume-size must be at least 10 (GB)"

if [[ -z $NAME ]]; then
  NAME=$(printf 'marinedata-%s-%s' "$RELEASE_ID" "$FLAVOUR" | tr '[:upper:]' '[:lower:]' | tr '._' '--')
fi
need "$NAME" '^[a-z0-9][a-z0-9-]*$' "--name"
VOL_NAME=$NAME-data
if [[ -z $OUT_DIR ]]; then OUT_DIR=${TMPDIR:-/tmp}/marinedata-release-vm/$NAME; fi
USER_DATA=$OUT_DIR/user-data.sh

preflight_run() {
  [[ -n ${HCLOUD_TOKEN:-} ]] || die "--run needs HCLOUD_TOKEN in the environment (a fresh project token, revoked after the build)"
  [[ -n $SSH_KEY ]] || die "--run needs --ssh-key NAME (see: hcloud ssh-key list)"
  command -v hcloud >/dev/null 2>&1 || die "--run needs the hcloud CLI on PATH"
}

confirm_run() {
  if ((YES)); then
    return 0
  fi
  [[ -t 0 ]] || die "--run needs a terminal for the confirmation (or pass --yes)"
  local reply
  read -r -p "Create server $NAME and volume $VOL_NAME ($VOLUME_GB GB, $LOCATION) now? Type the server name to confirm: " reply
  [[ $reply == "$NAME" ]] || die "not confirmed; nothing was created"
}

# User-data holds settings only: it is stored in plain text, so no secret goes in it.
render_user_data() {
  mkdir -p "$OUT_DIR"
  {
    printf '#!/usr/bin/env bash\n'
    printf '# Rendered by create.sh: settings only. No secret goes in user-data (it is stored in plain text).\n'
    printf 'export MARINEDATA_REF=%q\n' "$MARINEDATA_REF"
    printf 'export RELEASE_ID=%q\n' "$RELEASE_ID"
    printf 'export FLAVOUR=%q\n' "$FLAVOUR"
    if [[ -n $OPT_LOCK_FILE ]]; then printf 'export LOCK_FILE=%q\n' "$LOCK_FILE"; fi
    if [[ -n $OPT_SPLIT_MAP_FILE ]]; then printf 'export SPLIT_MAP_FILE=%q\n' "$SPLIT_MAP_FILE"; fi
    if [[ -n $OPT_REPO_URL ]]; then printf 'export REPO_URL=%q\n' "$REPO_URL"; fi
    if [[ -n $RELEASE_BUILD_ARGS ]]; then printf 'export RELEASE_BUILD_ARGS=%q\n' "$RELEASE_BUILD_ARGS"; fi
    if [[ -n $EXPORT_ARGS ]]; then printf 'export EXPORT_ARGS=%q\n' "$EXPORT_ARGS"; fi
    tail -n +2 "$HERE/bootstrap.sh"
  } >"$USER_DATA"
  chmod 600 "$USER_DATA"
  SIZE=$(wc -c <"$USER_DATA")
  ((SIZE <= 32768)) || die "user-data is $SIZE bytes; Hetzner accepts at most 32768"
}

show() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

print_plan() {
  printf '\n# 1. create the volume, then the server. The volume is attached at boot; bootstrap.sh mounts it at /data.\n'
  if [[ -z $SSH_KEY ]]; then
    printf '# replace YOUR_SSH_KEY with a key name from: hcloud ssh-key list\n'
  fi
  show "${vol_cmd[@]}"
  show "${srv_cmd[@]}"
  printf '\n# 2. copy the S3 secrets file (README step 4). It never goes in user-data.\n'
  printf '  IP=$(hcloud server ip %s)\n' "$NAME"
  printf '  scp ~/secure/marinedata.env root@$IP:/root/.marinedata.env.tmp\n'
  printf "  ssh root@\$IP 'chmod 600 /root/.marinedata.env.tmp && mv /root/.marinedata.env.tmp /root/marinedata.env'\n"
  printf '\n# 3. watch the build\n'
  printf "  ssh root@\$IP 'tail -f /data/logs/bootstrap.log'\n"
  printf '\n# 4. teardown, only after the copy-back (README step 8). It destroys the build output.\n'
  show hcloud server delete "$NAME"
  show hcloud volume delete "$VOL_NAME"
}

if [[ $MODE == run ]]; then
  preflight_run
fi
render_user_data

vol_cmd=(hcloud volume create --name "$VOL_NAME" --size "$VOLUME_GB" --location "$LOCATION" --format ext4)
srv_cmd=(hcloud server create --name "$NAME" --type "$SERVER_TYPE" --image "$OS_IMAGE"
  --location "$LOCATION" --ssh-key "${SSH_KEY:-YOUR_SSH_KEY}" --user-data-from-file "$USER_DATA"
  --volume "$VOL_NAME")

if [[ $MODE == run ]]; then
  printf 'create.sh: RUN. These commands are executed after you confirm.\n'
  print_plan
  confirm_run
  "${vol_cmd[@]}"
  "${srv_cmd[@]}"
  printf '\ncreate.sh: created %s and %s. Continue at README step 4 (the secrets file).\n' "$NAME" "$VOL_NAME"
else
  printf 'create.sh: PRINT ONLY. Nothing was created; no token or network was used.\n'
  printf 'create.sh: wrote %s (%s bytes, settings only). Read it, then run the commands below or add --run.\n' "$USER_DATA" "$SIZE"
  print_plan
fi
