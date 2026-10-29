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
#
# --target pve renders the Proxmox build VM instead: out/vm-VMID.sh (the qm sequence, run by the
# owner as root on the host) and out/NAME.yaml (the cloud-init vendor snippet, settings only). It
# contacts nothing, refuses --run and a --ref that is not a full 40-hex commit sha. The operator's
# own host, user, VM id, addresses and key comments are not in this file: they come from flags,
# PVE_* environment variables or --env-file (pve.env.example), and an unset one is refused.

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
  --target hetzner|pve   hetzner (default): the hcloud commands. pve: render the Proxmox files
                         (out/vm-VMID.sh and out/NAME.yaml, never run; --ref must be 40 hex)

Proxmox options (--target pve only). Each value is the flag, else the environment variable of the
same name, else the line in --env-file PATH (KEY=VALUE lines, read and never sourced; copy
pve.env.example to the git-ignored pve.env). The first group names your own host, user, VM and keys:
it has no default and create.sh refuses to render while one is unset.
  --host H               PVE_HOST                 Proxmox host: the ssh jump host in the printed commands
  --ci-user U            PVE_CI_USER              cloud-init login user of the VM
  --sshkeys-from-vmid N  PVE_SSHKEYS_FROM_VMID    the VM whose cloud-init sshkeys the new VM gets
  --key-comments "A, B"  PVE_KEY_COMMENTS         the comments those keys must carry (checked by eye)
  --ip CIDR              PVE_STATIC_IP            address of the VM, a.b.c.d/nn
  --gw ADDR              PVE_GATEWAY              --bridge B    PVE_BRIDGE    --vmid N    PVE_VMID
  --env-file PATH
With a generic default:
  --name NAME            PVE_NAME         marinedata-build     --nameserver ADDR  PVE_NAMESERVER  1.1.1.1
  --storage S            PVE_STORAGE      local-lvm            --cores N          PVE_CORES       6
  --memory MB            PVE_MEMORY       16384                --data-gb GB       PVE_DATA_GB     100
  --image local:iso/F    PVE_IMAGE        local:iso/debian-12-genericcloud-amd64.qcow2
  --keys-file PATH       PVE_KEYS_FILE    /root/vm<VMID>-keys.pub (on the host)
EOF
}

RELEASE_ID="" FLAVOUR="" REF="" SSH_KEY="" NAME="" OUT_DIR=""
LOCATION=hel1 SERVER_TYPE=cx32 VOLUME_GB=200 OS_IMAGE=ubuntu-24.04
OPT_LOCK_FILE="" OPT_SPLIT_MAP_FILE="" OPT_REPO_URL=""
MODE=print YES=0
TARGET=hetzner PVE_SEEN="" PVE_ENV_FILE=""
# --target pve values: a flag wins over the environment variable of the same name, which wins over
# --env-file. PVE_REQUIRED names the operator's own host, user, VM and keys: no default exists.
PVE_REQUIRED="PVE_HOST PVE_CI_USER PVE_SSHKEYS_FROM_VMID PVE_KEY_COMMENTS PVE_STATIC_IP PVE_GATEWAY PVE_BRIDGE PVE_VMID"
PVE_OPTIONAL="PVE_NAME PVE_NAMESERVER PVE_STORAGE PVE_IMAGE PVE_CORES PVE_MEMORY PVE_DATA_GB PVE_KEYS_FILE"
PVE_HOST=${PVE_HOST:-} PVE_CI_USER=${PVE_CI_USER:-} PVE_SSHKEYS_FROM_VMID=${PVE_SSHKEYS_FROM_VMID:-}
PVE_KEY_COMMENTS=${PVE_KEY_COMMENTS:-} PVE_STATIC_IP=${PVE_STATIC_IP:-} PVE_GATEWAY=${PVE_GATEWAY:-}
PVE_BRIDGE=${PVE_BRIDGE:-} PVE_VMID=${PVE_VMID:-} PVE_NAME=${PVE_NAME:-} PVE_NAMESERVER=${PVE_NAMESERVER:-}
PVE_STORAGE=${PVE_STORAGE:-} PVE_IMAGE=${PVE_IMAGE:-} PVE_CORES=${PVE_CORES:-} PVE_MEMORY=${PVE_MEMORY:-}
PVE_DATA_GB=${PVE_DATA_GB:-} PVE_KEYS_FILE=${PVE_KEYS_FILE:-}

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
    --target) TARGET=${2:?--target needs a value}; shift 2 ;;
    --vmid) PVE_SEEN="$PVE_SEEN $1"; PVE_VMID=${2:?--vmid needs a value}; shift 2 ;;
    --ip) PVE_SEEN="$PVE_SEEN $1"; PVE_STATIC_IP=${2:?--ip needs a value}; shift 2 ;;
    --gw) PVE_SEEN="$PVE_SEEN $1"; PVE_GATEWAY=${2:?--gw needs a value}; shift 2 ;;
    --bridge) PVE_SEEN="$PVE_SEEN $1"; PVE_BRIDGE=${2:?--bridge needs a value}; shift 2 ;;
    --storage) PVE_SEEN="$PVE_SEEN $1"; PVE_STORAGE=${2:?--storage needs a value}; shift 2 ;;
    --image) PVE_SEEN="$PVE_SEEN $1"; PVE_IMAGE=${2:?--image needs a value}; shift 2 ;;
    --cores) PVE_SEEN="$PVE_SEEN $1"; PVE_CORES=${2:?--cores needs a value}; shift 2 ;;
    --memory) PVE_SEEN="$PVE_SEEN $1"; PVE_MEMORY=${2:?--memory needs a value}; shift 2 ;;
    --data-gb) PVE_SEEN="$PVE_SEEN $1"; PVE_DATA_GB=${2:?--data-gb needs a value}; shift 2 ;;
    --keys-file) PVE_SEEN="$PVE_SEEN $1"; PVE_KEYS_FILE=${2:?--keys-file needs a value}; shift 2 ;;
    --host) PVE_SEEN="$PVE_SEEN $1"; PVE_HOST=${2:?--host needs a value}; shift 2 ;;
    --ci-user) PVE_SEEN="$PVE_SEEN $1"; PVE_CI_USER=${2:?--ci-user needs a value}; shift 2 ;;
    --sshkeys-from-vmid) PVE_SEEN="$PVE_SEEN $1"; PVE_SSHKEYS_FROM_VMID=${2:?--sshkeys-from-vmid needs a value}; shift 2 ;;
    --key-comments) PVE_SEEN="$PVE_SEEN $1"; PVE_KEY_COMMENTS=${2:?--key-comments needs a value}; shift 2 ;;
    --nameserver) PVE_SEEN="$PVE_SEEN $1"; PVE_NAMESERVER=${2:?--nameserver needs a value}; shift 2 ;;
    --env-file) PVE_SEEN="$PVE_SEEN $1"; PVE_ENV_FILE=${2:?--env-file needs a value}; shift 2 ;;
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

# ---- --target pve: render the host script and the cloud-init vendor snippet; nothing is run ----
# Both files come from the templates below, which are the build plan's vm-900.sh (section 3.2) and
# vendor snippet (section 3.3) with the values as @TOKENS@. Nothing that names the operator's own
# host, user, VM, addresses or keys lives in this file: those values come from a flag, else the
# environment variable of the same name, else --env-file, and an unset one stops the render.

PVE_DEFAULT_IMAGE=local:iso/debian-12-genericcloud-amd64.qcow2
PVE_KNOWN_KEYS=" $PVE_REQUIRED $PVE_OPTIONAL "

ipv4_ok() {
  local octets o
  [[ $1 =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
  IFS=. read -r -a octets <<<"$1"
  for o in "${octets[@]}"; do
    ((10#$o <= 255)) || return 1
  done
}

# The flag that sets a PVE_* variable: --ci-user for PVE_CI_USER, and so on.
pve_flag_name() {
  case $1 in
    PVE_STATIC_IP) printf -- '--ip' ;;
    PVE_GATEWAY) printf -- '--gw' ;;
    *) printf -- '--%s' "$(printf '%s' "${1#PVE_}" | tr 'A-Z_' 'a-z-')" ;;
  esac
}

# --env-file: KEY=VALUE lines (# comments, blank lines, optional quotes around the value). The file is
# read, never sourced, and only the PVE_* names above are accepted. A value already set by a flag or
# by the environment wins over the file.
pve_read_env_file() {
  local file=$1 line key value n=0
  local blank='^[[:space:]]*(#.*)?$' pair='^[[:space:]]*([A-Z][A-Z0-9_]*)=(.*)$' dq='^"(.*)"$' sq="^'(.*)'\$"
  [[ -f $file && -r $file ]] || die "--env-file '$file' is not a readable file"
  while IFS= read -r line || [[ -n $line ]]; do
    n=$((n + 1))
    line=${line%$'\r'}
    if [[ $line =~ $blank ]]; then continue; fi
    [[ $line =~ $pair ]] || die "$file:$n: expected KEY=VALUE or a # comment"
    key=${BASH_REMATCH[1]}
    value=${BASH_REMATCH[2]}
    [[ $PVE_KNOWN_KEYS == *" $key "* ]] || die "$file:$n: unknown key $key (known:${PVE_KNOWN_KEYS% })"
    if [[ $value =~ $dq || $value =~ $sq ]]; then value=${BASH_REMATCH[1]}; fi
    if [[ -z ${!key} ]]; then printf -v "$key" '%s' "$value"; fi
  done <"$file"
}

# Flag > environment variable > --env-file > the generic default below. The PVE_REQUIRED values name
# the operator's own host, user, VM, addresses and keys: none has a default, and a missing one stops here.
pve_resolve() {
  local key missing=""
  if [[ -n $NAME ]]; then PVE_NAME=$NAME; fi
  if [[ -n $PVE_ENV_FILE ]]; then pve_read_env_file "$PVE_ENV_FILE"; fi
  for key in $PVE_REQUIRED; do
    if [[ -z ${!key} ]]; then missing="$missing $key ($(pve_flag_name "$key"))"; fi
  done
  [[ -z $missing ]] ||
    die "--target pve has no value for:$missing. Pass each as its flag or environment variable, or list them in --env-file PATH (deploy/release-vm/pve.env.example). Nothing is guessed."
  PVE_NAME=${PVE_NAME:-marinedata-build}
  PVE_NAMESERVER=${PVE_NAMESERVER:-1.1.1.1}
  PVE_STORAGE=${PVE_STORAGE:-local-lvm}
  PVE_IMAGE=${PVE_IMAGE:-$PVE_DEFAULT_IMAGE}
  PVE_CORES=${PVE_CORES:-6}
  PVE_MEMORY=${PVE_MEMORY:-16384}
  PVE_DATA_GB=${PVE_DATA_GB:-100}
  PVE_KEYS_FILE=${PVE_KEYS_FILE:-/root/vm${PVE_VMID}-keys.pub}
}

pve_validate() {
  [[ $MODE != run ]] || die "--run is not supported with --target pve: the host commands are run by the owner (README, Proxmox section)"
  [[ -z $SSH_KEY && $LOCATION == hel1 && $SERVER_TYPE == cx32 && $VOLUME_GB == 200 && $OS_IMAGE == ubuntu-24.04 && $YES == 0 ]] ||
    die "--ssh-key, --location, --server-type, --volume-size, --os-image and --yes belong to --target hetzner"
  [[ $REF =~ ^[0-9a-f]{40}$ ]] ||
    die "--target pve needs --ref as a full 40-character commit sha (got '$REF'); a tag, a branch or a short sha can move"
  pve_resolve
  [[ $REPO_URL =~ ^https://[A-Za-z0-9._/-]+$ ]] || die "--repo-url '$REPO_URL' must be a plain https URL without credentials"
  if [[ -n $OPT_LOCK_FILE ]]; then need "$LOCK_FILE" '^[A-Za-z0-9][A-Za-z0-9._/-]*$' "--lock-file"; fi
  if [[ -n $OPT_SPLIT_MAP_FILE ]]; then need "$SPLIT_MAP_FILE" '^[A-Za-z0-9][A-Za-z0-9._/-]*$' "--split-map-file"; fi
  need "$RELEASE_BUILD_ARGS" '^[A-Za-z0-9 ._/=:,+-]*$' "--build-args"
  need "$EXPORT_ARGS" '^[A-Za-z0-9 ._/=:,+-]*$' "--export-args"
  need "$PVE_NAME" '^[a-z0-9][a-z0-9-]*$' "PVE_NAME"
  need "$PVE_HOST" '^[A-Za-z0-9][A-Za-z0-9._-]*$' "PVE_HOST"
  need "$PVE_CI_USER" '^[a-z_][a-z0-9_-]{0,31}$' "PVE_CI_USER"
  need "$PVE_SSHKEYS_FROM_VMID" '^[0-9]+$' "PVE_SSHKEYS_FROM_VMID"
  need "$PVE_KEY_COMMENTS" '^[A-Za-z0-9@._-]+(, [A-Za-z0-9@._-]+)*$' "PVE_KEY_COMMENTS (comma and space between comments)"
  PVE_KEY_COUNT=$(printf '%s\n' "$PVE_KEY_COMMENTS" | awk -F', ' '{ print NF }')
  need "$PVE_VMID" '^[0-9]+$' "PVE_VMID"
  ((10#$PVE_VMID >= 100)) || die "PVE_VMID must be at least 100"
  need "$PVE_STATIC_IP" '^[0-9.]+/[0-9]+$' "PVE_STATIC_IP"
  PVE_ADDR=${PVE_STATIC_IP%/*}
  { ipv4_ok "$PVE_ADDR" && ((10#${PVE_STATIC_IP#*/} <= 32)); } || die "PVE_STATIC_IP '$PVE_STATIC_IP' is not an a.b.c.d/nn address"
  ipv4_ok "$PVE_GATEWAY" || die "PVE_GATEWAY '$PVE_GATEWAY' is not an IPv4 address"
  ipv4_ok "$PVE_NAMESERVER" || die "PVE_NAMESERVER '$PVE_NAMESERVER' is not an IPv4 address"
  need "$PVE_BRIDGE" '^[A-Za-z0-9_]+$' "PVE_BRIDGE"
  need "$PVE_STORAGE" '^[A-Za-z0-9][A-Za-z0-9._-]*$' "PVE_STORAGE"
  [[ $PVE_IMAGE =~ ^local:iso/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "PVE_IMAGE '$PVE_IMAGE' must be local:iso/FILE (the 'local' directory storage, /var/lib/vz)"
  PVE_IMG_PATH=/var/lib/vz/template/iso/${PVE_IMAGE#local:iso/}
  need "$PVE_CORES" '^[0-9]+$' "PVE_CORES"
  ((10#$PVE_CORES >= 1 && 10#$PVE_CORES <= 128)) || die "PVE_CORES must be between 1 and 128"
  need "$PVE_MEMORY" '^[0-9]+$' "PVE_MEMORY"
  ((10#$PVE_MEMORY >= 1024)) || die "PVE_MEMORY must be at least 1024 (MB)"
  need "$PVE_DATA_GB" '^[0-9]+$' "PVE_DATA_GB"
  ((10#$PVE_DATA_GB >= 10)) || die "PVE_DATA_GB must be at least 10 (GB)"
  need "$PVE_KEYS_FILE" '^/[A-Za-z0-9._/-]+$' "PVE_KEYS_FILE"
}

# /etc/marinedata/settings.env: settings only, no credential of any kind.
pve_settings() {
  printf 'MARINEDATA_REF=%s\n' "$MARINEDATA_REF"
  printf 'RELEASE_ID=%s\n' "$RELEASE_ID"
  printf 'FLAVOUR=%s\n' "$FLAVOUR"
  if [[ -n $OPT_LOCK_FILE ]]; then printf 'LOCK_FILE=%s\n' "$LOCK_FILE"; fi
  if [[ -n $OPT_SPLIT_MAP_FILE ]]; then printf 'SPLIT_MAP_FILE=%s\n' "$SPLIT_MAP_FILE"; fi
  if [[ -n $RELEASE_BUILD_ARGS ]]; then printf 'RELEASE_BUILD_ARGS=%q\n' "$RELEASE_BUILD_ARGS"; fi
  if [[ -n $EXPORT_ARGS ]]; then printf 'EXPORT_ARGS=%q\n' "$EXPORT_ARGS"; fi
  printf 'DATA_DEV=%s\n' "$PVE_DATA_DEV"
}

pve_tpl_yaml() {
  cat <<'EOF'
#cloud-config
package_update: true
packages: [qemu-guest-agent, git, ca-certificates, curl, jq]
write_files:
  - path: /etc/marinedata/settings.env
    permissions: '0644'
    content: |
@SETTINGS@
runcmd:
  - [systemctl, enable, --now, qemu-guest-agent]
  - [git, clone, --no-checkout, "@REPO_URL@", @REPO_DIR@]
  - [sh, -c, '. /etc/marinedata/settings.env && git -C @REPO_DIR@ checkout --detach "$MARINEDATA_REF"']
EOF
}

pve_tpl_script() {
  cat <<'EOF'
set -euo pipefail
qm status @VMID@ >/dev/null 2>&1 && { echo "vmid @VMID@ exists"; exit 1; }
qm list; pct list
ping -c1 -W1 @IP@ >/dev/null && { echo "@IP@ answers"; exit 1; }
ip neigh show @IP@                                   # expect nothing or FAILED
pvesm status | grep -E '^(local|@STORAGE@) '              # both active
stat -c %s @IMG_PATH@@IMG_NOTE@
printf '%b\n' "$(qm config @SSHKEYS_VMID@ | sed -n 's/^sshkeys: //p' | sed 's/%/\\x/g')" | grep -v '^$' > @KEYS@
ssh-keygen -lf @KEYS@                       # exactly @KEY_COUNT@: @KEY_COMMENTS@
install -m 0644 @NAME@.yaml /var/lib/vz/snippets/@NAME@.yaml
qm create @VMID@ --name @NAME@ \
  --description 'Disposable marine-data @RELEASE_ID@ build. Raw qm, not in OpenTofu by design. Destroy after publish.' \
  --ostype l26 --cpu host --sockets 1 --cores @CORES@ --memory @MEMORY@ --scsihw virtio-scsi-single \
  --agent enabled=1 --serial0 socket --vga serial0 --onboot 0 --net0 virtio,bridge=@BRIDGE@,firewall=0,mtu=1
qm importdisk @VMID@ @IMG_PATH@ @STORAGE@
qm config @VMID@ | grep '^unused0: @STORAGE@:vm-@VMID@-disk-0'  # the name the next line uses
qm set @VMID@ --scsi0 @STORAGE@:vm-@VMID@-disk-0,discard=on,iothread=1,ssd=1,backup=0
qm resize @VMID@ scsi0 32G
qm set @VMID@ --scsi1 @STORAGE@:@DATA_GB@,discard=on,iothread=1,ssd=1,backup=0
qm set @VMID@ --ide2 @STORAGE@:cloudinit --boot order=scsi0
qm set @VMID@ --ciuser @CI_USER@ --sshkeys @KEYS@ \
  --ipconfig0 ip=@CIDR@,gw=@GW@ --nameserver @NAMESERVER@ --searchdomain localdomain
qm set @VMID@ --cicustom vendor=local:snippets/@NAME@.yaml
qm cloudinit dump @VMID@ user | grep -c 'ssh-'               # @KEY_COUNT@
qm start @VMID@
timeout 900 sh -c 'until qm guest cmd @VMID@ ping 2>/dev/null; do sleep 15; done'
qm guest exec @VMID@ --timeout 600 -- cloud-init status --wait
qm guest exec @VMID@ -- sh -c 'ip -br link; ls -l /dev/disk/by-id/; lsblk -o NAME,SIZE,TYPE,PTTYPE'
EOF
}

pve_render_yaml() {
  local t settings
  settings=$(pve_settings | sed 's/^/      /')
  t=$(pve_tpl_yaml)
  t=${t//@SETTINGS@/$settings}
  t=${t//@REPO_URL@/$REPO_URL}
  t=${t//@REPO_DIR@/$REPO_DIR}
  printf '%s\n' "$t"
}

# The shebang, two comment lines and the cd are the only lines added to the plan's block; the cd is
# there because the block installs ./NAME.yaml by a relative path.
pve_render_script() {
  local t note=""
  if [[ $PVE_IMAGE == "$PVE_DEFAULT_IMAGE" ]]; then
    note='   # 349634560'
  fi
  t=$(pve_tpl_script)
  t=${t//@IMG_PATH@/$PVE_IMG_PATH}
  t=${t//@IMG_NOTE@/$note}
  t=${t//@VMID@/$PVE_VMID}
  t=${t//@NAME@/$PVE_NAME}
  t=${t//@CIDR@/$PVE_STATIC_IP}
  t=${t//@IP@/$PVE_ADDR}
  t=${t//@GW@/$PVE_GATEWAY}
  t=${t//@BRIDGE@/$PVE_BRIDGE}
  t=${t//@STORAGE@/$PVE_STORAGE}
  t=${t//@CORES@/$PVE_CORES}
  t=${t//@MEMORY@/$PVE_MEMORY}
  t=${t//@DATA_GB@/$PVE_DATA_GB}
  t=${t//@KEYS@/$PVE_KEYS_FILE}
  t=${t//@CI_USER@/$PVE_CI_USER}
  t=${t//@NAMESERVER@/$PVE_NAMESERVER}
  t=${t//@SSHKEYS_VMID@/$PVE_SSHKEYS_FROM_VMID}
  t=${t//@KEY_COUNT@/$PVE_KEY_COUNT}
  t=${t//@RELEASE_ID@/$RELEASE_ID}  # validated: letters, digits, . _ -
  t=${t//@KEY_COMMENTS@/$PVE_KEY_COMMENTS}  # last: a key comment holds an @
  printf '#!/usr/bin/env bash\n'
  printf '# Rendered by create.sh --target pve. Run as root on the Proxmox host, in the folder that holds\n'
  printf '# %s.yaml: bash vm-%s.sh\n' "$PVE_NAME" "$PVE_VMID"
  # shellcheck disable=SC2016  # literal text for the rendered script
  printf 'cd "$(dirname "$0")" || exit 1\n'
  printf '%s\n' "$t"
}

print_pve() {
  local yaml=$1 script=$2
  printf 'create.sh: PROXMOX TARGET. Nothing was created; no host, token or network was used.\n'
  printf 'create.sh: wrote %s (settings only)\n' "$yaml"
  printf 'create.sh: wrote %s (the qm sequence)\n' "$script"
  printf '\n# 1. Read both files (printed below). The owner copies them into one folder on the Proxmox host (%s) and runs, as root:\n' "$PVE_HOST"
  printf '  bash vm-%s.sh\n' "$PVE_VMID"
  printf '\n# 2. From your own machine, once it has finished (the VM answers at %s):\n' "$PVE_ADDR"
  printf "  ssh -J root@%s %s@%s 'cloud-init status; ping -c1 -M do -s 1372 1.1.1.1'\n" "$PVE_HOST" "$PVE_CI_USER" "$PVE_ADDR"
  printf '\n# 3. In the VM as root (README, Proxmox section): the set-up first, then the stages\n'
  printf '  set -a; . /etc/marinedata/settings.env; set +a; cd %s\n' "$REPO_DIR"
  printf '  STAGES="" bash deploy/release-vm/bootstrap.sh\n'
  printf '\n# 4. Teardown, only after the copy-back (README step 8). It destroys the build output.\n'
  printf '  qm shutdown %s --timeout 120 || qm stop %s\n' "$PVE_VMID" "$PVE_VMID"
  printf '  qm destroy %s --purge 1 --destroy-unreferenced-disks 1\n' "$PVE_VMID"
  printf '  rm -f /var/lib/vz/snippets/%s.yaml %s\n' "$PVE_NAME" "$PVE_KEYS_FILE"
  printf '\n# --- %s ---\n' "$yaml"
  cat "$yaml"
  printf '\n# --- %s ---\n' "$script"
  cat "$script"
}

pve_main() {
  pve_validate
  local out=${OUT_DIR:-$HERE/out}
  local yaml=$out/$PVE_NAME.yaml script=$out/vm-$PVE_VMID.sh
  mkdir -p "$out"
  pve_render_yaml >"$yaml"
  pve_render_script >"$script"
  chmod 755 "$script"
  print_pve "$yaml" "$script"
}

case $TARGET in
  hetzner | pve) ;;
  *) die "--target must be hetzner or pve, got '$TARGET'" ;;
esac
if [[ $TARGET == pve ]]; then
  pve_main
  exit 0
fi
[[ -z $PVE_SEEN ]] || die "${PVE_SEEN# } work with --target pve only"
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
