#!/usr/bin/env bash
# marine-data release VM: publish one flavour by hand (README step 6), one part per call, and stop at the
# first failure. Run it after reading /data/logs/plan.log and GATES.json.
#
# It reads RELEASE_ID, FLAVOUR and LOCK_FILE from /etc/marinedata/settings.env. HF_TOKEN must already be in
# the environment: the VM never stores the token, so there is no file fallback. Launch it detached, so a
# dropped ssh does not stop it, with `read -rs HF_TOKEN` on a line of its own first:
#
#   read -rs HF_TOKEN
#   export HF_TOKEN; nohup setsid bash /opt/marine-data/deploy/release-vm/publish.sh >/dev/null 2>&1 </dev/null & unset HF_TOKEN
#
# Log: /data/logs/publish-$RELEASE_ID-$FLAVOUR.out. Markers, one per line (the log also holds progress bars
# that end in \r, so follow it with: tr '\r' '\n' < LOG | grep '^===' | tail):
#   === PUBLISH START|DONE <utc>         === PUBLISH FAILED rc=<n> at <step> <utc>
#   === PART <part> start|ok <utc>
# Exit: 0 every part uploaded, 1 a step failed, 2 nothing launched (token or settings missing).

set -u

SETTINGS=/etc/marinedata/settings.env
LOG=''

ts() { date -u +%FT%TZ; }
# A marker starts on a line of its own, even when a progress bar left the line unfinished.
mark() { printf '\n=== %s %s\n' "$*" "$(ts)"; }

die() {
  echo "publish.sh: $*" >&2
  # A detached run has stderr on /dev/null: leave the reason in the log as well.
  if [[ -n $LOG ]]; then mark "PUBLISH FAILED rc=2 at preflight: $*" >> "$LOG" 2>/dev/null; fi
  exit 2
}

[[ -r $SETTINGS ]] || die "$SETTINGS is not readable: nothing launched"
# shellcheck disable=SC1090  # the VM's settings file
. "$SETTINGS"
[[ -n ${RELEASE_ID:-} && -n ${FLAVOUR:-} && -n ${LOCK_FILE:-} ]] ||
  die "RELEASE_ID, FLAVOUR and LOCK_FILE must be set in $SETTINGS: nothing launched"

DIR=/data/export/$RELEASE_ID/$FLAVOUR
REL_DIR=/data/releases/$RELEASE_ID/$FLAVOUR
LOCK=/app/$LOCK_FILE
LOG=/data/logs/publish-$RELEASE_ID-$FLAVOUR.out

[[ -n ${HF_TOKEN:-} ]] ||
  die "HF_TOKEN is not in the environment: nothing launched (read -rs HF_TOKEN; export HF_TOKEN; the VM never stores it)"
export HF_TOKEN

ARGS=(release publish --release-id "$RELEASE_ID" --flavour "$FLAVOUR" --dir "$DIR" --release-dir "$REL_DIR" --lock "$LOCK")

{
  mark PUBLISH START
  parts=$(marinedata-run "${ARGS[@]}" --list-parts)
  rc=$?
  if [[ $rc -ne 0 || -z $parts ]]; then
    mark "PUBLISH FAILED rc=$rc at list-parts"
    exit 1
  fi
  echo "parts: ${parts//$'\n'/ }"
  for part in $parts; do
    mark "PART $part start"
    marinedata-run "${ARGS[@]}" --go --part "$part"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      mark "PUBLISH FAILED rc=$rc at part $part"
      exit 1
    fi
    mark "PART $part ok"
  done
  mark PUBLISH DONE
} >> "$LOG" 2>&1
