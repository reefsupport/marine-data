# WP-6c: server-side ingest Job

Runs `marinedata ingest-batch <specs_dir>` on a Hetzner node next to `<open-bucket>`
(hel1), instead of over the Mac's ~10 MB/s uplink. Everything here is for **you** to run —
no agent touches the cluster, pushes an image, or reads a secret value.

## What this touches

- Writes only to `s3://<open-bucket>/sources/<id>/<version>/…` for the specs you give it.
- Never deletes a bucket object (that's `delete_step.py`, run by you separately).
- The Job's own footprint is one `emptyDir` (`/work`, cleared when the pod is removed).

## 0. Prerequisites (one-time)

1. Populate `registry/ingest-specs/*.yaml` at the repo root with the ingest specs you want
   this run to cover (`docs/ingest-howto.md` — one file per source, same format as a local
   `ingest-source` run). The `kustomization.yaml` here turns that directory into a ConfigMap;
   an empty directory is a build error on purpose.
2. Create the S3 credentials Secret once (values from the `[rs-hel1]` section of
   `~/.config/rclone/rclone.conf` — this README never prints them):
   ```bash
   kubectl -n rs-prod-jobs create secret generic marinedata-ingest-s3-creds \
     --from-literal=access_key_id=<paste> \
     --from-literal=secret_access_key=<paste>
   ```

## 1. Build and push the image

```bash
cd ~/dev/.wt/marine-data/<this-worktree>
docker build -t <registry>/<path>/marinedata-ingest:$(git rev-parse --short HEAD) \
  -f deploy/ingest-job/Dockerfile .
docker push <registry>/<path>/marinedata-ingest:$(git rev-parse --short HEAD)
```

`<registry>/<path>` — use whatever this cluster already pulls from with the `gitlab-registry`
imagePullSecret (matching `rs-backend-jobs`' convention), or push to GHCR and add a matching
`imagePullSecrets` entry to `job.yaml` if you go that route instead.

## 2. Apply

```bash
cd deploy/ingest-job
kustomize edit set image marinedata-ingest=<registry>/<path>/marinedata-ingest:<tag>
kubectl apply -k .
```

Re-running `ingest-batch <specs_dir>` from step 0.1 with new/changed specs: update
`registry/ingest-specs/`, then `kubectl apply -k .` again — Jobs are immutable, so if a Job
with this name already exists and finished, delete it first (`kubectl -n rs-prod-jobs delete
job marinedata-ingest-batch`) or bump `metadata.annotations["marinedata.reef.support/run"]`
and change `metadata.name` to avoid clobbering a Job you still want the logs from.

## 3. Watch progress

```bash
kubectl -n rs-prod-jobs logs -f job/marinedata-ingest-batch
```

Each line is one JSON object: a per-source result (`status`: `ok` | `skip-done` |
`needs-yohan` | `error`) as it finishes, then one final summary line with `total`, `ok`,
`skip_done`, and the full `needs_yohan` / `errors` lists.

## 4. Resume after a restart, node drain, or OOM

Nothing to do — re-apply the same Job (or a fresh one with new specs). `ingest-batch` skips
any source whose `CHECKSUMS.sha256` marker already exists at its resolved
`sources/<id>/<version>/` prefix, and `ingest-source` itself resumes a half-uploaded file
(multipart checkpoint) or half-downloaded source (per-file S3 skip) — see
`docs/design/ingestion.md`. A `NEEDS-YOHAN` source (gated/login-required upstream) is
reported and skipped on every re-run until its spec is fixed or removed.

## 5. Stop

```bash
kubectl -n rs-prod-jobs delete job marinedata-ingest-batch
```

Safe at any point: in-progress multipart uploads are abandoned (not completed, so they're
never visible as finished objects) and re-running from step 2 resumes from whatever last
reached S3.
