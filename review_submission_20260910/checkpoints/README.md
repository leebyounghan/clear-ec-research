# Checkpoint distribution - release gate

## Recommended distribution

Keep code in the private GitHub review repository. For this deadline, use a
dedicated Google Drive/OneDrive review folder for the **unchanged model archive**
and optional Docker archive. This avoids an additional Hugging Face account/access
workflow for the organizer. For long-term machine-downloadable versioning, Hugging
Face is a good second option once publication rights and reviewer identities are settled.

### Option A: review-only Drive folder (recommended now)

Create a dedicated `Lamda_CLEAR-EC_PhaseI_20260910` folder, containing only:

```text
model_full_quality77_20260909.tar.gz       # required for venv and Docker
clear_ec_full_quality77_20260909.tar.gz    # optional for venv; required for archived Docker
SHA256SUMS.txt
```

Upload the original archives and checksum file, without unpacking/recompressing them.
Keep challenge images, labels, access tokens, training caches and unrelated files out.
Give the organizer `Peixian.Liang@Pennmedicine.upenn.edu` Viewer/download access in
advance and ensure any other designated evaluators also have access. A single-person
grant is not proof that the entire evaluation team can access the files.
Institutional external-sharing policies or visitor-account support may affect access.

If evaluators cannot be pre-authorized individually, an **Anyone with the link / Viewer**
setting avoids further permission requests only if you approve that broader exposure
and the model/data terms permit it. Such a link is a bearer-access link, not private
to the named organizer. Do not select it automatically. Test downloads as a recipient;
for an anyone-with-link setting, also test while signed out. A folder merely being
visible does not prove that large-file downloads work.

Put the final folder link and the two individual file links here and in the root
README. Keep download rights enabled and no expiry before technical review finishes.
Browser download is acceptable for this option: save the archive under `artifacts/`
and use the README's model-only checksum/extraction commands. Do not invent a direct
download URL by treating a Drive share page as a raw binary endpoint.

[Google Drive sharing and roles](https://support.google.com/drive/answer/2494822?hl=en-IN).

### Option B: versioned Hugging Face model repository

Suggested name: `<your-HF-account-or-organization>/lamda-clear-ec-phase1` (not created).
Start private. A private personal repository does not inherit GitHub collaborator
permissions; for private reviewer access, use an organization with pre-authorized
read members. The GitHub name `pxliang` is not evidence of an identical HF account.
Do not put a personal read/write token in GitHub, README or the submission email.

In a **separate upload-tool environment**, install `huggingface_hub`, authenticate
interactively with `hf auth login`, and create a private model repository in the HF UI.
Then upload the exact archive (replace the example namespace with the real one):

```bash
hf upload YOUR_NAMESPACE/lamda-clear-ec-phase1 \
  /absolute/path/model_full_quality77_20260909.tar.gz \
  model_full_quality77_20260909.tar.gz
hf upload YOUR_NAMESPACE/lamda-clear-ec-phase1 \
  /absolute/path/clear_ec_full_quality77_20260909.tar.gz \
  clear_ec_full_quality77_20260909.tar.gz
hf upload YOUR_NAMESPACE/lamda-clear-ec-phase1 \
  /absolute/path/SHA256SUMS.txt SHA256SUMS.txt
```

Record the final immutable commit hash after all files are present. Reviewers use
their own pre-authorized account (or no login only after approved public release):

```bash
hf download YOUR_NAMESPACE/lamda-clear-ec-phase1 \
  model_full_quality77_20260909.tar.gz \
  --revision FULL_COMMIT_HASH --local-dir artifacts
```

Verify the release SHA256 after download. Do not use a moving `main` as the sole
submission identity. Do not enable request-based gating for reviewers who have
not already been granted access. Review license/competition publication conditions
before changing visibility to public. Confirm available storage under your account.

[HF upload/download CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
[organization read permissions](https://huggingface.co/docs/hub/organizations-security),
[storage limits](https://huggingface.co/docs/hub/storage-limits).

### Why not put the archive directly in GitHub?

The 7.96 GB model archive exceeds even GitHub LFS's documented 5 GB maximum
per-file limit, and GitHub release assets must each be under 2 GiB. Individual
checkpoints could be stored using LFS subject to the plan's quotas, but external
hosting keeps the current verified archive intact and avoids extra splitting and
reassembly steps. [LFS limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
[release asset limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Exact artifacts and current publication status

The image and model remain separate. Do not add model weights to the Docker image
or replace them with later experimental weights.

| Resource | Bytes | SHA256 |
|---|---:|---|
| clear_ec_full_quality77_20260909.tar.gz | 4,064,075,652 | cfa6c2c8a234a5aefc8cf0ff0a3413bbaf7a353fc3e131cd7cddf33434896c8a |
| model_full_quality77_20260909.tar.gz | 7,964,629,097 | 51d8074a5fa45a71139bd6dd12c768dea7246c43c7002f8456777984396d0830 |

Download URLs: **not yet published**. This is a blocking release item, not an
instruction to request access from the team. Before submitting the repository,
provide accessible links and test that the organizer can retrieve the archives
without a further permission request. Public redistribution is not assumed.

`provenance/model_archive.json` lists every extracted model member, byte size and
SHA256, including the required Cellpose-SAM foundation checkpoint `cpsam_v2`.
That foundation checkpoint is included because model construction still uses it
before loading each fitted state. It is not an optional internet download.

Upstream provenance: https://huggingface.co/mouseland/cellpose-sam (cpsam_v2).
An upstream floating revision is not a replacement for the exact hash in this release.
Confirm the upstream code/weight redistribution terms and include the applicable
license notices before publication. No new license grant for third-party assets
is implied here.

Local preparation note: the verified archives are preserved in the sibling
`submission_candidate_20260909/artifacts/` directory of the research checkout.
They have not been duplicated into this review directory or uploaded by this task.
Replace this local preparation note with finalized hosting instructions for reviewers.
