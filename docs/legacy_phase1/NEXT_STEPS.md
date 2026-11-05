# Phase I technical-review checklist

This is an administrative/reproducibility checklist for the September 14 release,
not a training roadmap. Deadline in the organizer email: September 16
(timezone not specified).

## Prepared

- [x] September 14 model and unchanged Docker archive identified by SHA256.
- [x] Model/Docker files uploaded by the team to the linked Google Drive folder.
- [x] Viewer download controls checked in the browser.
- [x] Complete anonymous Drive downloads and archive/resource hashes verified on a separate Linux host.
- [x] Fresh GitHub checkout and unchanged Docker runner passed one-image RTX 4090 inference.
- [x] New isolated venv installation and direct source inference passed; all three outputs exactly matched Docker.
- [x] Docker-only inference guide and exact-resource runner prepared.
- [x] Technical report revised to the September 14 model with development appendix.

## Confirm before sending

- [ ] Confirm final team identity/member information.
- [x] Verify the 8.7010 evaluation's model/image versions: model `b4137c8d`, exact CRC64-NVME match to Drive; image `730ff3f7`, full SHA256 match.
- [x] Publish the report/documentation baseline: commit `d80342d`.
- [ ] Verify GitHub access for the organizer's requested account `pxliang`.
- [ ] Confirm the evaluator can access the shared links from their environment.
- [ ] Confirm applicable third-party code/weight notices.

The [reproduction scope](REPRODUCE.md) records remaining runtime and clean-training
verification limits; do not mark those checks passed from documentation alone.

Send the repository, artifact and report links to
`Peixian.Liang@Pennmedicine.upenn.edu`, using
`[Challenge Phase I Submission] Lamda&KIM` if that is the confirmed team name.
Repository access must be verified before the email is sent. No invitation,
permission change or email is performed by this documentation cleanup.

Latest September 15 access check: invitation for `pxliang` exists, but the account
still has no repository permission pending acceptance. Verify access before sending.

[Detailed historical record](../archive/snapshot_20260915/docs/NEXT_STEPS.md) — retained for provenance, not reviewer setup.
