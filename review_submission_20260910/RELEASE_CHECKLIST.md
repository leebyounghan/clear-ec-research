# Phase I release checklist

Status: **organized local draft; not ready to email as a complete technical submission**.
Organizer deadline: September 16; the supplied email does not specify a timezone.

## Completed in this local organization

- [x] Isolated the September 10 candidate from later research.
- [x] Preserved original inference sources, model configuration and Dockerfile.
- [x] Included original archive identities and individual checkpoint hashes.
- [x] Retrieved historical all-data trainers/configurations; kept early and regional source versions separate.
- [x] Documented exact inference interface, command, hardware evidence and limitations.
- [x] Produced a technical report without challenge images or per-case labels.
- [x] Added isolated Linux/Python 3.12 venv setup and single-MHA convenience execution;
      dependency resolution and eight synthetic wrapper tests pass.
- [x] Fresh Kubernetes CPU venv installation, pip check, imports and eight model
      resource SHA256 checks passed (135-second successful verification container).

## Required before dispatch

- [ ] Confirm registered team name and team members; proposed display name is Lamda.
- [ ] Record the successful scored evaluation URL, image ID and model ID.
      The previously pasted duplicate upload ID 482c14b4-19f9-4fb4-a28c-a2e39652eb67
      was Import Failed; it is not proof of the successful scoring pair.
- [ ] Finalize private review repository URL, commit and release tag.
- [ ] Publish exact archives with accessible URLs; verify reviewer download and SHA256.
- [ ] Check data and third-party code/weight terms; add required notices.
- [ ] Test inference from this clean checkout with the original image and model archives,
      preferably on a T4, with no network and documented output comparison tolerance.
- [ ] Verify full clean-venv CUDA inference against the archived-image outputs;
      installation/import checks alone do not close this gate.
- [ ] Close the training-environment lock/data-bootstrap/OOF-selection reproduction
      gaps described in TRAINING.md. Do not claim complete retraining reproduction yet.
- [ ] Remove local-only notes and unresolved draft fields after evidence is attached.
- [ ] Add GitHub user pxliang to the private repository; verify actual access.
- [ ] Email repository link to Peixian.Liang@Pennmedicine.upenn.edu with subject
      `[Challenge Phase I Submission] <confirmed team name>` before September 16.

Creating a private repository, changing sharing, inviting a collaborator, uploading
large assets and sending the email have not been performed by this task.
