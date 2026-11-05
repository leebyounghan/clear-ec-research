# Citations and data obligations for the submission-2 hex6 correction

## External training data (PI approval 2026-10-30)

- **srep_topcon_sp3000** — Daniel, M. C., Atzrodt, L., Bucher, F., Wacker, K., Böhringer, S., Reinhard, T. & Böhringer, D.
  "Automated segmentation of the corneal endothelium in a large set of 'real-world' specular microscopy images
  using the U-Net architecture." *Scientific Reports* 9, 4752 (2019). PMID 30894636; publisher link
  https://rdcu.be/brYSJ (as given in the corpus README); DOI 10.1038/s41598-019-41034-2 (verify before
  publication). Licence (README.upstream.md): "You may freely use these data for your research. However, you
  are obliged to cite our article if you publish anything that is based on this dataset." Used for detector
  pre-training (cell-centre points of 307 of the 385 images after dropping duplicates and images with fewer
  than 10 points); never for the regressor.
- **rodrep_confocal** (secondary, not yet used) — Rotterdam Ophthalmic Data Repository, confocal corneal
  endothelium data set (Confoscan 4), http://rod-rep.com; data owned by the Rotterdam Ophthalmic Institute;
  cite the repository and Vigueras-Guillén et al. if used.

## Models

- CellposeDINO ViT-B encoder (Cellpose, revision 7c61431b5fbb078f3296754bd15d9f51b320f837) as recorded in
  experiments/phase2_foundation_preflight_20261002.py; recipe A regression encoders of this repository.

## Annotated CLEAR-EC images

- 25 expert-annotated images (cell-centre markers and ROI boxes) from the challenge organisers' green
  reference set; 20 used for fine-tuning, 5 held out for localisation F1.
