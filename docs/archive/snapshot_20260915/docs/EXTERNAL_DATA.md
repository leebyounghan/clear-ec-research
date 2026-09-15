> Historical snapshot retained during the September 15 documentation cleanup.
> This is not live status or current execution authorization. See the
> [current document](../../../EXTERNAL_DATA.md) for the release-facing summary.

# External domain-pretraining data

This inventory separates genuinely downloadable image data from papers that
only publish summary measurements or provide images on request.  Raw external
data and generated tiles remain under `datasets/external/` and are never
committed.

## Included

| source | images used | relation to CLEAR-EC | terms | sampling policy |
|---|---:|---|---|---|
| [SREP Topcon SP-3000](https://github.com/daboe01/SREP-18-33533B) | 385 originals; 378 after exact-pHash deduplication | Same corneal-endothelium tissue and non-contact specular modality; centroids included | Free research use; cite Boehringer et al., *Scientific Reports* 2019 | Highest external weight |
| [RODREP Confocal Corneal Endothelium](https://link.springer.com/article/10.1186/s12880-015-0054-3) | 52 | Same tissue and morphometry task, adjacent in-vivo confocal modality; manual points/outlines included | ROI research-only, non-commercial, non-transferable license; do not redistribute; cite Selig et al., *BMC Medical Imaging* 2015 | Lower weight |
| [U-Net_Segmentation-Cornea_Cells](https://github.com/svdeepak99/U-Net_Segmentation-Cornea_Cells) | 160 originals; 157 after exact-pHash deduplication | Corneal-endothelium cell microscopy and masks, but acquisition provenance is not documented upstream | Repository carries an MIT license | Quarantined and capped at 5% of external sampling |
| [FECD whole flat-mounted Descemet membrane, parts 1–5](https://zenodo.org/records/18340536) | 499 high-resolution TIFFs (98.8 GiB); multi-scale high-information tiles are generated locally | Same endothelium tissue, but transmitted-light whole-mount microscopy rather than specular microscopy | CC BY 4.0; cite Maurin and Thuret, Zenodo 2026 | Source-level cap of 25% of external sampling; it cannot dominate by file/tile count |

The first-stage manifest contains 587 decoded and deduplicated images.  The
large FECD set is admitted only after every file passes its Zenodo size and
MD5 check.  Each whole-mount image then contributes up to 16 high-information
tiles at each of 512, 1024, and 2048-pixel source scales.  Exact SHA-256 and
same-source exact perceptual-hash duplicates are removed.

Within JEPA pretraining, CLEAR-EC remains 70% of samples and all external data
combined is 30%.  External source selection is done by the capped source
weights above, not by raw file count.  This is important because the adjacent
FECD collection is orders of magnitude larger than the true specular sets.

## Checked but not admitted as public image data

- The 90-image Topcon SP-3000P study links an [OSF project](https://osf.io/75kmu/),
  but its public files are measurements, documentation, and code rather than
  the source microscopy images.
- The 1,203-image Topcon SP-1P guttae dataset described by Vigueras-Guillén et
  al., *Scientific Reports* 2022 is available from the corresponding author on
  reasonable request, not as an anonymous public download.
- The 383-image Rotterdam SP-1P dataset used in related work is likewise
  author-request access; the public repository contains code and an example,
  not the complete image corpus.
- The 2,000-image Figshare item titled *Measurements of ex vivo human corneal
  endothelium using Voronoi segmentation* contains a cell-measurement table,
  not 2,000 source images.
- The 30-image Alizarine/porcine set is repeatedly cited as public, but its
  original BioImLab URL is no longer live and no authoritative downloadable
  mirror was found.  It is not fetched from unverified reposts.
- Corneal nerve, keratitis, OCT, slit-lamp, and external-eye datasets are not
  endothelial cell microscopy and are excluded to avoid negative transfer.

The [ESCRS ophthalmic dataset catalog](https://www.escrs.org/special-interest-groups/digital-health/current-datasets)
independently lists the two established downloadable endothelial datasets:
SREP (385 specular images) and RODREP (52 confocal images).
