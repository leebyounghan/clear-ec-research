# CD literature re-search (2026-11-13)

Question: which published or public methods for estimating corneal endothelial cell density (CD / ECD) from specular (or confocal) microscopy, or for count/density regression elsewhere, has the project not already tried?

How to read this file:
- "KG" = the project knowledge graph (`clear-ec-research/kg/wiki/`). "Closed list" = section 4(b) of `final_summary_cd_20261113.md`.
- Every paper has a reading depth: **full text**, **abstract**, **page summary** (a web page read through an automated summariser, so numbers should be re-checked at the source before reuse), or **snippet** (search-result text only).
- "My arithmetic" marks a number I derived from a paper's reported values; the paper does not state it.
- No leaderboard forecasts are given. Ranking is by (a) how different the idea is from the closed list and (b) how strong the published evidence is.
- PubMed is the source of all records cited with a PMID; DOIs are given as links.

## 1. Synthesis (one page)

**1. The search found no published CD method that is both materially different from the closed list and backed by strong evidence.** The specular-microscopy literature is segmentation pipelines (U-Net families plus watershed), one direct-regression paper, and frequency-domain estimators from 1997-2007. The project has tried each family. What is left is a short list of ideas from general machine learning (section 2), none with evidence on endothelium images.

**2. An independent participant has published a full experiment ledger, and it closes the same directions.** The public repository `Visilant/CLEAR-EC` (Phase I rank 4-5, hidden-set score 8.7583, out-of-fold CD 6.30) records 210 experiment rows up to 2026-09-14. It is new to the KG. Its findings agree with ours on almost every point:
- direct whole-frame regression beats every segmentation or detector readout (detector Voronoi readout CD 21-35 %; a trained cell-centre detector CD 10.1 against the CNN's 6.3);
- a density-map CD head, count supervision of the detector, scale jitter with label transform, trimmed loss, label cleaning, larger backbones, higher resolution, longer training and extra seeds all tie or lose;
- a label-free lattice-spacing estimate is median-unbiased but has MAPE 24 and adds nothing in a stack (6.375 against 6.374);
- the ensemble saturates (CD log-residual correlation 0.955-0.974 between families), and their larger 17-member ensemble scored worse on the hidden set than the 10-member one (8.8531 against 8.7583) despite a better out-of-fold value.
Their reading of the remaining CD error matches ours: heavy-tailed, donor-level label error (fellow-eye CD residual correlation 0.50; the top 5 % of images carry 24 % of CD MAPE).

**3. The peer-reviewed agreement literature puts human-against-human CD disagreement at the same size as our model error.** Details in section 3:
- Two reading-centre readers on the same donor image sets, same method: difference SD 180 cells/mm² (about 6.5 % of the mean); 66 % of pairs within 5 % (Benetz 2019).
- Eye-bank technician against a reading-centre analyst, same images, same Konan centre method: limits of agreement -304 to +323 cells/mm²; 89 % within 10 % (Huang 2022).
- Eye bank (usual method) against reading centre: difference SD 330 cells/mm² (about 12 %); 69 % within 10 %, 44 % within 5 % (Benetz 2019); 65 % within 10 % a decade earlier (Lass 2005).
- Within one US eye bank, 51 technicians' mean CD ranged from 2386 to 3005 cells/mm², with significant technician and calendar-date effects (Rand 2017).
- Re-analysing the same dotted cells is nearly deterministic (repeat CV 0.25 %, Ohno 1999), so the scatter comes from which cells and region a person selects, and from who that person is.
Our candidate's 6.17 % MAPE is measured against single-technician labels of this kind.

**4. Where the label scatter comes from, according to the literature.** (a) Region and cell selection: technicians pick the area with the most clearly visible cells, which reads about 100 cells/mm² above a reading centre (Lass 2005; Benetz 2019). (b) Number of cells analysed: disagreement doubles below 100 cells (SD 603 against 308 cells/mm²). (c) Technician identity and date. (d) Instrument calibration: Konan's internal calibration read 6 % lower than an external scale in one study (Ohno 1999). Items (c) and (d) are not visible in a single image.

**5. Ideas that differ from the closed list** (section 2): a donor random-intercept training loss; training-time weighting by predicted label reliability; a contrastive regression objective for the CD encoders; hard latent-window regression; class-agnostic counting models as a new readout. Evidence for each is generic or indirect.

**6. One cheap measurement would put the model on the published scale.** The agreement papers report "fraction within 5 % / within 10 %" and Bland-Altman limits. The same statistics on the stored out-of-fold CD predictions allow a direct comparison. This is a read, not an experiment.

## 2. Methods not yet tried by the project, ranked

### 2a. Ranked candidates

| # | Idea | Source (reading depth) | Evidence in the source | Needs annotations? Inference cost | Why it differs from the closed list |
|---|---|---|---|---|---|
| 1 | **Donor random intercept in the training loss** (mixed-effects deep regression): label = f(image) + donor/session effect + noise; fit f with the shared effect estimated and discounted; predict with f alone | Xiong, Kim, Singh 2019 MeNets (snippet); Simchoni & Rosset 2023 LMMNN (snippet); Nguyen, Treacher, Montillo 2023 ARMED (abstract). Domain motivation: Rand 2017 (abstract); Visilant ledger (page summary); our O-CDANAT-002 | MeNets: 10-20 % gain on gaze data with tens of users and hundreds of images each. ARMED: 5-28 % relative gain on seen clusters, 2-9 % on unseen clusters, four datasets. Domain: technician means 2386-3005 cells/mm²; fellow-eye CD residual correlation 0.50 | No annotations; uses the donor grouping already used for folds. Training-only, no inference cost | `exp-gpu2-within-donor-training` (+0.103) put a loss on the difference of fellow-eye errors, which makes the model explain within-donor label differences. A random intercept discounts the shared component instead. `exp-gpu1-donor-memory` acted at inference. Caveats: only two images per donor for about half the donors; under an L1-type loss the gain is efficiency only; check against `con-no-source-id-shortcuts` before running |
| 2 | **Training-time weighting by predicted label reliability** (heteroscedastic Laplace/Gaussian likelihood, beta-NLL form) | Seitzer et al. 2022 (snippet); Kendall & Gal 2017 (KG). Domain: Benetz 2019 (full text); Lass 2005 (abstract); Doughty 2000 (abstract); Abib 2012 (abstract) | Label disagreement SD by cells analysed: 603 (<100), 308 (100-300), 260 (>300) cells/mm². Agreement falls with image quality. Visilant: ensemble disagreement predicts CD error (top quintile 8.95 against 5.5) | No annotations. Training-only | The closed uncertainty work used predicted noise for a post-hoc MAPE shift (`exp-cd-uncertainty-shift`, +0.012 to +0.025). Using it as a per-sample training weight is not in the closed list. Caveat: noisy rows still count in the score, so down-weighting them can move the fit either way |
| 3 | **Contrastive regression objective (Rank-N-Contrast)** for the CD encoders | Zha, Cao, Son, Yang, Katabi 2023 (abstract) | Improves five regression benchmarks; authors report better data efficiency and robustness to corrupted data and spurious targets | No annotations. Training-only | Order losses were tried on CV only (`exp-cv-order-loss-dinov3`, not adopted); RankSim is recorded but not run on CD. This method shapes the representation before the regression head |
| 4 | **Hard latent-window (prime-instance) regression**: the label is produced by one unknown window; train per-window density with a min-over-windows or EM assignment; predict the MAPE-optimal summary over the window posterior | Ray & Page 2001 (snippet). Domain: Benetz 2024 (KG); Benetz 2019 (full text); Luznik 2019 (page summary) | MIR literature is general, with no image-density evidence. Domain: matched-area crop cut eye-bank CD bias from 123 to 46 cells/mm²; protocols ask for at least 150 contiguous cells in a rectangular frame | No annotations. One extra head | Attention pooling (+0.0047) is a soft, unconstrained weighting; the technician-window crop (+0.0026) fixes one heuristic window. `idea-technician-hypothesis-selection` in the KG may be this idea; its test and result are not recorded, so check first |
| 5 | **Class-agnostic counting models as a new density readout** (reference-less or exemplar-based) | Hobley & Prisacariu 2022 RCC (abstract); CounTR 2022, LOCA 2023, CountGD 2024, GeCo 2024 (snippets) | Benchmarks on FSC-147 only; RCC regresses counts from self-supervised ViT features with count-level labels. No evidence on endothelium or other dense regular mosaics | No annotations. Needs new pretrained weights (`con-no-downloads-without-approval`). One forward pass per model | Our readouts are Cellpose-SAM masks and a peak detector; these are a different family. Prior is low: detector density already reads weight 0 and mask density 0.1 |

### 2b. Literature ideas that map onto closed directions

| Literature idea | Source | Our closest closed item | What differs |
|---|---|---|---|
| Fourier-ring density (2-D DFT circular band) | Fitzke 1997; Foracchia & Ruggeri 2004; Ruggeri 2005, 2007 | Label-free Fourier-ring CD, weight 0 [O-CD-FFT-001] | Same idea. Published accuracy on eye-bank light-microscopy images: mean absolute difference 3.9 %, SD 5.1 % against manual (Ruggeri 2005). Visilant's lattice-spacing estimate: MAPE 24, stack +0.001 |
| Lattice generating-matrix estimate | Grisan et al. 2005 | Same | Estimates a 2-D lattice basis instead of a ring radius; 21 images, 6.5 % absolute difference. Not tried in this exact form; two negative reads on the ring version make the prior low |
| Frequency-derived cell size used to set the segmenter's scale | Selig 2015; Vigueras-Guillen 2019/2020 (KG) | Watershed and mask routes | Uses the frequency estimate as a parameter, not as a prediction |
| Density-map head trained from image-level labels | Lempitsky 2010 (KG); von Borstel 2016; Xue 2016 | `exp-hist-cd-density-channel`; Visilant density head (fold-0 score 10.43 / 12.39, dead) | None |
| Count-supervised detector / count-consistent pseudo-labels | Visilant handoff 2026-09-14; Kervadec 2019 (KG) | `exp-rc-pseudo-label-pool`, `exp-bb-aux-detector-targets`, `exp-hex-count-matched-hex6` | Visilant found that count supervision turns the detector into a density regressor (CD about 8 against 6.3) |
| Outlier-mixture likelihood (Gaussian + uniform) | Lathuiliere 2018 DeepGUM | Label-clean training +0.0003; Visilant trimmed loss ties/loses, label cleaning -0.028 n.s. | Soft down-weighting instead of dropping rows. With an L1/MAPE loss the fit is already outlier-robust |
| Crop-ranking self-supervision for counting | Liu 2018; Yang 2020 | none | Not applicable as published: density is intensive, so a crop does not have a lower target than its parent |
| Scale augmentation with label transform | Visilant (loses on both folds) | `exp-hist-canonical-zoom` (+0.067 to +0.235) | None |
| Frozen multi-backbone embeddings + SVR | PetFinder first place (secondary blog) | `exp-gpu1-frozen-ridge-member` (+0.001), frozen pooling heads | Kernel head instead of ridge |
| Classical dual watershed + learned calibration (GBM + randomised trees, shrink on disagreement) | Two Grand Challenge algorithm pages (Penn, Aug 2026) | Mask density, GBM residual stackers | Their cross-validated CD mean error is 9.96 % (median 6.55 %) |
| Segmentation restricted to an area matched to the counted region | Benetz 2024 (KG) | Technician-window crop; mask-density window variants | None |

## 3. How device CD labels are produced, and their noise and bias

### 3.1 The procedure

- **Konan centre method** (used by eye banks for 84 % of preoperative image sets in the Cornea Preservation Time Study). The technician dots the centre of each cell in a contiguous group; the software builds cell polygons and drops the outermost ring (the flex-centre variant keeps it). CD = analysed cells / their area = 1e6 / mean cell area; CV = SD / mean of cell area; HEX = share of six-sided cells. (McCarey 2008 and Takahashi 1996 in the KG; Benetz 2019 full text; Luznik 2019 page summary.)
- **How many cells, and where.** A published eye-bank protocol for the Konan CellChek D+ asks for at least 150 contiguous cells in minimally overlapping rectangular frames, with a 5 % dual-grading tolerance on CD (Luznik 2019). This matches our measured median of 148 markers in a fixed box.
- **Other counting rules in eye banks.** Variable frame (count cells inside a traced outline; used by the reading centre and by HAI instruments) and fixed frame. In the Cornea Preservation Time Study the eye banks used centre 84 %, variable frame 12 %, fixed frame 4 %; instruments were 32 Konan and 7 HAI.

### 3.2 Agreement numbers

| Comparison | Design | Result | Source (depth) |
|---|---|---|---|
| Same image, same dotted cells, repeated | Konan centre method, normal corneas | repeat CV 0.25 % (centre), 0.99 % (corners) | Ohno 1999 (abstract) |
| Same image, same examiner, two days | Topcon semi-automated, 86 eyes | limits of agreement about ±100 cells/mm² with the two largest analysis boxes, ±179 and ±488 with the smaller ones | Ding 2012 (abstract) |
| Two certified graders, centre method, good images | 278 healthy eyes, Konan NSP-9900 | inter-grader ICC > 0.99; "ECD variance" 2.0 % ± 1.6 % (the abstract does not say whether between graders or frames); no significant difference among three frames | Shi 2019 (abstract) |
| Two reading-centre readers, variable frame, same donor image sets | 1276 preoperative sets | mean difference 40, SD 180 cells/mm², limits -320 to +400; 66 % within 5 % | Benetz 2019 (full text) |
| Two reading-centre readers, earlier study | 645 analysable donor images | 69 % within 5 % | Benetz 2006 (abstract) |
| Eye-bank technician vs reading-centre analyst, same images, same centre method | 200 donors, two eye banks | mean difference +10 cells/mm²; limits -304 to +323; 89 % within 10 % | Huang 2022 (abstract) |
| Eye bank (usual method) vs reading centre | 1286 donors, 23 eye banks | mean +15, SD 330, limits -644 to +675 cells/mm²; 69 % within 10 %, 44 % within 5 % | Benetz 2019 (full text) |
| Same, by cells analysed | <100 / 100-300 / >300 cells | SD 603 / 308 / 260 cells/mm² | Benetz 2019 (full text) |
| Same, screening images | 984 donors | eye bank +108 cells/mm²; 70 % within 10 %; 24 % more than 10 % higher, 6 % lower | Benetz 2019 (full text) |
| Eye bank vs reading centre, 2005 | 663 images, 23 eye banks | 65 % within 10 %; 28 % more than 10 % higher, 7 % lower | Lass 2005 (abstract) |
| Eye-bank centre method vs reading-centre variable frame | one eye bank, 176 images before preparation | 2678 vs 2599 cells/mm² (+79) | Mayko 2016 (abstract) |
| Technicians within one US eye bank | 51 technicians, 35,067 eyes | technician means 2386 ± 431 to 3005 ± 560 cells/mm²; nine technicians and the 2007-2009 period with significant effects | Rand 2017 (abstract) |
| Two instruments and methods, same corneas | 97 donor corneas, HAI variable frame vs Konan centre | 2764 vs 2605 cells/mm²; per-cornea difference 8.05 % (0.26-27.2 %); 5.6 % when both regraded in one software | Shi 2020 (abstract) |
| Small-sample dotting between two sites | 89 eyes, 30 dotted cells | site differences -65 % to +42 %; -15 % to +9 % when all visible cells were evaluated | Bohringer 2012 (abstract) |
| Konan centre method vs Tomey, healthy eyes | 49 eyes | mean absolute difference 93 cells/mm²; limits -222 to +247 | Jbara 2021 (abstract) |
| Eye-bank donor images, two graders | 54 corneas, Konan CellChek D | ICC 0.82 (centre) | Vidal-Villegas 2024 (KG) |

My arithmetic, from the table (the papers do not state these):
- Huang 2022: limits 627 cells/mm² wide give a difference SD of about 160 cells/mm², about 5.9 % at a typical 2,700 cells/mm² (the abstract does not give the mean). Under a normal approximation the mean absolute technician-against-analyst difference is about 4.7 %. If the two people err independently and equally, each has an SD of about 4.2 %, so a predictor of the error-free value would still show about 3.3 % mean absolute error against one technician's label.
- Benetz 2019, reader against reader: SD 180 on a mean of 2,758 is 6.5 %. A normal distribution with that SD would put about 54 % within 5 %; the reported 66 % means the differences are heavy-tailed.
- These are donor image sets (one to three images), trained staff and study conditions. Single-image, routine eye-bank labels should be noisier.

### 3.3 Known biases

- **Selection toward dense, clear areas.** Eye banks read higher than the reading centre by about 100 cells/mm² on screening images in two studies ten years apart; the authors attribute this to choosing the area where the most cells can be dotted (Lass 2005; Benetz 2019). In our KG: matched-area cropping cut the automated-against-eye-bank bias from 123 to 46 cells/mm², and the bias grew with CD (Benetz 2024).
- **Centre against flex-centre against corners.** Normal corneas: 2870 / 2846 / 2892 cells/mm² (Patel 2010, KG). With guttae the centre method reads 2.4 % higher than flex-centre; the two disagree only below about 30 identified cells (Huang 2017 guttae paper). Eye-bank comparison of 67 corneas: no significant difference (Villalba 2014).
- **Calibration.** Konan's internal calibration marks gave CD 6 % lower than an external scale (Ohno 1999). French eye banks: missing microscope calibration was the leading cause of between-bank differences (Thuret 2003).
- **Fixed-frame counting.** Residual variability of about ±5 % even when ten counts are averaged; grids smaller than 200 × 200 µm should not be used (Campolmi 2013).
- **Number of cells.** ±10 % with 25 cells, about ±2 % with 75 (Doughty 2000). Konan examinations with 80 ± 27 analysed cells had a 10.6 % relative sampling error for inference to the whole endothelium (Abib 2012). About 425 marked cells over about three images are needed for a 5 % error (van den Berg 2020). Single small fields were at least 10 % away from the central 4 mm mean in all but the most regular endothelia (Hirst 1989).
- **Fully automatic device modes** overestimate CD against manual dotting: Konan Auto Tracer +5.7 % (Karmakar 2022); Nidek NAVIS +7.0 %, too low at high CD and too high at low CD (Hirneiss 2007); Konan Robo and Nidek Confoscan in grafted eyes (Price 2013); Topcon IMAGEnet (Cheung & Cho 2000); Nidek ConfoScan 3 (Kitzmann 2005). Centre-dot precision with three replicates was only about ±10 % across two devices (Maruoka 2018).
- **Organ-culture light microscopy** reads higher than specular microscopy (10.5 % manual, 31.9 % automated; Rickmann 2019). Not our setting.

### 3.4 What the participant ledger adds (Visilant, page summary, not peer-reviewed)

- On the 25 released overlays, CD recomputed from the technician's own clicks has bias -3.0 % and scatter 0.8 %; a bootstrap counting floor for CD is 1.52 %.
- CD residuals: skew 2.1, excess kurtosis 37; top 5 % of images carry 24 % of CD MAPE; dropping the top 1 / 2 / 5 % gives CD 5.71 / 5.49 / 5.02.
- Fellow-eye residual correlation: CD 0.50, CV 0.30, HEX 0.05.
- The same rows we flagged (donors 2020-0644, -0646, -0651: labels 700-900 on mosaics read at 2,700-3,150).
- Hidden-set noise: one candidate's mean score has SD about 0.30 on 1,000 images; the paired SD between two similar candidates is 0.018.
- Their Phase II plan (not executed in the repository): more click sets across image series, a click-noise-tolerant detector target with spacing-aware NMS, count-consistent pseudo-labels, series-balanced training. The first needs human annotation.

## 4. Annotated bibliography

Tags: **[NEW]** = not in the KG; **[KG]** = already a KG paper node.

### Area 1. Corneal endothelium analysis: methods

1. [NEW] Fitzke FW, Masters BR, Buckley RJ, Speedwell L. Fourier transform analysis of human corneal endothelial specular photomicrographs. Exp Eye Res 1997;65(2):205-214. [doi](https://doi.org/10.1006/exer.1997.0326). Abstract. First direct Fourier analysis of clinical specular images; ring radius follows cell size.
2. [NEW] Foracchia M, Ruggeri A. Automatic estimation of endothelium cell density in donor corneas by means of Fourier analysis. Med Biol Eng Comput 2004;42(5):725-731. [doi](https://doi.org/10.1007/BF02347557). Abstract. 18 corneas; 3 % mean, 19 % maximum difference from two experts.
3. [NEW] Ruggeri A, Grisan E, Jaroszewski J. A new system for the automatic estimation of endothelial cell density in donor corneas. Br J Ophthalmol 2005;89(3):306-311. [doi](https://doi.org/10.1136/bjo.2004.051722). Abstract. 100 donor corneas; mean difference 0.9 %, SD 5.1 %, mean absolute 3.9 % against three experts.
4. [NEW] Grisan E, Paviotti A, Laurenti N, Ruggeri A. A lattice estimation approach for the automatic evaluation of corneal endothelium density. Conf Proc IEEE Eng Med Biol Soc 2005:1700-1703. [doi](https://doi.org/10.1109/IEMBS.2005.1616771). Abstract. Density from an estimated lattice generating matrix; 21 images, 6.5 % absolute difference.
5. [NEW] Ruggeri A, Grisan E, Schroeter J. Evaluation of repeatability for the automatic estimation of endothelial cell density in donor corneas. Br J Ophthalmol 2007;91(9):1213-1215. [doi](https://doi.org/10.1136/bjo.2006.112763). Abstract. 200 corneas; accuracy -2 % (SD 4 %); repeat precision 3 %.
6. [NEW] Foracchia M, Ruggeri A. Corneal endothelium cell field analysis by means of interacting Bayesian shape models. Conf Proc IEEE Eng Med Biol Soc 2007:6036-6039. [doi](https://doi.org/10.1109/IEMBS.2007.4353724). Abstract. Shape-prior cell-field model; preliminary.
7. [NEW] Ruggeri A, Scarpa F, De Luca M, Meltendorf C, Schroeter J. A system for the automatic estimation of morphometric parameters of corneal endothelium in alizarine red-stained images. Br J Ophthalmol 2010;94(5):643-647. [doi](https://doi.org/10.1136/bjo.2009.166561). Abstract. Density difference -12 ± 52 cells/mm² on 30 porcine images.
8. [NEW] Selig B, Vermeer KA, Rieger B, Hillenaar T, Luengo Hendriks CL. Fully automatic evaluation of the corneal endothelium from in vivo confocal microscopy. BMC Med Imaging 2015;15:13. [doi](https://doi.org/10.1186/s12880-015-0054-3). Abstract. Frequency-based density plus stochastic watershed; Nidek NAVIS overestimates below about 1,200 cells/mm².
9. [NEW] Scarpa F, Ruggeri A. Automated morphometric description of human corneal endothelium from in-vivo specular and confocal microscopy. Conf Proc IEEE Eng Med Biol Soc 2016:1296-1299. [doi](https://doi.org/10.1109/EMBC.2016.7590944). Abstract. Kernel centre detection plus genetic-algorithm polygon fit; 30 images.
10. [NEW] Heinzelmann S, Daniel MC, Maier PC, Reinhard T, Bohringer D. Automated cell counting using "deep learning" in donor corneas from organ culture achieves high precision and accuracy. Klin Monbl Augenheilkd 2019;236(12):1407-1412. [doi](https://doi.org/10.1055/a-1023-4339). Abstract. U-Net on phase-contrast images; r 0.90 with an expert, 0.81 for a less experienced grader.
11. [NEW] Canavesi C, Cogliati A, Hindman HB. Unbiased corneal tissue analysis using Gabor-domain optical coherence microscopy and machine learning for automatic segmentation of corneal endothelial cells. J Biomed Opt 2020;25(9):092902. [doi](https://doi.org/10.1117/1.JBO.25.9.092902). Abstract. 1 mm² field; r 0.91 with manually counted specular results; argues that counting all cells removes selection bias.
12. [NEW] Karmakar R, Nooshabadi S, Eghrari A. An automatic approach for cell detection and segmentation of corneal endothelium in specular microscope. Graefes Arch Clin Exp Ophthalmol 2022;260(4):1215-1224. [doi](https://doi.org/10.1007/s00417-021-05483-8). Abstract. 303 Konan CellChek images; relative mean absolute error 6.9 % against manual; Konan Auto Tracer 7.9 % with +5.7 % bias.
13. [NEW] Karmakar R, Nooshabadi SV, Eghrari AO. Mobile-CellNet: automatic segmentation of corneal endothelium using an efficient hybrid deep learning model. Cornea 2023;42(4):456-463. [doi](https://doi.org/10.1097/ICO.0000000000003186). Abstract. CD mean absolute error 4.06 % (U-Net 3.80 %).
14. [NEW] Shilpashree PS, Suresh KV, Sudhir RR, Srinivas SP. Automated image segmentation of the corneal endothelium in patients with Fuchs dystrophy. Transl Vis Sci Technol 2021;10(13):27. [doi](https://doi.org/10.1167/tvst.10.13.27). Abstract.
15. [NEW] Mantena S, Chandra J, Pecyna E, et al. Low-cost, smartphone-based specular imaging and automated analysis of the corneal endothelium. Transl Vis Sci Technol 2021;10(4):4. [doi](https://doi.org/10.1167/tvst.10.4.4). Abstract. 30 eyes; CD 2799 vs 2779 cells/mm² against a Tomey EM-4000.
16. [NEW] Qu JH, Qin XR, Peng RM, et al. A fully automated segmentation and morphometric parameter estimation system for assessing corneal endothelial cell images. Am J Ophthalmol 2022;239:142-153. [doi](https://doi.org/10.1016/j.ajo.2022.02.026). Abstract. Confocal images; r 0.932 with device density; limits of agreement -572 to +272 cells/mm².
17. [NEW] Zhang Y, Higashita R, Fu H, et al. A multi-branch hybrid transformer network for corneal endothelial cell segmentation. MICCAI 2021. [arXiv:2106.07557](https://arxiv.org/abs/2106.07557). Snippet; author list from memory.
18. [NEW] Semi-automatic analysis of donor cornea EC images using SSL. Proc SPIE 12930, 129301A (2024). [doi](https://doi.org/10.1117/12.3009714). Snippet; authors not confirmed. Self-supervised ViT segmenter on donor images: mean CD difference 9.74 %. Conference precursor of Benetz 2024.
19. [NEW] Foo VHX, Lim GYS, Liu YC, et al. Deep learning for detection of Fuchs endothelial dystrophy from widefield specular microscopy imaging: a pilot study. Eye Vis (Lond) 2024;11(1):11. [doi](https://doi.org/10.1186/s40662-024-00378-1). Abstract. Classification from images; AUC 0.96 internal, 0.77 external.
20. [NEW] Cheong EZK, Win KY, Wong QY, et al. Deep learning-assisted widefield endothelial imaging in Descemet membrane endothelial keratoplasty. Front Med (Lausanne) 2025;12:1715673. [doi](https://doi.org/10.3389/fmed.2025.1715673). Abstract. Image-quality and low-CD classifiers, AUC 0.98.
21. [NEW] Doke AR, et al. Automated corneal endothelial cell detection and segmentation in specular microscopy images using Attention and Faster U-Net. Sci Rep 2026. Abstract via Consensus; DOI not retrieved. Public 385-image set; R² 0.95 for CD, cell recall 0.65.
22. [NEW] Piaia M, Rodella U, Ragazzi E, et al. A novel hybrid software-assisted method to evaluate quantitatively corneal endothelium from light microscopy images. Microsc Res Tech 2026;89(3):412-422 (online 2025). [doi](https://doi.org/10.1002/jemt.70079). Abstract.
23. [NEW] Brookes NH. Morphometry of organ cultured corneal endothelium using Voronoi segmentation. Cell Tissue Bank 2017;18(2):167-183. [doi](https://doi.org/10.1007/s10561-017-9622-9). Abstract. Voronoi from centroids on 2,000 images.
24. [NEW] Jukic A, Culina K, Markusic V, Masnec S, Matic S, Kalauz M. Artificial intelligence in corneal endothelial disease and endothelial keratoplasty: current evidence, clinical readiness, and future directions. Front Ophthalmol (Lausanne) 2026;6:1951354. [doi](https://doi.org/10.3389/fopht.2026.1951354). Abstract only. Review to July 2026.
25. [KG] viguerasguillen2019regression: the only published direct regression of CD/CV/HEX (from edge images), CD MAPE 4.32 % on 738 images.
26. [KG] Also re-encountered: benetz2024donorecd, daniel2019unet, fabijanska2018unet, viguerasguillen2019fcn, viguerasguillen2020dsaek, viguerasguillen2022feedbackattn, kolluru2019unet, joseph2020postkeratoplasty, sierra2023sdm, tey2025widefield, scarpa2016morphometry, piorkowski2017segmentation, marrugo2026review.

### Area 1 (continued). Agreement, repeatability and counting-rule papers

27. [NEW] Lass JH, Gal RL, Ruedy KJ, et al. An evaluation of image quality and accuracy of eye bank measurement of donor cornea endothelial cell density in the Specular Microscopy Ancillary Study. Ophthalmology 2005;112(3):431-440. [doi](https://doi.org/10.1016/j.ophtha.2004.10.045). Abstract.
28. [NEW] Benetz BA, Gal RL, Ruedy KJ, et al. Specular microscopy ancillary study methods for donor endothelial cell density determination of Cornea Donor Study images. Curr Eye Res 2006;31(4):319-327. [doi](https://doi.org/10.1080/02713680500536738). Abstract.
29. [NEW] Benetz BA, Stoeger CG, Patel SV, et al. Comparison of donor cornea endothelial cell density determined by eye banks and by a central reading center in the Cornea Preservation Time Study. Cornea 2019;38(4):426-432. [doi](https://doi.org/10.1097/ICO.0000000000001846). Full text (PMC9490486). Main source for section 3.
30. [NEW] Huang H, Benetz BA, Clover JM, et al. Comparison of donor corneal endothelial cell density determined by eye banks and by a central image analysis reading center using the same image analysis method. Cornea 2022;41(5):664-668. [doi](https://doi.org/10.1097/ICO.0000000000002935). Abstract.
31. [NEW] Rand GM, Kwon JW, Gore PK, McCartney MD, Chuck RS. Technician consistency in specular microscopy measurements: a "real-world" retrospective analysis of a United States eye bank. Cornea 2017;36(10):1172-1177. [doi](https://doi.org/10.1097/ICO.0000000000001266). Abstract.
32. [NEW] Mayko ZM, Benetz BA, Menegay H, et al. Donor endothelial cell density measurements do not change immediately after DMEK preparation. Cornea 2016;35(12):1556-1561. [doi](https://doi.org/10.1097/ICO.0000000000001044). Abstract.
33. [NEW] Shi Y, Huang J, Conrick P, Hoover C, Lee OL. Discrepancies in endothelial cell density values of human donor corneas resulting from comparison between specular microscopes and endothelial analysis methods. Cornea 2020;39(4):495-500. [doi](https://doi.org/10.1097/ICO.0000000000002208). Abstract.
34. [NEW] Shi Y, Huang J, Baghdasaryan E, et al. Representation of central endothelial cell density by analysis of single best specular microscopy image regardless of cell size variance. Transl Vis Sci Technol 2019;8(3):23. [doi](https://doi.org/10.1167/tvst.8.3.23). Abstract.
35. [NEW] Luznik Z, Sun Z, Yin J, Benetz BA, Lass JH, Dana R. A standardized methodology for longitudinal assessment of corneal endothelial morphometry in eye banked corneas. J Biol Methods 2019;6(4):e120. [doi](https://doi.org/10.14440/jbm.2019.304). Page summary. Konan CD-15 / CellChek D+ protocol.
36. [NEW] Ohno K, Nelson LR, McLaren JW, Hodge DO, Bourne WM. Comparison of recording systems and analysis methods in specular microscopy. Cornea 1999;18(4):416-423. [doi](https://doi.org/10.1097/00003226-199907000-00005). Abstract.
37. [NEW] Benetz BA, Diaconu E, Bowlin SJ, Oak SS, Laing RA, Lass JH. Comparison of corneal endothelial image analysis by Konan SP8000 noncontact and Bio-Optics Bambi systems. Cornea 1999;18(1):67-72. PMID 9894940. Abstract. Fixed frame vs dot vs corners: differences of 19-157 cells/mm².
38. [NEW] Huang J, Liu X, Tepelus TC, et al. Comparison of the center and flex-center methods of corneal endothelial cell analysis in the presence of guttae. Cornea 2017;36(12):1514-1520. [doi](https://doi.org/10.1097/ICO.0000000000001334). Abstract.
39. [NEW] Huang J, Maram J, Tepelus TC, et al. Comparison of noncontact specular and confocal microscopy for evaluation of corneal endothelium. Eye Contact Lens 2018;44 Suppl 1:S144-S150. [doi](https://doi.org/10.1097/ICL.0000000000000362). Abstract.
40. [NEW] Villalba R, Jimenez A, Fornes G, Eisman M, Villagran JL. Flex center method versus center method for endothelial corneal evaluation in eye banking: a comparative analysis. Cell Tissue Bank 2014;15(4):507-512. [doi](https://doi.org/10.1007/s10561-014-9434-0). Abstract.
41. [NEW] Maruoka S, Nakakura S, Matsuo N, et al. Comparison of semi-automated center-dot and fully automated endothelial cell analyses from specular microscopy images. Int Ophthalmol 2018;38(6):2495-2507. [doi](https://doi.org/10.1007/s10792-017-0760-7). Abstract.
42. [NEW] Jbara D, Achiron A, Antman G, et al. Agreement of corneal endothelial cell analysis between Konan-Noncon Robo SP-6000 and Tomey EM-3000 specular microscopes in healthy subjects. Eye Contact Lens 2021;47(4):191-195. [doi](https://doi.org/10.1097/ICL.0000000000000712). Abstract.
43. [NEW] Price MO, Fairchild KM, Price FW. Comparison of manual and automated endothelial cell density analysis in normal eyes and DSEK eyes. Cornea 2013;32(5):567-573. [doi](https://doi.org/10.1097/ICO.0b013e31825de8fa). Abstract.
44. [NEW] Hirneiss C, Schumann RG, Gruterich M, et al. Endothelial cell density in donor corneas: a comparison of automatic software programs with manual counting. Cornea 2007;26(1):80-83. [doi](https://doi.org/10.1097/ICO.0b013e31802be629). Abstract.
45. [NEW] Kitzmann AS, Winter EJ, Nau CB, McLaren JW, Hodge DO, Bourne WM. Comparison of corneal endothelial cell images from a noncontact specular microscope and a scanning confocal microscope. Cornea 2005;24(8):980-984. [doi](https://doi.org/10.1097/01.ico.0000159737.68048.97). Abstract.
46. [NEW] Luft N, Hirnschall N, Schuschitz S, Draschl P, Findl O. Comparison of 4 specular microscopes in healthy eyes and eyes with cornea guttata or corneal grafts. Cornea 2015;34(4):381-386. [doi](https://doi.org/10.1097/ICO.0000000000000385). Abstract.
47. [NEW] Cheung SW, Cho P. Endothelial cells analysis with the TOPCON specular microscope SP-2000P and IMAGEnet system. Curr Eye Res 2000;21(4):788-798. [doi](https://doi.org/10.1076/ceyr.21.4.788.5548). Abstract. CV/HEX reliability ICC 0.4-0.8.
48. [NEW] Vecchi M, Braccio L, Orsoni JG. The Topcon SP 1000 and Image-NET systems: a comparison of four methods for evaluating corneal endothelial cell density. Cornea 1996;15(3):271-277. [doi](https://doi.org/10.1097/00003226-199605000-00008). Abstract.
49. [NEW] Prinz A, Varga J, Findl O. Reliability of a video-based noncontact specular microscope for assessing the corneal endothelium. Cornea 2007;26(8):924-929. [doi](https://doi.org/10.1097/ICO.0b013e3180cfe4f1). Abstract. Test-retest ICC 0.77 automatic, 0.99 semi-automatic, 0.91 Konan.
50. [NEW] Ding X, Huang Q, Zheng Y, Jiang Y, Huang S, He M. Measurement area and repeatability of semiautomated assessment of corneal endothelium in the Topcon specular microscope SP-2000P and IMAGEnet system. Cornea 2012;31(10):1111-1118. [doi](https://doi.org/10.1097/ICO.0b013e31820e42b5). Abstract.
51. [NEW] Doughty MJ, Muller A, Zaman ML. Assessment of the reliability of human corneal endothelial cell-density estimates using a noncontact specular microscope. Cornea 2000;19(2):148-158. [doi](https://doi.org/10.1097/00003226-200003000-00006). Abstract.
52. [NEW] Doughty MJ. Could the coefficient of variation (COV) of the corneal endothelium be overestimated when a centre-dot method is used? Clin Exp Optom 2008;91(1):103-110. [doi](https://doi.org/10.1111/j.1444-0938.2007.00203.x). Abstract. Same mean cell area, CV 43.8 % vs 29.0 %.
53. [NEW] Doughty MJ, Oblak E. A comparison of two methods for estimating polymegethism in cell areas of the human corneal endothelium. Ophthalmic Physiol Opt 2008;28(1):47-56. [doi](https://doi.org/10.1111/j.1475-1313.2007.00533.x). Abstract.
54. [NEW] Doughty MJ, Aakre BM. Further analysis of assessments of the coefficient of variation of corneal endothelial cell areas from specular microscopic images. Clin Exp Optom 2008;91(5):438-446. [doi](https://doi.org/10.1111/j.1444-0938.2008.00281.x). Abstract.
55. [NEW] Doughty MJ. Evaluation of possible error sources in corneal endothelial morphometry with a semiautomated noncontact specular microscope. Cornea 2013;32(9):1196-1203. [doi](https://doi.org/10.1097/ICO.0b013e318295e72a). Abstract.
56. [NEW] van den Berg AB, Holzchuh R, Hida RY. Possible sampling error in corneal specular microscopy: can central endothelial data generated by specular microscope represent the real clinical condition of the whole cornea? Cornea 2020;39(6):779-781. [doi](https://doi.org/10.1097/ICO.0000000000002290). Abstract.
57. [NEW] Abib FC, Holzchuh R, Schaefer A, Schaefer T, Godois R. The endothelial sample size analysis in corneal specular microscopy clinical examinations. Cornea 2012;31(5):546-550. [doi](https://doi.org/10.1097/ICO.0b013e3181cc7961). Abstract.
58. [NEW] Hirst LW, Yamauchi K, Enger C, Vogelpohl W, Whittington V. Quantitative analysis of wide-field specular microscopy. II. Precision of sampling from the central corneal endothelium. Invest Ophthalmol Vis Sci 1989;30(9):1972-1979. PMID 2777517. Abstract.
59. [NEW] Campolmi N, Acquart S, He Z, et al. Inherent errors of the fixed-frame counting method for corneal endothelial cell density in eye banks. Cell Tissue Bank 2014;15(3):451-459. [doi](https://doi.org/10.1007/s10561-013-9411-z). Abstract.
60. [NEW] Campolmi N, He Z, Acquart S, et al. Comparison of endothelial cell density of organ cultured corneas with cornea donor study. Cornea 2014;33(6):597-603. [doi](https://doi.org/10.1097/ICO.0000000000000124). Abstract.
61. [NEW] Thuret G, Manissolle C, Acquart S, et al. Is manual counting of corneal endothelial cell density in eye banks still acceptable? The French experience. Br J Ophthalmol 2003;87(12):1481-1486. [doi](https://doi.org/10.1136/bjo.87.12.1481). Abstract.
62. [NEW] Deb-Joardar N, Thuret G, Racine GA, et al. Standard microlithographic mosaics to assess endothelial cell counting methods by light microscopy in eye banks using organ culture. Invest Ophthalmol Vis Sci 2006;47(10):4373-4377. [doi](https://doi.org/10.1167/iovs.06-0536). Abstract.
63. [NEW] Bohringer D, Hettich L, Maier PC, Reinhard T. Data quality of unsupervised endothelial cell counting vs. reading centre analysis in multicentric clinical trials. Klin Monbl Augenheilkd 2012;229(6):628-631. [doi](https://doi.org/10.1055/s-0031-1299508). Abstract.
64. [NEW] Rickmann A, Boden KE, Wahl S, et al. Significant differences between specular microscopy and corneal bank endothelial cell counts: a pilot study. Acta Ophthalmol 2019;97(8):e1077-e1081. [doi](https://doi.org/10.1111/aos.14185). Abstract.
65. [NEW] Geroski DH, Edelhauser HF. Morphometric analysis of the corneal endothelium: specular microscopy vs. alizarin red staining. Invest Ophthalmol Vis Sci 1989;30(2):254-259. PMID 2464555. Abstract.
66. [KG] Already recorded: patel2010flexcenter, mccarey2008review, huang2017autovsmanual, vidalvillegas2024eyebank, gasser2015konantopcon, desanctis2006twodevices, doughty2018regional, doughty2020cvcells, takahashi1996konanpatent, bucht2011simulation.

### Area 2. The CLEAR-EC challenge and related public material

67. [NEW] Visilant/CLEAR-EC (public fork of the organisers' baseline repository): `docs/EXPERIMENTS.md`, `docs/HANDOFF_2026-09-12.md`, `docs/HANDOFF_2026-09-14.md`, `docs/CAMPAIGN_2026-09-13_C5_WEIGHTS.md`, `docs/RESEARCHER_GUIDE.md`. https://github.com/Visilant/CLEAR-EC . Page summaries. ConvNeXt-V2-Tiny / ConvNeXt-Tiny / V2-Base direct regression at 486 × 648 with a relative loss, flip TTA, geometric-mean ensemble; out-of-fold CD 6.30, CV 10.25, HEX 9.97; hidden Phase I 8.7583. Last commit 2026-09-14; no Phase II entries.
68. [NEW] "Quality Triaged Endothelial Morphometry with Calibrated Confidence", Grand Challenge algorithm page (editor at Penn Medicine; versions 2026-08-09/10). https://grand-challenge.org/algorithms/quality-triaged-endothelial-morphometry-with-calib/ . Page summary. Dual watershed, contiguous gradable region, learned calibration and confidence. Five-fold CD: median 8.0 %, mean 12.1 % (5,958 records). A search snippet showed Phase I rank 7; the fetched page showed no rank.
69. [NEW] "Triaged ML", Grand Challenge algorithm page (same editor; 2026-08-10; Phase I rank 25). https://grand-challenge.org/algorithms/triaged-ml/ . Page summary. Same classical pipeline plus a median-error gradient-boosted model and a randomised-tree model in log space, shrunk when they disagree. Five-fold CD: median 6.55 %, mean 9.96 %; by gradable fraction 11.84 / 11.07 / 9.14 % (low / medium / high).
70. [NEW] Karmakar R, Kiros EK, Nath S, et al. Agentic artificial intelligence in eye banking: a proposed workflow. Eye Bank Corneal Transpl 2026;5(2). [doi](https://doi.org/10.1097/ebct.0000000000000054). Abstract. Proposal only.
71. [KG] clearec2026challenge. Re-read 2026-11-13: still no label-production description, no challenge paper, no forum. The AAO 2025 poster and ARVO abstract named on the Zenodo record could not be found.

### Area 3. Counting, density regression and regression under label noise

72. [NEW] Hobley M, Prisacariu V. Learning to count anything: reference-less class-agnostic counting with weak supervision. 2022. [arXiv:2205.10203](https://arxiv.org/abs/2205.10203). Abstract.
73. [NEW] Liu X, van de Weijer J, Bagdanov AD. Leveraging unlabeled data for crowd counting by learning to rank. CVPR 2018, pp. 7661-7669. Snippet.
74. [NEW] Yang Y, Li G, Wu Z, Su L, Huang Q, Sebe N. Weakly-supervised crowd counting learns from sorting rather than locations. ECCV 2020. [doi](https://doi.org/10.1007/978-3-030-58598-3_1). Snippet.
75. [NEW] Lei Y, Liu Y, Zhang P, Liu L. Towards using count-level weak supervision for crowd counting. [arXiv:2003.00164](https://arxiv.org/abs/2003.00164) (journal version not verified). Snippet.
76. [NEW] von Borstel M, Kandemir M, Schmidt P, et al. Gaussian process density counting from weak supervision. ECCV 2016, LNCS 9905, pp. 365-380. Snippet. Density from region-level counts only.
77. [NEW] Xue Y, Ray N, Hugh J, Bigras G. Cell counting by regression using convolutional neural network. ECCV 2016 Workshops, pp. 274-290. [doi](https://doi.org/10.1007/978-3-319-46604-0_20). Snippet.
78. [NEW] He S, Minn KT, Solnica-Krezel L, Anastasio MA, Li H. Deeply-supervised density regression for automatic cell counting in microscopy images. Med Image Anal 2021;68:101892 (PMC7856299). Snippet; citation details from memory.
79. [NEW] Lu E, Xie W, Zisserman A. Class-agnostic counting. ACCV 2018. [arXiv:1811.00472](https://arxiv.org/abs/1811.00472). Snippet.
80. [NEW] Liu C, Zhong Y, Zisserman A, Xie W. CounTR: transformer-based generalised visual counting. BMVC 2022. [arXiv:2208.13721](https://arxiv.org/abs/2208.13721). Snippet.
81. [NEW] Djukic N, Lukezic A, Zavrtanik V, Kristan M. A low-shot object counting network with iterative prototype adaptation. ICCV 2023. Snippet.
82. [NEW] Amini-Naieni N, Han T, Zisserman A. CountGD: multi-modal open-world counting. NeurIPS 2024. [arXiv:2407.04619](https://arxiv.org/abs/2407.04619). Snippet.
83. [NEW] Pelhan J, Lukezic A, Zavrtanik V, Kristan M. A novel unified architecture for low-shot counting by detection and segmentation. NeurIPS 2024. [arXiv:2409.18686](https://arxiv.org/abs/2409.18686). Snippet.
84. [NEW] Knobel L, Han T, Asano YM. Learning to count without annotations. [arXiv:2307.08727](https://arxiv.org/abs/2307.08727) (venue not verified). Snippet.
85. [NEW] Zha K, Cao P, Son J, Yang Y, Katabi D. Rank-N-Contrast: learning continuous representations for regression. NeurIPS 2023. [arXiv:2210.01189](https://arxiv.org/abs/2210.01189). Abstract.
86. [NEW] Lathuiliere S, Mesejo P, Alameda-Pineda X, Horaud R. DeepGUM: learning deep robust regression with a Gaussian-uniform mixture model. ECCV 2018, pp. 205-221. [doi](https://doi.org/10.1007/978-3-030-01228-1_13). Snippet.
87. [NEW] Seitzer M, Tavakoli A, Antic D, Martius G. On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. ICLR 2022. [arXiv:2203.09168](https://arxiv.org/abs/2203.09168). Snippet.
88. [NEW] Xiong Y, Kim HJ, Singh V. Mixed effects neural networks (MeNets) with applications to gaze estimation. CVPR 2019. [doi](https://doi.org/10.1109/CVPR.2019.00793). Snippet.
89. [NEW] Simchoni G, Rosset S. Integrating random effects in deep neural networks. J Mach Learn Res 2023;24(156). [arXiv:2206.03314](https://arxiv.org/abs/2206.03314). Snippet.
90. [NEW] Nguyen KP, Treacher AH, Montillo AA. Adversarially-regularized mixed effects deep learning (ARMED) models improve interpretability, performance, and generalization on clustered (non-iid) data. IEEE Trans Pattern Anal Mach Intell 2023;45(7):8081-8093. [doi](https://doi.org/10.1109/TPAMI.2023.3234291). Abstract.
91. [NEW] PetFinder Pawpularity (Kaggle) first-place approach: frozen embeddings from many pretrained models, SVR head, hill-climbing selection. Secondary blog summary only (https://zenn.dev/kohecchi/articles/efc6d291dec289); primary write-up not retrieved.
92. [KG] Already recorded: liang2022transcrowd, lempitsky2010counting, xie2018counting, wang2020dmcount, wustefeld2023blobcount, gong2022ranksim, lee2026stochorder, yao2022cmixup, farebrother2024hlgauss, gao2017dldl, kendall2017uncertainties, kim2024confrag, tanno2019confusion, demyttenaere2016mape, gneiting2011point, tofallis2015logratio, delange2022mindgap, adachi2025ssa, kiet2026psc.

### Area 4. Label emulation

93. [NEW] Ray S, Page D. Multiple instance regression. ICML 2001, pp. 425-432. Snippet. Bag label determined by one "prime" instance.
94. Sources listed above that bear on emulation: Luznik 2019 (no. 35), Benetz 2019 (no. 29), Bohringer 2012 (no. 63), Visilant handoff 2026-09-14 (no. 67), von Borstel 2016 (no. 76).
95. [KG] Already recorded: benetz2024donorecd, kervadec2019constrained, takahashi1996konanpatent, patel2010flexcenter, qu2020pointnuclei.

**Count: 88 entries new to the KG (84 papers or preprints; 4 other public sources: nos. 67, 68, 69, 91).**

## 5. Searches run and sources not accessed

**Read first:** KG views (`papers-table`, `tried-and-closed`, `open-ideas-ranked`, `constraint-list`), about 20 KG idea/finding/experiment notes, and `final_summary_cd_20261113.md`.

**PubMed (12 queries, about 190 records screened, about 110 abstracts read):** specular microscopy + deep learning + cell density; agreement / repeatability + centre method / automated / Konan / CellChek; eye bank / donor cornea + variability / reading centre / validation; confocal or specular automated segmentation; Cornea Preservation Time Study / Specular Microscopy Ancillary Study + reading centre; eye-bank counting variability by named authors; fixed / variable frame, corner and centre methods + bias; automated vs manual comparisons; Ruggeri / Foracchia / Grisan / Fitzke / Scarpa; Doughty reliability papers; inter-/intra-observer + limits of agreement; flex-centre and calibration papers. One full text retrieved from PMC (Benetz 2019).

**Web search (about 40 queries):** CLEAR-EC challenge, write-ups, repositories, algorithm pages and the two related abstracts; 2024-2026 deep-learning papers on endothelium (segmentation-free, foundation models, SAM, Cellpose, self-supervision); eye-bank calibration and protocols; centre-method bias; technician consistency; Fourier / lattice density estimation; weakly supervised and class-agnostic counting; contrastive and robust regression; mixed-effects deep learning; multiple-instance regression; Kaggle image-regression solutions; annotator bias in regression.

**Consensus (3 queries; three papers returned per query on this plan).**

**Pages fetched:** challenge home page; eye-bank news post; `Visilant/CLEAR-EC` README, docs listing, five docs files and commit list; two Grand Challenge algorithm pages; PMC page of Luznik 2019.

**Not accessed or not found:**
- Phase II leaderboard (rendered by script; no rows returned). No public description of the top-three methods was found.
- AAO 2025 poster PO200 and the ARVO abstract named on the Zenodo record.
- Full texts of Lass 2005, Huang 2022, Rand 2017, Ohno 1999 and most pre-2015 Cornea papers (abstracts only).
- Jukic 2026 review full text (PMC returned a browser check).
- `eyebankingjournal.org/?p=231` (connection refused); a search snippet attributes "about 1-1.5 % input error of the centre method" to it, not verified.
- bioRxiv / medRxiv: the available tool lists preprints by date and category only, so it was not used; web search returned no relevant preprints.
- Google Scholar: no direct access; covered through web search and Consensus.
- The Kaggle first-place write-up (secondary summary only); DOI of Doke 2026; author list of SPIE 12930-129301A.

Consensus tool notice, reproduced as returned by the tool: "Create or connect a free Consensus account to return more than 3 results per search in Claude Code.: https://consensus.app/sign-up/?utm_source=claude_code&auth=claude_code"