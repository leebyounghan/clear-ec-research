"""Build the September 14 release report; preserve the September 10 PDF.

Run with Python + reportlab. No model loading, data access or GPU is required.
Content is an evidence-qualified summary of repository sources as of 2026-09-15.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/Lamda_KIM_CLEAR_EC_Technical_Report.pdf"
REPO = "https://github.com/leebyounghan/clear-ec-research"
DRIVE = "https://drive.google.com/drive/folders/1QyNtGhaEpePwBEyjj-UorqRkSFDRmKUn?usp=sharing"
NAVY, TEAL = colors.HexColor("#17324B"), colors.HexColor("#087F83")
INK, MUTED = colors.HexColor("#283B4C"), colors.HexColor("#5B6D7E")
WIDTH = 503.28
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=10.1,
                         leading=14.4, spaceAfter=9, textColor=INK))
styles.add(ParagraphStyle(name="SmallReport", parent=styles["BodyReport"], fontSize=8.6,
                         leading=11.6, spaceAfter=7))
styles.add(ParagraphStyle(name="CellReport", parent=styles["BodyReport"], fontSize=8.6,
                         leading=11.5, spaceAfter=0))
styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=8,
                         leading=11.4, spaceAfter=8, textColor=INK))
styles.add(ParagraphStyle(name="KickerReport", fontName="Helvetica-Bold", fontSize=8.4,
                         leading=11, spaceAfter=12, textColor=TEAL))
styles.add(ParagraphStyle(name="TitleReport", fontName="Helvetica-Bold", fontSize=27,
                         leading=32, spaceAfter=13, textColor=NAVY, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="PageReport", fontName="Helvetica-Bold", fontSize=21,
                         leading=26, spaceAfter=13, textColor=NAVY))
styles.add(ParagraphStyle(name="SubReport", fontName="Helvetica-Bold", fontSize=11.5,
                         leading=15, spaceBefore=5, spaceAfter=7, textColor=NAVY))
story = []


def p(text, small=False):
    story.append(Paragraph(text, styles["SmallReport" if small else "BodyReport"]))


def h(text):
    story.append(Paragraph(text, styles["SubReport"]))


def page(title, section, first=False):
    if story:
        story.append(PageBreak())
    story.append(Paragraph("LAMDA&amp;KIM  /  CLEAR-EC  /  " + section.upper(), styles["KickerReport"]))
    story.append(Paragraph(title, styles["TitleReport" if first else "PageReport"]))


def table(rows, widths):
    cells = [[Paragraph(str(x), styles["CellReport"]) for x in row] for row in rows]
    obj = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    obj.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5F1F0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), .8, TEAL),
        ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#D8E2E8")),
    ]))
    story.extend([obj, Spacer(1, 10)])


def link(url, label):
    return '<link href="' + escape(url, {'"': '&quot;'}) + '" color="#087F83">' + escape(label) + '</link>'


def source(path, label=None):
    return link(REPO + "/blob/master/" + path, label or path)


page("Native-resolution regression<br/>with domain-adapted features", "Phase I technical report", True)
p("<b>Team / algorithm display name: Lamda&amp;KIM</b><br/>"
  "September 14 release revision | Prepared 15 September 2026")
h("Summary")
p("We estimate cell density (CD), coefficient of variation (CV) and hexagonality (HEX) "
  "directly from corneal specular microscopy. Six Cellpose-SAM (CPSAM) regression members "
  "combine native-scale local views, different seeds and training objectives. Five members "
  "sample the central image field; a sixth uses spatially diverse, quality-ranked tiles "
  "from the full native image. No cell segmentation or manually specified ROI is required at inference.")
p("The latest delivered model changes <b>only the sixth member</b>: custom masked "
  "exponential-moving-average (EMA) feature pretraining on all 9,000 challenge training "
  "images, followed by regression on 8,998 valid original-label rows. The first five "
  "checkpoints, equal ensemble weights, calibration and Docker image are retained.")
table([
    ["Version", "Phase I score", "What changed"],
    ["September 10 baseline", "8.7042", "Six-member native-resolution ensemble; slide-averaged training targets."],
    ["September 14 verified result", "8.7010", "Official evaluation linked to the checksum-matched sixth-only SSL + raw-label package."],
], [126, 83, WIDTH-209])
p("Lower is better. The reported change is <b>-0.0032 score points (~0.037% relative)</b>, "
  "not evidence of a large SSL effect. Official results are listed separately from internal "
  "development scores throughout this report. [R1, R2]", True)
h("Scope and evidence boundary")
p("This revision describes the checksum-identified September 14 model package. The "
  "official 8.7010 evaluation, attached model/image versions and shared archive "
  "checksums have been verified. The scope is limited to this release and the "
  "completed development comparisons that informed its selection.")
p("<b>Reading guide:</b> method and training, pp. 2-3; results and limitations, p. 4; "
  "Docker reproduction, p. 5; experiment timeline and measured compute, pp. 6-7; "
  "artifact identities and evidence index, p. 8.", True)

page("1. Inference method", "Method")
h("Native scale and complementary fields of view")
p("Inputs are grayscale uint8 MHA images, historically 972 x 1296 pixels; the checked "
  "interface requires at least 512 x 512. We retain native 256 x 256 tiles instead of "
  "globally resizing the image. For each of the first five members, four corner tiles "
  "and one center tile come from the central 512 x 512 field.")
p("For the sixth member, a 4 x 6 full-field grid is ranked by an unlabeled heuristic: "
  "intensity standard deviation multiplied by the square root of Laplacian variance "
  "and the unsaturated fraction. Greedy selection enforces minimum origin separations "
  "of 192, then 128, then 0 pixels until eight distinct origins are available. The "
  "first five are used at inference. This is quality selection, not learned ROI localization.")
h("Preprocessing and regression network")
p("Each tile receives CLAHE (clip limit 2.0, 8 x 8 grid), then 1st/99th-percentile "
  "normalization and clipping to [0, 1]. The grayscale image is replicated into three "
  "channels. Each tile is evaluated under eight rotations/reflections (D8): "
  "<b>5 locations x 8 transforms = 40 views per member</b>.")
p("The CPSAM encoder supplies a 256-channel feature map. Spatial mean pooling and "
  "LayerNorm(256) feed three target-specific heads, each Linear(256,128), GELU, "
  "Dropout(0.2), Linear(128,1). Outputs are standardized log targets. The segmentation "
  "decoder is not used to derive CD, CV or HEX. [R3, R8]")
h("Aggregation order is part of the method")
table([
    ["Stage", "Operation"],
    ["Within member m", "Average 40 standardized-log prediction vectors to obtain z_m."],
    ["Decode each member", "y_m = exp(z_m * LS + LM), applied elementwise."],
    ["Across members", "y = c * (y_1 + ... + y_6) / 6, in physical-label units."],
], [124, WIDTH-124])
p("Fixed vectors, ordered CD / CV / HEX:<br/>"
  "LM = [7.91991997, -0.98247713, -0.62114316]<br/>"
  "LS = [0.18971863, 0.18813571, 0.27489457]<br/>"
  "c = [0.9970446110788349, 0.987972130802936, 0.9892234872488541]", True)
p("Full-precision constants are preserved in the frozen source/configuration. Members "
  "load strictly and sequentially; inference uses FP16 and batches eight views. CD is "
  "returned in cells/mm<super>2</super>, while CV and HEX remain fractions, not percentages.", True)

page("2. Training and SSL adaptation", "Method")
h("Data and target policy")
p("The raw archive contains 9,000 images. An ID audit preserved 8,954 exact matches "
  "and reconciled 46 separator-only aliases with one-to-one checks. Two rows with "
  "nonpositive metrics are excluded from final supervised fitting, leaving 8,998 rows. "
  "The retained first five models use slide-averaged positive labels. The replacement "
  "sixth uses original per-image labels without averaging. The 25 train_green ROI/marker "
  "references supported diagnostics, not ROI-supervised training. [R3, R4]")
table([
    ["Checkpoint / role", "Seed", "Crops", "Objective", "Updates"],
    ["seed42 / central", "42", "1", "Log L1", "3,576"],
    ["seed77 / central", "77", "1", "Log L1", "2,831"],
    ["seed2026 / central", "2026", "1", "Log L1", "2,980"],
    ["metric35 / central", "42", "1", "Mixed", "4,470"],
    ["predbag2_35 / central", "42", "2", "Log L1", "3,874"],
    ["full_quality / SSL + raw", "77", "2", "Mixed", "5,215"],
], [189, 42, 43, 80, WIDTH-354])
p("Supervised fitting: batch 48, AdamW, encoder LR 1e-5, head/norm LR 1e-3, "
  "weight decay 1e-4, BF16, activation checkpointing and gradient clipping at 1. "
  "The three baseline seeds use 200-update warmup then constant LR; the other recipes "
  "use warmup and a 5,215-update cosine schedule with a 0.05 floor. Two-crop models "
  "average crop predictions during training. Mixed loss is 0.5 log-L1 + "
  "0.5 relative-error / 0.2. The sixth is a fixed terminal fit, not validation-best selection.", True)
h("Custom masked EMA pretraining - only the sixth release member")
p("Starting from cpsam_v2, an online encoder predicts an EMA teacher's clean "
  "same-crop features at masked positions. The spatial predictor uses two 3 x 3 "
  "convolutions with GroupNorm/GELU. Block-mask targets are sampled between 40% and "
  "60% coverage; filling whole blocks can overshoot the target. Native tiles, D8 and "
  "mild intensity perturbations preserve cell scale; no resizing or external corpus is used.")
p("L_SSL = masked normalized-feature Smooth-L1 + 0.1 variance penalty + "
  "0.01 covariance penalty. AdamW uses LR 1e-5, weight decay 0.05, batch 32, "
  "200-update warmup/cosine decay, and 6,000 updates. Teacher EMA ramps from "
  "0.996 to 0.9999. All 9,000 image IDs were actually sampled without labels. "
  "The terminal teacher encoder initializes fresh regression heads. This is our "
  "custom feature-prediction objective with explicit encoder regularization. [R4, R5]", True)

page("3. Results and interpretation", "Evidence")
h("Official Phase I results")
p("The September 10 evaluation reports 8.7042; the September 14 "
  "evaluation confirms 8.7010. Their difference is -0.0032. These are challenge scores, "
  "not the internal values below. The comparison changes both the sixth member's "
  "initialization and label policy, so it does not isolate the causal effect of SSL. [R1, R2]")
h("Matched SSL + raw-label comparison")
p("The same 5,350 primary validation images from development folds 2/3/4 were "
  "used for all rows below. Lower uncalibrated mean MAPE is better. Each SSL fold "
  "excluded its validation images and duplicate-linked training slide groups. The "
  "fixed-six calculation changes only the sixth member, at weight 1/6. [R6]")
p("Per-metric MAPE = 100 x mean(|prediction - target| / target) over positive "
  "targets; the reported internal mean averages the three metric MAPEs.", True)
table([
    ["Sixth-member initialization", "Standalone", "Fixed six", "Delta vs raw"],
    ["Original CPSAM + raw labels", "9.271957", "9.029185", "Reference"],
    ["Standard SSL + raw, LR 1e-5", "9.222232", "9.019599", "-0.009586"],
    ["Standard SSL + raw, LR 5e-6", "9.229801", "9.021819", "-0.007366"],
], [236, 85, 85, WIDTH-406])
p("Standard-rate SSL improves the fixed ensemble in all three folds, but the "
  "gain is small: its standalone improvement (-0.049725) is diluted in the "
  "ensemble (-0.009586). The raw-only reference is <b>not</b> the September 10 "
  "slide-smoothed baseline. Do not subtract this table from official scores.")
h("Limits on generalization claims")
p("<b>Reused development folds.</b> Historical selection and calibration were "
  "exploratory, not fully nested. Later audits found exact duplicate-image overlap; "
  "new comparisons removed connected training groups and used a primary subset "
  "excluding affected historical validation groups. This improves control but does "
  "not create a pristine test set. Slide IDs are not verified patient IDs.")
p("<b>All-data fitting versus validation.</b> The delivered release uses all 9,000 "
  "training images for SSL and 8,998 valid rows for regression. It has no independent "
  "internal holdout score. Public foundation-model training overlap and near-duplicate "
  "or unrecorded patient overlap have not been newly audited.")
p("<b>Not a proven noise floor.</b> Correlated model errors and the small 25-image "
  "ROI diagnostic do not establish irreducible error. The measured improvement "
  "does not establish statistical significance, and no lower bound on achievable "
  "error is claimed.")

page("4. Reproducing release inference", "Reviewer execution")
p("The reviewer path runs inside the archived Docker image on a Linux x86_64 "
  "NVIDIA host. <b>No Kubernetes, private cluster configuration or host Python/venv "
  "is required.</b> The source, model archive and container are separate resources. [R7]")
table([
    ["Resource", "Location / role"],
    ["Private repository", link(REPO, "leebyounghan/clear-ec-research") + "<br/>Code, dependency pins, runner and evidence."],
    ["Artifact hosting", link(DRIVE, "Google Drive - model, Docker archive and SHA256SUMS.txt") + "<br/>Complete anonymous downloads and archive/resource hashes verified."],
    ["Execution guide", source("docker_review_20260914/README.md", "docker_review_20260914/README.md") + "<br/>Exact extraction, input layout, image loading and run commands."],
], [114, WIDTH-114])
h("Execution sequence")
p("1. Download the two archives and verify SHA256 before extraction/loading.<br/>"
  "2. Extract the model into a new directory and verify all eight resource hashes.<br/>"
  "3. Load the original image and confirm its exact image ID.<br/>"
  "4. Prepare one authorized MHA plus inputs.json and a new, writable output directory.<br/>"
  "5. From docker_review_20260914, execute:")
story.append(Paragraph("bash run.sh test_input checkpoints/model test_output 0", styles["CodeReport"]))
p("The runner mounts input/model read-only and output writable, disables networking, "
  "uses a read-only root filesystem and temporary /tmp, and runs as UID 10001. "
  "Only those directories are mounted. Outputs are cell-density.json, "
  "coefficient-of-variation.json and hexagonality.json.", True)
h("Verification performed and remaining checks")
table([
    ["Completed evidence", "Boundary"],
    ["Archive / eight resource hashes; strict checkpoint reload", "Identity and compatibility, not predictive accuracy."],
    ["Frozen CUDA 12.6 app, eight-image numerical check", "First-five outputs identical; maximum FP16/BF16 relative drift ~0.0651%. Not the actual archived Docker on T4."],
    ["Fresh checkout + archived Docker, RTX 4090", "One-image six-member inference passed; application 13.5 s, runner 45.4 s."],
    ["Fresh isolated venv + direct source, RTX 4090", "Install and inference passed; all three outputs exactly equal to Docker. Application 12.7 s, wrapper 17.8 s."],
], [239, WIDTH-239])
p("Host driver must support CUDA 12.6. The image contains Python 3.12.3 and "
  "PyTorch 2.11.0+cu126. Challenge allocation: T4 with 16 GiB VRAM and 16 GB RAM; "
  "4090 timings are not T4 measurements. Source venv used Python 3.12.7 and the same "
  "PyTorch/CUDA versions. The optional standalone training adapter removes old-run "
  "dependencies; import, workspace initialization and synthetic-loop checks passed. "
  "Full cache preparation and complete CPSAM refits have not been rerun. "
  "Guides: " + source("docs/VENV_SMOKE_20260915.md", "source venv") + " / " +
  source("training_reproduction/README.md", "training") + ".", True)

page("Appendix A. Experiment timeline", "Development before the September 10 release")
p("This timeline summarizes completed decisions and corrections, not every queued "
  "job. Calendar labels follow the dated research records; ranges cover work spanning "
  "days. Exact clock times elsewhere are UTC. Early scores used different cohorts "
  "and are not combined into a single progress curve.", True)
table([
    ["Period", "Question / action", "Outcome and release relevance"],
    ["Early research<br/>through Sep 1", "Direct regression; segmentation/FFT/point-based metrics; ROI diagnostics; external domain pretraining.", "Established regression and native multi-view baselines. Older claims of a proven noise floor were later withdrawn. These alternatives are not all release components. [T1]"],
    ["Sep 2-3", "Audit optimizer coverage, invalid labels, crop-bag train/eval consistency; run a 24-job follow-up queue.", "Corrected omitted LayerNorm optimization and uncleared gradients, zero-target handling and view-count mismatch. All 24 jobs completed; correction alone did not yield a material baseline gain. [T2]"],
    ["Sep 4-5", "Recover full 9,000-image raw archive; reconcile IDs; compare losses and prediction bags; verify initialization and inference.", "46 aliases reconciled; 8,998 supervised rows. Three central seeds plus mixed-loss and two-crop models form the first five members. Source/RNG differences retained for provenance. [T3]"],
    ["Sep 5-7", "Compare native full-uniform, quality-ranked and anchored crops; replicate seeds; test external-only SSL.", "Full-quality native region sampling added as the sixth member. It avoids resized global views and requires no manual ROI. Numerical ensemble integration completed. [T4]"],
    ["Sep 8", "Matched per-target specialization; mean versus standard-deviation residual pooling.", "Neither recipe passed its fixed ensemble gate. No claim that all specialists or distribution-aware pooling are ineffective. Existing six members retained. [T5]"],
    ["Sep 9-10", "Package model separately from Docker; frozen-source and runtime checks; Phase I submission.", "September 10 baseline scored 8.7042. This report preserves its image and first five weights; the latest sixth-only change is described separately. [R1, T6]"],
], [74, 205, WIDTH-279])
h("Why the trajectory matters")
p("The practical progression was not simply to enlarge the backbone. Data/ID "
  "integrity, training correctness, label policy, native-scale field coverage and "
  "ensemble complementarity each required separate tests. Negative screens are "
  "retained so that new hypotheses can be distinguished from repetitions.")
p("The original September 10 PDF is preserved as a historical snapshot. This "
  "revision adds the later evidence without rewriting completed experiments or "
  "implying that research candidates were submitted.", True)

page("Appendix A. Timeline continued", "Development of the September 14 release")
table([
    ["Period", "Experiment / milestone", "Result and disposition"],
    ["Sep 10-11", "Artifact-guard crop screen on folds 0/1; four fits.", "Fixed add7 mean 8.964352 vs current6 8.975950 on 3,582 rows. Improvement 0.011599 missed the 0.02 gate. Not promoted. [T7]"],
    ["Sep 11", "Duplicate-clean, native masked-EMA SSL versus fresh and continued-supervised controls.", "On 3,568 primary rows, fixed-six 8.975492 to 8.971051. Gain 0.004441 missed the 0.02 gate. The screen motivated better-controlled label comparisons, not immediate replacement. [T8]"],
    ["Sep 12-13", "Original versus slide-averaged label experiments across folds; raw-label release; matched SSL learning-rate comparison.", "Raw-only all-data sixth packaged separately. On 5,350 rows, standard SSL + raw fixed-six 9.019599 vs raw 9.029185; half LR did not beat standard overall. [R6, T9]"],
    ["Sep 14-15", "Final sixth-only standard SSL + raw release.", "SSL completed Sep 14 20:34 UTC; regression Sep 15 00:51; archive 01:02; local SHA verified 02:09. Team later supplied the Sep 14-dated official score 8.7010. Platform date and UTC process date differ. [R2, R4]"],
], [71, 186, WIDTH-257])
p("<b>Release boundary:</b> the terminal entry is the September 14 submission "
  "package. UTC completion after midnight does not denote another model version. "
  "The first five checkpoints and calibration remain unchanged.", True)
h("Selected measured compute - not total project cost")
table([
    ["Work", "Recorded GPU-hours", "Accounting basis"],
    ["September 10 six final fits", "~14.46", "Summed fit process time; excludes search/evaluation."],
    ["September 13 SSL + raw comparison", "33.077", "SSL + regression + evaluation, three-fold experiment."],
    ["Latest sixth release fit", "6.284", "(7,221.98 s SSL + 15,402.14 s regression) / 3,600; one H200."],
], [204, 92, WIDTH-296])
p("These scopes differ and should not be summed as a complete training budget or "
  "billing total. Queue wall time, useful GPU work and reserved device time are "
  "different quantities. [R3, R4, R6]", True)

page("Appendix B. Artifact and source index", "Audit trail")
h("Exact archive identities")
p("<b>Model:</b> model_full_quality77_ssl_raw_20260914.tar.gz<br/>"
  "7,964,665,157 bytes; contains six checkpoints, cpsam_v2 and candidate configuration.<br/>"
  "<b>Docker:</b> clear_ec_full_quality77_20260909.tar.gz<br/>"
  "4,064,075,652 bytes; unchanged archived inference image.", True)
for label, value in [
    ("Model archive SHA256", "e37c83db830cc5fa5215f7fd63267beb11e6c09a8778aefe2b8c6180a37a9730"),
    ("Docker archive SHA256", "cfa6c2c8a234a5aefc8cf0ff0a3413bbaf7a353fc3e131cd7cddf33434896c8a"),
    ("Docker image ID (sha256)", "275803d816ef934ef6665a12c693a2cd31bc1e418fdb734a7672a51404111a7d"),
]:
    p("<b>" + label + "</b><br/><font face='Courier' size='7.7'>" + value + "</font>", True)
p("Individual hashes: " + source("docker_review_20260914/MODEL_SHA256SUMS.txt", "MODEL_SHA256SUMS.txt") +
  ". Archive SHA and Docker image ID are different identities. Verified platform "
  "image ID: 730ff3f7-ea96-4180-936d-bc02a4eca762.<br/>"
  "Model ID: b4137c8d-a71a-4f83-ab1f-d0635f11590b, attached to the 8.7010 evaluation. "
  "Platform and recomputed Drive model CRC64-NVME both equal 65bbad82ffbb157e. " +
  source("docs/PLATFORM_BINDING_20260915.json", "Verification record") + ".", True)
h("Primary evidence and method references")
refs = [
    ("R1", link("https://clear-ec.grand-challenge.org/evaluation/0a82fe58-7f9a-4a88-b744-bcfa84b1e43a/", "September 10 official evaluation") + "; R2: " + link("https://clear-ec.grand-challenge.org/evaluation/7606a2b6-818f-46d0-a1fc-d060648fc2c6/", "September 14 verified evaluation")),
    ("R3", source("review_submission_20260910/TRAINING.md", "Historical recipes") + "; " + source("review_submission_20260910/app/final_candidate_core.py", "frozen model / normalization") + "; " + source("review_submission_20260910/app/region_inference_core.py", "regional inference")),
    ("R4-R5", source("docs/SSL_RELEASE_20260914.md", "SSL release log") + "; " + source("code/ssl_native_core.py", "custom masked EMA implementation")),
    ("R6", source("results/raw_ssl_next_20260913/ssl_audit.json", "Matched 5,350-row SSL aggregate audit")),
    ("R7", source("docker_review_20260914/README.md", "Docker execution guide and verification scope")),
    ("R8", link("https://github.com/MouseLand/cellpose", "Cellpose-SAM: Pachitariu, Rariden & Stringer (2025), upstream code / citation")),
]
for key, value in refs:
    p("<b>[" + key + "]</b> " + value, True)
h("Timeline source map")
p("[T1] " + source("docs/FINDINGS.md", "Findings (with later qualifications)") + "; " + source("docs/EXTERNAL_DATA.md", "external research inventory") +
  "<br/>[T2] " + source("docs/FOLLOWUP_20260902.md", "September 2 correction / validation") +
  "; [T3] " + source("docs/FINAL_CANDIDATE_20260904.md", "data / final candidate") +
  "<br/>[T4] " + source("docs/REGION_SEARCH_20260905.md", "native region search") + "; " + source("docs/REGION_SEEDS_20260906.md", "seed confirmation / integration") +
  "<br/>[T5] " + source("docs/SPECIALIZE_SCREEN_20260908.md", "specialists") + "; " + source("docs/STATRES_SCREEN_20260908.md", "residual pooling") +
  "; [T6] " + source("docs/PACKAGING_20260909.md", "packaging") +
  "<br/>[T7] " + source("docs/CROP_DINO_SCREEN_20260910.md", "artifact-guard crops") +
  "; [T8] " + source("docs/SSL_SCREEN_20260911.md", "fold-clean SSL screen") +
  "<br/>[T9] " + source("docs/TARGET_REPLICATION_20260912.md", "label replication") + "; " + source("docs/RAW_SSL_NEXT_20260913.md", "raw / SSL release and comparison") + ".", True)
p("Historical external corpora were investigated but are not part of the latest "
  "additional SSL stage. Foundation and external resources retain their upstream "
  "terms; raw challenge images, labels and per-case predictions are not distributed "
  "in this report. Sources describe recorded evidence as of 15 September, not a "
  "live status or certification of full retraining reproducibility.", True)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CEDBE2"))
    canvas.line(46, 39, 549.28, 39)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(46, 26, "Lamda&KIM | September 14 release | Revised 15 September 2026")
    canvas.drawRightString(549.28, 26, str(doc.page))
    canvas.restoreState()


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=(595.28, 841.89), rightMargin=46,
                            leftMargin=46, topMargin=45, bottomMargin=53,
                            title="Lamda&KIM - CLEAR-EC Phase I Technical Report",
                            author="Lamda&KIM", subject="September 14 release and experimental timeline")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)
