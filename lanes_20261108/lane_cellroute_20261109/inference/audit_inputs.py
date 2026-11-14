"""Audit (cell-route lane, 2026-11-10): which code on, or needed by, the new-recipe inference path reads labels, fold ids, split files or OOF-only files.
AST scan: (a) module-level statements (run at import) that load files, and (b) function bodies named in FUNCS, flagged when the statement mentions
label / fold / OOF patterns. Prints file:line and the statement head; a human reads the list (patterns over-report)."""
import ast, re, sys
R = '/workspace/clear-ec-research/'
PAT = re.compile(r"(oof|hybrid_oof|load_base|\bY\b|\['Y'\]|target|\by\[|fold|split|preflight/fold|validation|seen_leaked|green_features|REF_BEST|BEST\b|GRAPH\b|best_oof|predictions_|stack5|_5f\.npz|chain_cvw3|freeze_cd_oof|hex_inputs_oof|image_stats\.npz|seg_all|legible_window\.npz|tta_peaks|feats/)", re.I)
LOADS = re.compile(r"(np\.load|json\.load|memmap|open\(|load_base|pd\.read|torch\.load|loadtxt|np\.fromfile)")
FILES = {  # file: functions used at inference (None = module import only)
 'experiments/lane_cellroute_20261109/inference/members.py': ['preprocess', '_inputs', '_build', 'read_logs', 'recipeA_tta', 'recipeA_cd', 'dino_cd', 'dino_cv', 'vitl_cd'],
 'experiments/lane_cellroute_20261109/inference/recipe_infer.py': ['image_stats', 'seg', 'p1', 'peaks', 'predict'],
 'experiments/lane_cellroute_20261109/inference/seg_run_feats.py': ['feats'],
 'experiments/lane_cellroute_20261109/source/deploy_hexkonan.py': ['windows', 'features', 'design', 'correct', 'seg_points'],
 'experiments/lane_cellroute_20261109/source/deploy_hexkonan_image.py': ['load_detector', 'tta_heat'],
 'experiments/lane_cellroute_20261109/source/cr_lib.py': ['konan_cells_idx', 'vote', 'neighbours'],
 'experiments/lane_cd_20261108/source/cd_combine_20261110.py': ['combine'],
 'experiments/lane_hex_20261108/source/hex_apply.py': ['apply', 'ridge', 'deploy_steps'],
 'experiments/lane_cd_20261108/source/family_cd_20261108.py': ['build_tokens'],
 'experiments/lane_cd_20261108/source/family_bb_cd_20261108.py': ['build_tokens'],
 'experiments/green_track_20261028/source/green_point_features_20261027.py': ['flip_image'],
 'experiments/green_track_20261028/source/green_sg2_specialist_20261028.py': ['heat', 'peaks2', 'frame_support'],
 'experiments/green_track_20261028/deploy/hex6_correction.py': ['preprocess', 'feature', 'hex6_per_detector', 'hex6_all'],
 'experiments/submission1_test_20261030/source/phase1_app/final_candidate_core.py': ['build_model', 'central512', 'predict_one', 'eval_views', 'normalize'],
 'experiments/submission1_test_20261030/source/phase1_app/base_inference.py': ['load_strict'],
 # needed by the recipe, not yet usable per image:
 'experiments/lane_hex_20261108/source/feat_prior.py': None, 'experiments/lane_hex_20261108/source/feat_ridge.py': None, 'experiments/lane_hex_20261108/source/feat_fuse.py': None,
 'experiments/lane_cv_20261108/source/graph_train_cv_20261108.py': None, 'experiments/lane_cv_20261108/source/graph_order_cv_20261110.py': None,
 'experiments/lane_cv_20261108/source/graph_prep_sp_20261108.py': None, 'experiments/lane_cv_20261108/source/tta_infer_20261108.py': None,
 'experiments/lane_cv_20261108/source/cv_stack5_20261108.py': None, 'experiments/lane_cv_20261108/source/token_stats_cv_20261108.py': None,
 'experiments/meta_ceiling_20261103/source/cell_graph_sweep3_20261105.py': None, 'experiments/meta_ceiling_20261103/source/sim_embed_eval_20261105.py': None,
 'experiments/meta_ceiling_20261103/source/legible_window_features_20261105.py': None, 'experiments/lane_hex_20261108/source/gnn_hex.py': None,
}
for f, funcs in FILES.items():
    try: src = open(R + f).read(); tree = ast.parse(src)
    except FileNotFoundError: print(f'## {f}: NOT FOUND'); continue
    lines = src.splitlines(); hits = []
    for node in tree.body:   # module level
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(getattr(node, 'value', None), ast.Constant)): continue
        seg = ast.get_source_segment(src, node) or ''
        if LOADS.search(seg) and PAT.search(seg): hits.append(('import', node.lineno, seg.splitlines()[0][:170]))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and funcs and node.name in funcs:
            for sub in ast.walk(node):
                if isinstance(sub, ast.stmt) and not isinstance(sub, (ast.FunctionDef, ast.If, ast.For, ast.With, ast.While, ast.Try)):
                    seg = ast.get_source_segment(src, sub) or ''
                    if LOADS.search(seg) and PAT.search(seg): hits.append((f'in {node.name}()', sub.lineno, seg.splitlines()[0][:170]))
    print(f'## {f}: {len(hits)} flagged' + ('' if funcs else ' (import-time only scanned; module not yet usable per image)'))
    for w, ln, s in sorted(set(hits), key=lambda h: h[1]): print(f'   {w:>16} :{ln}  {s}')
