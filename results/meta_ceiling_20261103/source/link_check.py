import glob, os, re
files = ['review_submission_20260910/TRAINING.md', 'review_submission_20260910/checkpoints/README.md', 'README.md', 'results/README.md', 'code/README.md', 'docker_review_20260914/README.md', 'training_reproduction/README.md', 'review_submission_20260910/RELEASE_CHECKLIST.md', 'results/meta_ceiling_20261103/REPORT.md'] + glob.glob('docs/*.md') + glob.glob('docs/legacy_phase1/*.md')
bad = 0; n = 0
for f in files:
    t = open(f, errors='ignore').read()
    for m in re.finditer(r'\]\(([^)\s]+)\)', t):
        u = m.group(1)
        if re.match(r'[a-z]+://|#|mailto:', u): continue
        p = os.path.normpath(os.path.join(os.path.dirname(f), u.split('#')[0])); n += 1
        if not os.path.exists(p): bad += 1; print('BROKEN', f, u[:90])
print('links', n, 'broken', bad)
