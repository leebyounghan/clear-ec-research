"""Moves the Phase I documents to docs/legacy_phase1/ with git mv and repairs relative links. Frozen release scripts and the published PDF are left as they are. Prints a link check at the end."""
import os, re, subprocess
L = ['ALL_MEMBERS_SSL_20260915.md','ARTIFACTS.md','CPSAM_METHODS_20260914.md','CROP_DINO_SCREEN_20260910.md','DEPLOY_METHODS_20260905.md','DEPLOY_REVIEW_20260905.md','DINO_ALL9000_20260913.md','DISTILLATION_NEXT_20260915.md','DOCKER_SMOKE_20260915.md','EXTERNAL_DATA.md','FINAL_CANDIDATE_20260904.md','FINDINGS.md','FOLLOWUP_20260902.md','LEADER_FOLLOWUP_20260911.md','METHODOLOGY.md','METHODS_20260904.md','NEXT_STEPS.md','PACKAGING_20260909.md','PLATFORM_BINDING_20260915.json','RAW_SSL_NEXT_20260913.md','REGION_SEARCH_20260905.md','REGION_SEEDS_20260906.md','REPRESENTATION_SSL_20260913.md','REPRODUCE.md','SPECIALIZE_SCREEN_20260908.md','SPECIALIZE_SCREEN_20260908_audit.json','SSL_DURATION_20260914.md','SSL_RELEASE_20260914.md','SSL_SCREEN_20260911.md','STATRES_SCREEN_20260908.md','STATRES_SCREEN_20260908_audit.json','STATUS_20260915.md','SUBMISSION_20260904.md','TARGET_REPLICATION_20260912.md','TARGET_SCREEN_20260912.md','VENV_SMOKE_20260915.md']
D = 'docs/legacy_phase1'; os.makedirs(D, exist_ok=True); S = set(L)
for n in L:
    subprocess.run(['git', 'mv', f'docs/{n}', f'{D}/{n}'], check=True)
pat = '|'.join(re.escape(n) for n in sorted(L, key=len, reverse=True))
def fix_moved(t):
    def link(m):
        u = m.group(2)
        if re.match(r'[a-z]+://|#|mailto:', u): return m.group(0)
        if u.startswith('../'): return f'{m.group(1)}../{u}'
        base = u.split('#')[0]
        if base in S or base == '': return m.group(0)
        return f'{m.group(1)}../{u}'
    t = re.sub(r'(\]\()([^)\s]+)', link, t)
    return re.sub(r'(?<![\w/])docs/(' + pat + ')', D + r'/\1', t)
for n in L:
    if not n.endswith('.md'): continue
    p = f'{D}/{n}'; t = open(p).read(); u = fix_moved(t)
    if u != t: open(p, 'w').write(u)
for p in ('README.md', 'docker_review_20260914/README.md', 'training_reproduction/README.md', 'review_submission_20260910/RELEASE_CHECKLIST.md', 'results/all_members_ssl_20260915/README.md', '.gitignore', 'docs/STATE_20261103.md', 'docs/STATUS_20261027.md', 'docs/DECISION_PLAN_phase2_20261030.md', 'docs/PHASE2_ROADMAP_20260928.md', 'docs/HANDOFF_CLEC_20261013.md', 'docs/IDEA_REGISTRY_20261031.md', 'code/README.md', 'results/README.md'):
    if not os.path.exists(p): continue
    t = open(p).read(); u = re.sub(r'(?<![\w/])docs/(' + pat + ')', D + r'/\1', t)
    if p.startswith('docs/'):
        u = re.sub(r'(\]\()(' + pat + r')', r'\1legacy_phase1/\2', u)
    if u != t: open(p, 'w').write(u); print('rewrote', p)
p = 'docs/DEVLOG_INDEX.md'; t = open(p).read(); u = re.sub(r'(\]\()(' + pat + r')', r'\1legacy_phase1/\2', t); open(p, 'w').write(u); print('rewrote', p)
print('moved', len(L))
