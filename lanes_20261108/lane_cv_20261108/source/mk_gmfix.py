p = 'source/cv_deploy_fit_20261110.py'; s = open(p).read()
old = "DEP = LANE / os.environ.get('CV_DEPLOY_DIR', 'deploy')   # CV_DEPLOY_DIR: write an alternative candidate's params elsewhere; GM = LANE / 'graph_models'"
assert s.count(old) == 1, 'pattern'
s = s.replace(old, "DEP = LANE / os.environ.get('CV_DEPLOY_DIR', 'deploy'); GM = LANE / 'graph_models'   # CV_DEPLOY_DIR: write an alternative candidate's params elsewhere")
open(p, 'w').write(s); print('fixed')
