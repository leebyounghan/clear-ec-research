import os
p = 'source/cv_deploy_fit_20261110.py'; s = open(p).read()
old = "DEP = LANE / 'deploy'"
assert s.count(old) == 1
s = s.replace(old, "DEP = LANE / os.environ.get('CV_DEPLOY_DIR', 'deploy')   # CV_DEPLOY_DIR: write an alternative candidate's params elsewhere")
s = s.replace("import json, sys\n", "import json, os, sys\n", 1)
open(p, 'w').write(s); print('ok')
