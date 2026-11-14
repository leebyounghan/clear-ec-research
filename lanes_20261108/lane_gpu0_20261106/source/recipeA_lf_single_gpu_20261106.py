"""Label-fraction learning curve on one GPU (lane GPU 0, 2026-11-06): recipeA_label_fraction_20261103 over the frozen runner, in the one-process runner. Flips only, so the 100 % point is the two-rank recipe A reference. LABEL_FRACTION from the environment, pinned by the wrapper."""
from pathlib import Path

import phase2_pooling_regression_20261019 as R
import recipeA_label_fraction_20261103 as LF
import recipeA_single_gpu_20261106 as SG

WRAPPER = Path(__file__).resolve()
_S = R.source_files
R.source_files = lambda: _S() + [WRAPPER]

if __name__ == '__main__':
    R.main()
