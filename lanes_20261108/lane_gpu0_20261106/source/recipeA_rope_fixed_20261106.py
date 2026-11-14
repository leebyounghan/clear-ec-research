"""Recipe A with the random RoPE coordinate rescale off during regression training (lane GPU 0, 2026-11-06).
The encoder draws a rescale in [1/2, 2] at every train-mode forward and validates at 1, a setting nobody chose for this task: it comes from the DINOv3 hub default. A stored member's score depends on the scale at inference (0.707 +0.20, 1.414 -0.07 on folds 0 and 4 of seed 46, -0.02 over five folds and two seeds).
Everything else is the frozen runner in the one-process form, flips only. The SSL encoder is unchanged (it trained with the rescale on). Validation is at scale 1 as before."""
from pathlib import Path

import phase2_pooling_regression_20261019 as R
import recipeA_single_gpu_20261106 as SG

WRAPPER = Path(__file__).resolve()
_S = R.source_files
_SAVE = R.save
_SETUP = R.setup


def setup(m):
    out = _SETUP(m)
    rope = out[0][0].original.rope_embed
    assert rope.rescale_coords == 2
    rope.rescale_coords = None
    return out


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['rope_rescale_train'] = None
    return _SAVE(path, obj)


R.setup = setup
R.save = save
R.source_files = lambda: _S() + [WRAPPER]

if __name__ == '__main__':
    R.main()
