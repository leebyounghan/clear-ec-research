"""Pure target transformation and fixed decision rules. No image I/O."""
import numpy as np


def targets_for_training(y, slides, train, policy):
    y = np.asarray(y)
    train = np.asarray(train)
    if policy not in ('raw', 'slidesmooth'):
        raise ValueError('Unknown target policy')
    if y.ndim != 2 or y.shape[1] != 3 or len(slides) != len(y):
        raise ValueError('Invalid metadata shape')
    if len(train) != len(np.unique(train)) or not len(train):
        raise ValueError('Training indices must be unique and nonempty')
    if not np.isfinite(y[train]).all() or not (y[train] > 0).all():
        raise ValueError('Invalid training targets')
    out = y.copy()
    if policy == 'slidesmooth':
        for slide in np.unique(slides[train]):
            ix = train[slides[train] == slide]
            if len(ix) > 1:
                out[ix] = y[ix].mean(0)
    return out


def summarize_target_shift(y, slides, train):
    smoothed = targets_for_training(y, slides, train, 'slidesmooth')
    difference = abs(smoothed[train] - y[train])
    relative = difference / y[train] * 100
    _, sizes = np.unique(slides[train], return_counts=True)
    return dict(train_rows=len(train), slide_groups=len(sizes),
                multi_image_groups=int((sizes > 1).sum()),
                changed_rows=int((difference > 1e-6).any(1).sum()),
                mean_relative_target_shift=relative.mean(0).tolist(),
                p95_relative_target_shift=np.percentile(relative, 95, axis=0).tolist(),
                max_relative_target_shift=relative.max(0).tolist())


def decision(aggregate, folds):
    # Research eligibility and submission eligibility are deliberately separate.
    research = aggregate['standalone_vs_matched'] <= -.02 and all(
        f['standalone_vs_matched'] < 0 for f in folds)
    ensemble = aggregate['replace6_vs_current6'] <= -.02 and all(
        f['replace6_vs_current6'] < 0 and f['replace6_vs_matched'] < 0 for f in folds)
    return dict(research_signal=bool(research), passes_ensemble_screen=bool(ensemble),
                deployment_approved=False)
