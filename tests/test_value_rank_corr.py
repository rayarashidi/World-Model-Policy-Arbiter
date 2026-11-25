import numpy as np
import pytest

from evaluation.diagnostics import spearman


def test_tied_values():
    targets = np.array([0, 0, 1, 1])
    values = np.array([0, 1, 0, 1])
    assert spearman(targets, values) == pytest.approx(0.0)


def test_permutation_invariance():
    targets = np.array([0, 0, 1, 1])
    values = np.array([0, 1, 0, 1])
    order = [1, 0, 2, 3]
    assert spearman(targets[order], values[order]) == pytest.approx(
        spearman(targets, values)
    )


def test_constant_values():
    assert np.isnan(spearman([0, 0, 1, 1], [5, 5, 5, 5]))


def test_perfect_positive_correlation():
    assert spearman([0, 0, 1, 2], [1, 1, 3, 4]) == pytest.approx(1.0)


def test_perfect_negative_correlation():
    assert spearman([0, 0, 1, 2], [4, 4, 3, 1]) == pytest.approx(-1.0)


def test_training_rank_metric():
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_world_model.py"
    )

    tree = ast.parse(path.read_text())

    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(t)
            == "metrics['validation/value_rank_corr']"
            for t in node.targets
        )
    )

    namespace = {
        "metrics": {},
        "np": np,
        "spearman": spearman,
        "tg": np.array([0, 0, 1, 1]),
        "lg": np.array([0, 1, 0, 1]),
        "r_t": np.arange(4),
        "r_l": np.array([0, 2, 1, 3]),
    }

    exec(
        compile(
            ast.Module(body=[assignment], type_ignores=[]),
            str(path),
            "exec"
        ),
        namespace
    )

    assert namespace["metrics"][
        "validation/value_rank_corr"
    ] == pytest.approx(0.0)
