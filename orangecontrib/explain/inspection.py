""" Permutation feature importance for models. """
from typing import Callable, Tuple, Optional, Dict

import numpy as np
import scipy.sparse as sp
from sklearn.inspection import partial_dependence

from Orange.base import Model, SklModel
from Orange.classification import Model as ClsModel
from Orange.data import Table, Variable, DiscreteVariable
from Orange.evaluation import Results
from Orange.evaluation.scoring import Score, TargetScore, RegressionScore, R2
from Orange.regression import Model as RegModel
from Orange.util import dummy_callback, wrap_callback


def permutation_feature_importance(
        model: Model,
        data: Table,
        score: Score,
        n_repeats: int = 5,
        progress_callback: Callable = None
) -> np.ndarray:
    """
    Function calculates feature importance of a model for a given data.

    Parameters
    ----------
    model : Model
        Fitted Orange Learner.
    data : Table
        Data to calculate the feature importance for.
    score : Score
        Score to use for model evaluation.
    n_repeats : int, optional, default 5
        Number of times a feature is randomly shuffled.
    progress_callback : callable
        The callback for reporting the progress.

    Returns
    -------
    np.ndarray
         Feature importance.

    """
    if progress_callback is None:
        progress_callback = dummy_callback

    data = data.copy()
    _check_data(data)
    needs_pp = _check_model(model, data)

    scorer = _wrap_score(score, needs_pp)
    baseline_score = scorer(model, data)

    n_features = data.X.shape[1]
    step = 1 / n_features
    with data.unlocked():
        perm_scores = [_calculate_permutation_scores(
            model, data, i, n_repeats, scorer,
            wrap_callback(progress_callback, start=i * step,
                          end=(i + 1) * step)
        ) for i in range(n_features)]

    names = [attr.name for attr in data.domain.attributes]
    scores = baseline_score - np.array(perm_scores)
    if isinstance(score, RegressionScore) and not isinstance(score, R2):
        scores = -scores
    return scores, names


def _check_data(data: Table):
    if not data.domain.class_var:
        raise ValueError("Data with a target variable required.")
    if not data.domain.attributes:
        raise ValueError("Data with features required.")


def _check_model(model: Model, data: Table) -> bool:
    # return whether data.X and model_domain_data.X differ
    if data.domain.has_discrete_class and isinstance(model, RegModel):
        raise ValueError(
            f"{model} can not be used for data with discrete class."
        )
    elif data.domain.has_continuous_class and isinstance(model, ClsModel):
        raise ValueError(
            f"{model} can not be used for data with continuous class."
        )

    mod_data_X = model.data_to_model_domain(data).X
    if data.X.shape != mod_data_X.shape:
        return True
    elif sp.issparse(data.X) and sp.issparse(mod_data_X):
        return (data.X != mod_data_X).nnz != 0
    else:
        return (data.X != mod_data_X).any()


def _wrap_score(
        score: Score,
        needs_preprocessing: bool
) -> Callable:
    """
    Construct a scoring function based on `score`.

    Consider a `needs_preprocessing` flag to optimize the scoring procedure.
    When the flag is True the data transformation (onto model's domain)
    can be skipped when predicting.

    Parameters
    ----------
    score : Score
        Scoring metric.
    needs_preprocessing : bool
        True, if original_data.X and model_domain_data.X are not equal.

    Returns
    -------
    scorer : callable

    """

    def scorer(model: Model, data: Table) -> float:
        is_cls = data.domain.has_discrete_class
        if not needs_preprocessing and hasattr(model, "skl_model"):
            pred = model.skl_model.predict(data.X)
            if is_cls:
                prob = model.skl_model.predict_proba(data.X)
        # TODO - unify model.predict() output for all Models
        # elif not needs_preprocessing:
        #     pred = model.predict(data.X)
        #     if is_cls:
        #         assert isinstance(pred, tuple)
        #         pred, prob = pred
        else:
            if is_cls:
                pred, prob = model(data, ret=Model.ValueProbs)
            else:
                pred = model(data, ret=Model.Value)

        results = Results(data, domain=data.domain, actual=data.Y,
                          predicted=pred.reshape((1, len(data))))
        if is_cls:
            results.probabilities = prob.reshape((1,) + prob.shape)

        if isinstance(score, TargetScore):
            return score.compute_score(results, average="weighted")[0]
        else:
            return score.compute_score(results)[0]

    return scorer


def _calculate_permutation_scores(
        model: Model,
        data: Table,
        col_idx: int,
        n_repeats: int,
        scorer: Callable,
        progress_callback: Callable
) -> np.ndarray:
    random_state = np.random.RandomState(209652396)  # seed copied from sklearn

    x = data.X[:, col_idx].copy()
    shuffling_idx = np.arange(len(data))
    scores = np.zeros(n_repeats)

    for n_round in range(n_repeats):
        progress_callback(n_round / n_repeats)
        random_state.shuffle(shuffling_idx)
        data.X[:, col_idx] = data.X[shuffling_idx, col_idx]
        scores[n_round] = scorer(model, data)

    data.X[:, col_idx] = x

    progress_callback(1)
    return scores


def individual_condition_expectation(
        model: SklModel,
        data: Table,
        feature: Variable,
        grid_resolution: int = 1000,
        kind: str = "both",
        progress_callback: Callable = dummy_callback
) -> Dict[str, np.ndarray]:
    progress_callback(0)
    _check_data(data)
    needs_pp = _check_model(model, data)
    if needs_pp:
        data = model.data_to_model_domain(data)

    assert feature.name in [a.name for a in data.domain.attributes]
    feature_index = data.domain.index(feature.name)

    assert isinstance(model, SklModel), f"Model ({model}) is not supported."
    progress_callback(0.1)

    dep = partial_dependence(model.skl_model,
                             data.X,
                             [feature_index],
                             grid_resolution=grid_resolution,
                             kind=kind)

    results = {"average": dep["average"], "values": dep["values"][0]}
    if kind == "both":
        results["individual"] = dep["individual"]

    if data.domain.has_discrete_class and \
            len(data.domain.class_var.values) == 2:
        results = {"average": np.vstack([1 - dep["average"], dep["average"]]),
                   "values": dep["values"][0]}
        if kind == "both":
            results["individual"] = \
                np.vstack([1 - dep["individual"], dep["individual"]])

    progress_callback(1)

    return results
