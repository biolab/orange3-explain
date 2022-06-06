import unittest
from unittest.mock import Mock
import pkg_resources

import numpy as np
from sklearn.inspection import permutation_importance, partial_dependence

from Orange.base import Model
from Orange.classification import NaiveBayesLearner, RandomForestLearner, \
    LogisticRegressionLearner, TreeLearner
from Orange.data import Table, Domain, DiscreteVariable
from Orange.data.table import DomainTransformationError
from Orange.evaluation import CA, MSE, AUC
from Orange.regression import RandomForestRegressionLearner, \
    TreeLearner as TreeRegressionLearner

from orangecontrib.explain.inspection import permutation_feature_importance, \
    _wrap_score, _check_model, individual_condition_expectation


def _permutation_feature_importance_skl(
        model: Model,
        data: Table,
        n_repeats: int = 5
) -> np.ndarray:
    return permutation_importance(model.skl_model,
                                  X=data.X, y=data.Y,
                                  n_repeats=n_repeats,
                                  random_state=0).importances


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table.from_file("iris")
        cls.housing = Table.from_file("housing")
        cls.titanic = Table.from_file("titanic")
        cls.heart = Table.from_file("heart_disease")
        cls.housing_missing = cls.housing.copy()
        cls.housing_missing.X[0, 0] = np.nan

    def test_check_model_cls_true(self):
        data = self.heart
        model = RandomForestLearner(random_state=0)(data)
        self.assertTrue(_check_model(model, data))

    def test_check_model_cls_false(self):
        data = self.iris
        model = RandomForestLearner(random_state=0)(data)
        self.assertFalse(_check_model(model, data))

    def test_check_model_reg_true(self):
        data = self.housing_missing
        model = RandomForestRegressionLearner(random_state=0)(data)
        self.assertTrue(_check_model(model, data))

    def test_check_model_reg_false(self):
        data = self.housing
        model = RandomForestRegressionLearner(random_state=0)(data)
        self.assertFalse(_check_model(model, data))

    def test_wrap_score_cls(self):
        data = self.heart
        model = RandomForestLearner(random_state=0)(data)
        scorer = _wrap_score(CA(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        mocked_model.assert_called_once()
        self.assertAlmostEqual(baseline_score, 0.987, 3)

    def test_wrap_score_predict_cls(self):
        data = self.titanic
        model = NaiveBayesLearner()(data)
        scorer = _wrap_score(CA(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        # mocked_model.assert_not_called()
        # mocked_model.predict.assert_called_once()
        self.assertAlmostEqual(baseline_score, 0.778, 3)

    def test_wrap_score_skl_predict_cls(self):
        data = self.iris
        model = RandomForestLearner(random_state=0)(data)
        scorer = _wrap_score(CA(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        mocked_model.assert_not_called()
        mocked_model.predict.assert_not_called()
        self.assertAlmostEqual(baseline_score, 0.993, 3)

    def test_wrap_score_reg(self):
        data = self.housing_missing
        model = RandomForestRegressionLearner(random_state=0)(data)
        scorer = _wrap_score(MSE(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        mocked_model.assert_called_once()
        self.assertAlmostEqual(baseline_score, 2, 0)

    def test_wrap_score_predict_reg(self):
        data = self.housing
        model = TreeRegressionLearner()(data)
        scorer = _wrap_score(MSE(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        # mocked_model.assert_not_called()
        # mocked_model.predict.assert_called_once()
        self.assertAlmostEqual(baseline_score, 0, 3)

    def test_wrap_score_skl_predict_reg(self):
        data = self.housing
        model = RandomForestRegressionLearner(random_state=0)(data)
        scorer = _wrap_score(MSE(), _check_model(model, data))

        mocked_model = Mock(wraps=model)
        baseline_score = scorer(mocked_model, data)
        mocked_model.assert_not_called()
        mocked_model.predict.assert_not_called()
        self.assertAlmostEqual(baseline_score, 2, 0)

    def test_remove_init_unlocked(self):
        """
        When this test starts to fail:
        - remove code in
        /Users/vesna/orange3-explain/orangecontrib/explain/__init__.py
        - remove this test
        - set minimum Orange version to 3.31.0
        """
        self.assertGreater(
            "3.35.0",
            pkg_resources.get_distribution("orange3").version
        )


class TestPermutationFeatureImportance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table.from_file("iris")
        cls.housing = Table.from_file("housing")
        cls.titanic = Table.from_file("titanic")
        cls.heart = Table.from_file("heart_disease")
        cls.n_repeats = 5

    def test_discrete_class(self):
        data = self.iris
        model = RandomForestLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, CA(), self.n_repeats)
        shape = len(data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

        mean = np.array([0.013333, 0, 0.322667, 0.474667])
        np.testing.assert_array_almost_equal(res[0].mean(axis=1), mean)

    def test_compare_to_skl(self):
        data = self.iris
        model = LogisticRegressionLearner()(data)
        res1 = _permutation_feature_importance_skl(model, data, self.n_repeats)
        res2 = permutation_feature_importance(model, data, CA(),
                                              self.n_repeats)
        np.testing.assert_array_equal(res1, res2[0])

    def test_retain_data(self):
        data = self.heart
        orig_X = data.X.copy()

        model = RandomForestLearner(random_state=0)(data)
        permutation_feature_importance(model, data, CA(), self.n_repeats)
        np.testing.assert_array_equal(data.X, orig_X)

    def test_discrete_attrs(self):
        data = self.titanic
        model = RandomForestLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, CA(), self.n_repeats)
        shape = len(data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

    def test_continuous_class(self):
        data = self.housing
        model = RandomForestRegressionLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, MSE(),
                                             self.n_repeats)
        shape = len(data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

    def test_missing_values(self):
        data = self.heart
        model = RandomForestLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, CA(), self.n_repeats)
        shape = len(data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

    def test_orange_models(self):
        data = self.heart
        n_repeats = self.n_repeats
        model = NaiveBayesLearner()(data)
        res = permutation_feature_importance(model, data, CA(), n_repeats)
        shape = len(data.domain.attributes), n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

        data = self.iris
        model = TreeLearner()(data)
        res = permutation_feature_importance(model, data, AUC(), n_repeats)
        shape = len(data.domain.attributes), n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

        data = self.housing
        model = TreeRegressionLearner()(data)
        res = permutation_feature_importance(model, data, MSE(), n_repeats)
        shape = len(data.domain.attributes), n_repeats
        self.assertEqual(res[0].shape, (shape))
        self.assertEqual(res[1], [a.name for a in data.domain.attributes])

    def test_auc(self):
        data = self.iris
        model = RandomForestLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, AUC(),
                                             self.n_repeats)
        self.assertAlmostEqual(res[0].mean(), 0.073, 3)

    def test_auc_missing_values(self):
        data = self.heart
        model = RandomForestLearner(random_state=0)(data)
        res = permutation_feature_importance(model, data, AUC(),
                                             self.n_repeats)
        self.assertAlmostEqual(res[0].mean(), 0.013, 3)

    def test_auc_orange_model(self):
        data = self.titanic
        model = NaiveBayesLearner()(data)
        res = permutation_feature_importance(model, data, AUC(),
                                             self.n_repeats)
        self.assertAlmostEqual(res[0].mean(), 0.044, 3)

    def test_inadequate_data(self):
        model = RandomForestLearner()(self.iris)
        args = model, self.titanic, self.n_repeats
        self.assertRaises(DomainTransformationError,
                          permutation_feature_importance, *args)

    def test_inadequate_data(self):
        domain = Domain([],
                        class_vars=self.iris.domain.class_vars,
                        metas=self.iris.domain.attributes)
        data = self.iris.transform(domain)
        model = RandomForestLearner()(self.iris)
        args = model, data, self.n_repeats
        self.assertRaises(ValueError, permutation_feature_importance, *args)

    def test_inadequate_model(self):
        model = RandomForestLearner()(self.iris)
        args = model, self.housing, self.n_repeats
        self.assertRaises(ValueError, permutation_feature_importance, *args)

    def test_sparse_data(self):
        sparse_data = self.heart.to_sparse()
        model = RandomForestLearner(random_state=0)(sparse_data)
        res = permutation_feature_importance(model, sparse_data,
                                             CA(), self.n_repeats)
        shape = len(sparse_data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(
            res[1], [a.name for a in sparse_data.domain.attributes]
        )

        sparse_data = self.iris.to_sparse()
        model = RandomForestLearner(random_state=0)(sparse_data)
        res = permutation_feature_importance(model, sparse_data,
                                             CA(), self.n_repeats)
        shape = len(sparse_data.domain.attributes), self.n_repeats
        self.assertEqual(res[0].shape, shape)
        self.assertEqual(
            res[1], [a.name for a in sparse_data.domain.attributes]
        )


class TestIndividualConditionalExpectation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iris = Table.from_file("iris")
        cls.heart = Table.from_file("heart_disease")
        cls.housing = Table.from_file("housing")

    def test_discrete_class(self):
        data = self.iris[:100]
        class_var = DiscreteVariable("iris", data.domain.class_var.values[:2])
        data = data.transform(Domain(data.domain.attributes, class_var))
        model = RandomForestLearner(n_estimators=10, random_state=0)(data)
        res = individual_condition_expectation(model, data, data.domain[0])
        self.assertIsInstance(res, dict)
        self.assertEqual(res["average"].shape, (2, 28))
        self.assertEqual(res["individual"].shape, (2, 100, 28))
        self.assertEqual(res["values"].shape, (28,))

    def test_discrete_class_result_values(self):
        data = self.iris[:100]
        class_var = DiscreteVariable("iris", data.domain.class_var.values[:2])
        data = data.transform(Domain(data.domain.attributes, class_var))
        model1 = RandomForestLearner(n_estimators=10, random_state=0)(data)

        data.Y = np.abs(data.Y - 1)
        model2 = RandomForestLearner(n_estimators=10, random_state=0)(data)

        res = individual_condition_expectation(model1, data, data.domain[0])
        dep1 = partial_dependence(model1.skl_model, data.X, [0], kind="both")
        dep2 = partial_dependence(model2.skl_model, data.X, [0], kind="both")
        np.testing.assert_array_almost_equal(
            res["average"][:1], dep2["average"])
        np.testing.assert_array_almost_equal(
            res["average"][1:], dep1["average"])
        np.testing.assert_array_almost_equal(
            res["individual"][:1], dep2["individual"])
        np.testing.assert_array_almost_equal(
            res["individual"][1:], dep1["individual"])

    def test_continuous_class(self):
        data = self.housing
        model = RandomForestRegressionLearner(n_estimators=10, random_state=0)(data)
        res = individual_condition_expectation(model, data, data.domain[0])
        self.assertIsInstance(res, dict)
        self.assertEqual(res["average"].shape, (1, 504))
        self.assertEqual(res["individual"].shape, (1, 506, 504))
        self.assertEqual(res["values"].shape, (504,))

    def test_multi_class(self):
        data = self.iris
        model = RandomForestLearner(n_estimators=10, random_state=0)(data)
        res = individual_condition_expectation(model, data, data.domain[0])
        self.assertIsInstance(res, dict)
        self.assertEqual(res["average"].shape, (3, 35))
        self.assertEqual(res["individual"].shape, (3, 150, 35))
        self.assertEqual(res["values"].shape, (35,))

    def test_mixed_features(self):
        data = self.heart
        model = RandomForestLearner(n_estimators=10, random_state=0)(data)
        res = individual_condition_expectation(model, data, data.domain[0])
        self.assertIsInstance(res, dict)
        self.assertEqual(res["average"].shape, (2, 41))
        self.assertEqual(res["individual"].shape, (2, 303, 41))
        self.assertEqual(res["values"].shape, (41,))

    def _test_sklearn(self):
        from matplotlib import pyplot as plt
        from sklearn.ensemble import RandomForestClassifier, \
            RandomForestRegressor
        from sklearn.inspection import PartialDependenceDisplay

        X = self.housing.X
        y = self.housing.Y
        model = RandomForestRegressor(random_state=0)

        # X = self.iris.X[:100]
        # y = self.iris.Y[:100]
        # y = np.abs(y - 1)
        # model = RandomForestClassifier(random_state=0)
        model.fit(X, y)
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [X.shape[1] - 1],
            target=0,
            kind="both",
            centered=True,
            subsample=1000,
            # grid_resolution=100,
            random_state=0,
        )

        plt.show()


if __name__ == "__main__":
    unittest.main()
