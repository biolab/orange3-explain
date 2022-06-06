# pylint: disable=missing-docstring
import unittest

from orangewidget.tests.base import WidgetTest
from Orange.classification import RandomForestLearner
from Orange.data import Table
from Orange.regression import RandomForestRegressionLearner
from orangecontrib.explain.widgets.owice import OWICE


class TestOWICE(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        cls.titanic = Table("titanic")
        kwargs = {"random_state": 0}
        cls.rf_cls = RandomForestLearner(**kwargs)(cls.heart)
        cls.rf_reg = RandomForestRegressionLearner(**kwargs)(cls.housing)

    def setUp(self):
        self.widget = self.create_widget(OWICE)

    def test_input_cls(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.unknown_err.is_shown())

        self.send_signal(self.widget.Inputs.model, None)
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_discrete_features(self):
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

    def test_order_features(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)

        model = self.widget._features_view.model()
        model_data = [model.data(model.index(i, 0))
                      for i in range(model.rowCount())]
        attrs = self.heart.domain.attributes
        cont_var_names = [a.name for a in attrs if a.is_continuous]
        self.assertEqual(model_data, cont_var_names)

        self.widget.controls.order_by_importance.setChecked(True)
        model_data = [model.data(model.index(i, 0))
                      for i in range(model.rowCount())]
        cont_var_names = ["max HR", "ST by exercise", "cholesterol",
                          "age", "rest SBP", "major vessels colored"]
        self.assertEqual(model_data, cont_var_names)

    def test_sample_data(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:1])
        self.assertTrue(self.widget.Error.not_enough_data.is_shown())
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertTrue(self.widget.Information.data_sampled.is_shown())
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
