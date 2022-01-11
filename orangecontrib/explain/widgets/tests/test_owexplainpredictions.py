# pylint: disable=missing-docstring
import unittest
from typing import Type

from Orange.base import Learner
from Orange.classification import RandomForestLearner, CalibratedLearner, \
    ThresholdLearner
from Orange.data import Table
from Orange.regression import RandomForestRegressionLearner
from Orange.tests.test_classification import all_learners as all_cls_learners
from Orange.tests.test_regression import all_learners as all_reg_learners, \
    init_learner as init_reg_learner
from orangecontrib.explain.widgets.owexplainpredictions import ForcePlot, \
    OWExplainPredictions
from orangewidget.tests.base import WidgetTest


def init_learner(learner: Type[Learner], table: Table) -> Learner:
    if learner in (CalibratedLearner, ThresholdLearner):
        return CalibratedLearner(RandomForestLearner())
    return init_reg_learner(learner, table)


class TestOWExplainPredictions(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        kwargs = {"random_state": 0}
        cls.rf_cls = RandomForestLearner(**kwargs)(cls.heart)
        cls.rf_reg = RandomForestRegressionLearner(**kwargs)(cls.housing)

    def setUp(self):
        self.widget = self.create_widget(OWExplainPredictions)

    def test_classification_data_classification_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget.graph)

    def test_classification_data_regression_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.graph)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_regression_data_regression_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget.graph)

    def test_regression_data_classification_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.data, self.housing[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.graph)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_target_combo(self):
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._target_combo.currentText(), "0")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._target_combo.currentText(), "0")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.model, None)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

    def test_order_combo(self):
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Original instance ordering")
        self.assertEqual(self.widget._order_combo.count(), 3)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Original instance ordering")
        # 1 separator
        self.assertEqual(self.widget._order_combo.count(),
                         len(self.rf_cls.domain.attributes) + 1 +
                         len(self.widget.ORDERS))

        self.send_signal(self.widget.Inputs.model, None)
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Original instance ordering")
        self.assertEqual(self.widget._order_combo.count(), 3)

    def test_annotation_combo(self):
        self.assertEqual(self.widget._annot_combo.currentText(), "Enumeration")
        self.assertEqual(self.widget._annot_combo.count(), 1)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._annot_combo.currentText(), "Enumeration")
        # 2 separators
        self.assertEqual(self.widget._annot_combo.count(),
                         len(self.rf_reg.domain) + 2 +
                         len(self.widget.ANNOTATIONS))

        self.send_signal(self.widget.Inputs.model, None)
        self.assertEqual(self.widget._annot_combo.currentText(), "Enumeration")
        self.assertEqual(self.widget._annot_combo.count(), 1)

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self.heart)
        self.assertPlotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertPlotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget.graph)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertPlotEmpty(self.widget.graph)

    def test_plot_tooltip(self):
        self.assertEqual(True, False)

    def test_plot_zoom_select(self):
        self.assertEqual(True, False)

    def test_plot_multiple_selection(self):
        self.assertEqual(True, False)

    def test_all_models(self):
        def run(data):
            self.send_signal(self.widget.Inputs.background_data, data)
            self.send_signal(self.widget.Inputs.data, data)
            model = init_learner(learner, data)(data)
            self.send_signal(self.widget.Inputs.model, model)
            self.wait_until_finished(timeout=50000)

        for learner in all_reg_learners():
            run(self.housing[::4])
        for learner in all_cls_learners():
            run(self.heart[::4])

    def test_output_scores(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertEqual(True, False)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_output_selection(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(True, False)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_output_data(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(True, False)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_saved_workflow(self):
        self.assertEqual(True, False)

    def test_visual_settings(self):
        self.assertEqual(True, False)

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.widget.send_report()

    def assertPlotNotEmpty(self, plot: ForcePlot):
        self.assertGreater(len(plot.plotItem.items), 0)

    def assertPlotEmpty(self, plot: ForcePlot):
        self.assertEqual(len(plot.plotItem.items), 0)


if __name__ == "__main__":
    unittest.main()
