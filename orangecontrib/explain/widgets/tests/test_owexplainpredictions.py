# pylint: disable=missing-docstring
import unittest
from typing import Type
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import QPointF, Qt
from AnyQt.QtGui import QFont
from AnyQt.QtWidgets import QToolTip

from Orange.base import Learner
from Orange.classification import RandomForestLearner, CalibratedLearner, \
    ThresholdLearner
from Orange.data import Table
from Orange.regression import RandomForestRegressionLearner
from Orange.tests.test_classification import all_learners as all_cls_learners
from Orange.tests.test_regression import all_learners as all_reg_learners
from Orange.widgets.tests.utils import simulate
from orangecontrib.explain.explainer import INSTANCE_ORDERINGS
from orangecontrib.explain.widgets.owexplainpredictions import ForcePlot, \
    OWExplainPredictions
from orangewidget.tests.base import WidgetTest

# FIXME: remove when the minimum supported version is 3.32
try:
    from Orange.tests.test_regression import init_learner as init_reg_learner
except ImportError:
    from Orange.regression import CurveFitLearner


    def init_reg_learner(learner, table):
        if learner == CurveFitLearner:
            return CurveFitLearner(
                lambda x, a: x[:, -1] * a, [],
                [table.domain.attributes[-1].name]
            )
        return learner()


def init_learner(learner: Type[Learner], table: Table) -> Learner:
    if learner in (CalibratedLearner, ThresholdLearner):
        return CalibratedLearner(RandomForestLearner())
    return init_reg_learner(learner, table)


class TestForcePlot(WidgetTest):
    def setUp(self):
        widget = self.create_widget(OWExplainPredictions)
        self.plot = ForcePlot(widget)
        self.housing = Table("housing")

    def test_zoom_select(self):
        self.plot.select_button_clicked()
        self.plot.pan_button_clicked()
        self.plot.zoom_button_clicked()
        self.plot.reset_button_clicked()

    def test_selection(self):
        selection_handler = Mock()

        view_box = self.plot.getViewBox()
        self.plot.selectionChanged.connect(selection_handler)

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(10, 0)
        event.pos.return_value = QPointF(30, 0)
        event.isFinish.return_value = True

        # select before data is sent
        view_box.mouseDragEvent(event)

        # set data
        x_data = np.arange(5)
        pos_y_data = [(np.arange(5) - 1, np.arange(5))]
        neg_y_data = [(np.arange(5), np.arange(5) + 1)]
        labels = [a.name for a in self.housing.domain.attributes]
        self.plot.set_data(x_data, pos_y_data, neg_y_data, "", "",
                           labels, labels, self.housing)

        # select after data is sent
        view_box.mouseDragEvent(event)
        selection_handler.assert_called_once()

        # select other instances
        event.buttonDownPos.return_value = QPointF(40, 0)
        event.pos.return_value = QPointF(60, 0)

        selection_handler.reset_mock()
        view_box.mouseDragEvent(event)
        selection_handler.assert_called_once()
        self.assertEqual(len(selection_handler.call_args[0][0]), 2)

        # click on the plot resets selection
        selection_handler.reset_mock()
        view_box.mouseClickEvent(event)
        selection_handler.assert_called_once()
        self.assertEqual(len(selection_handler.call_args[0][0]), 0)


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

    def test_input_one_instance(self):
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:1])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        self.assertTrue(self.widget.Error.not_enough_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_data.is_shown())

    def test_input_too_many_instances(self):
        titanic = Table("titanic")
        model = RandomForestLearner(random_state=0)(titanic)
        self.send_signal(self.widget.Inputs.background_data, titanic)
        self.send_signal(self.widget.Inputs.data, titanic)
        self.send_signal(self.widget.Inputs.model, model)
        self.wait_until_finished()
        self.assertTrue(self.widget.Information.data_sampled.is_shown())

        output = self.get_output(self.widget.Outputs.scores)
        self.assertEqual(len(output), 1000)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

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
        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotNotEmpty(self.widget.graph)

    def test_regression_data_classification_model(self):
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.graph)
        self.assertTrue(self.widget.Error.domain_transform_err.is_shown())

    def test_target_combo(self):
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.heart[:5])
        self.assertEqual(self.widget._target_combo.currentText(), "0")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.housing[:5])
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.heart[:5])
        self.assertEqual(self.widget._target_combo.currentText(), "0")
        self.assertTrue(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.housing[:5])
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertFalse(self.widget._target_combo.isEnabled())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget._target_combo.currentText(), "")
        self.assertTrue(self.widget._target_combo.isEnabled())

    def test_order_combo(self):
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Order instances by similarity")
        self.assertEqual(self.widget._order_combo.count(), 3)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Order instances by similarity")
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        # 2 separators
        self.assertEqual(self.widget._order_combo.count(),
                         len(self.heart.domain) + 2 + len(INSTANCE_ORDERINGS))
        self.wait_until_finished()
        simulate.combobox_run_through_all(self.widget._order_combo)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget._order_combo.currentText(),
                         "Order instances by similarity")
        self.assertEqual(self.widget._order_combo.count(), 3)

    def test_annotation_combo(self):
        self.assertEqual(self.widget._annot_combo.currentText(), "None")
        self.assertEqual(self.widget._annot_combo.count(), 2)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.assertEqual(self.widget._annot_combo.currentText(), "None")
        self.assertEqual(self.widget._annot_combo.count(), 2)

        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:5])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.assertEqual(self.widget._annot_combo.currentText(), "None")
        self.assertEqual(self.widget._annot_combo.count(), 18)
        self.wait_until_finished()

        self.widget.graph.set_axis = Mock()
        simulate.combobox_activate_index(self.widget._annot_combo, 3)
        args = ([[(0, "0"), (1, "0"), (2, "0"), (3, "1"), (4, "1")]],)
        self.widget.graph.set_axis.assert_called_once_with(*args)

        self.widget.graph.set_axis.reset_mock()
        simulate.combobox_activate_index(self.widget._annot_combo, 1)
        args = ([[(0, "4"), (1, "5"), (2, "1"), (3, "3"), (4, "2")]],)
        self.widget.graph.set_axis.assert_called_once_with(*args)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget._annot_combo.currentText(), "None")
        self.assertEqual(self.widget._annot_combo.count(), 2)

    def test_setup_plot(self):
        self.widget.graph.set_data = Mock()
        self.widget.graph.set_axis = Mock()

        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.data, self.heart[:5])
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.widget.graph.set_axis.assert_called()

        self.widget.graph.set_axis.reset_mock()
        simulate.combobox_activate_index(self.widget._annot_combo, 3)
        self.widget.graph.set_data.assert_called_once()
        args = ([[(0, "0"), (1, "0"), (2, "0"), (3, "1"), (4, "1")]],)
        self.widget.graph.set_axis.assert_called_once_with(*args)

        self.widget.graph.set_data.reset_mock()
        self.widget.graph.set_axis.reset_mock()
        simulate.combobox_activate_index(self.widget._order_combo, 4)
        self.widget.graph.set_data.assert_called_once()
        args = ([[(0, "0"), (1, "0"), (2, "0"), (3, "1"), (4, "1")]],)
        self.widget.graph.set_axis.assert_called_once_with(*args)

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

    def test_plot_multiple_selection(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(10, 0)
        event.pos.return_value = QPointF(30, 0)
        event.isFinish.return_value = True
        self.widget.graph.getViewBox().mouseDragEvent(event)

        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(output), 4)

        event.buttonDownPos.return_value = QPointF(40, 0)
        event.pos.return_value = QPointF(50, 0)
        self.widget.graph.getViewBox().mouseDragEvent(event)

        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(output), 5)

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
        self.assertEqual(len(output), 10)
        self.assertEqual(output.X.shape[1], len(self.rf_cls.domain.attributes))
        self.assertEqual(len(output.Y), 10)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_output_selection(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(10, 0)
        event.pos.return_value = QPointF(30, 0)
        event.isFinish.return_value = True
        self.widget.graph.getViewBox().mouseDragEvent(event)

        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), 4)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_output_data(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsInstance(output, Table)
        self.assertEqual(len(output), 10)

        self.send_signal(self.widget.Inputs.model, None)
        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsInstance(output, Table)

        self.send_signal(self.widget.Inputs.background_data, None)
        output = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsInstance(output, Table)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_settings(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)

        simulate.combobox_activate_index(self.widget._target_combo, 1)
        simulate.combobox_activate_index(self.widget._order_combo, 4)
        simulate.combobox_activate_index(self.widget._annot_combo, 3)

        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.send_signal(self.widget.Inputs.background_data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)

        self.assertEqual(self.widget.target_index, -1)
        self.assertEqual(self.widget.order_index, 0)
        self.assertEqual(self.widget.annot_index, 0)

        simulate.combobox_activate_index(self.widget._order_combo, 6)
        simulate.combobox_activate_index(self.widget._annot_combo, 7)

        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.assertEqual(self.widget.target_index, 1)
        self.assertEqual(self.widget.order_index, 4)
        self.assertEqual(self.widget.annot_index, 3)

        self.send_signal(self.widget.Inputs.data, self.housing[:10])
        self.assertEqual(self.widget.target_index, -1)
        self.assertEqual(self.widget.order_index, 6)
        self.assertEqual(self.widget.annot_index, 7)

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = QPointF(10, 0)
        event.pos.return_value = QPointF(30, 0)
        event.isFinish.return_value = True
        self.widget.graph.getViewBox().mouseDragEvent(event)

        event.buttonDownPos.return_value = QPointF(40, 0)
        event.pos.return_value = QPointF(50, 0)
        self.widget.graph.getViewBox().mouseDragEvent(event)

        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(output), 5)

        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWExplainPredictions, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.heart[:10], widget=w)
        self.send_signal(w.Inputs.background_data, self.heart, widget=w)
        self.send_signal(w.Inputs.model, self.rf_cls, widget=w)
        self.wait_until_finished()

        output = self.get_output(w.Outputs.selected_data, widget=w)
        self.assertEqual(len(output), 5)

    def test_visual_settings(self):
        setter = self.widget.graph.parameter_setter

        def test_settings():
            font = QFont("Helvetica", italic=True, pointSize=20)
            font.setPointSize(15)
            for item in setter.axis_items:
                self.assertFontEqual(item.style["tickFont"], font)

            bottom_axis = setter.master.getAxis("bottom")
            self.assertFalse(bottom_axis.style["rotateTicks"])

        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        self.send_signal(self.widget.Inputs.background_data, self.heart)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Bottom axis", "Vertical ticks"), False
        self.widget.set_visual_settings(key, value)

        test_settings()

        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        test_settings()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self.heart[:10])
        test_settings()

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

    def assertFontEqual(self, font1: QFont, font2: QFont):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())


if __name__ == "__main__":
    unittest.main()
