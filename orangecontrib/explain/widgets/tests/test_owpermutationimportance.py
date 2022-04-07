# pylint: disable=missing-docstring,protected-access,invalid-name
import inspect
import itertools
import unittest
from unittest.mock import patch, Mock

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtGui import QFont
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QGraphicsGridLayout, QComboBox, QGraphicsTextItem

import pyqtgraph as pg

from orangecanvas.gui.test import mouseMove
from orangewidget.tests.base import WidgetTest

import Orange
from Orange.base import Learner
from Orange.classification import RandomForestLearner, \
    OneClassSVMLearner, IsolationForestLearner, \
    EllipticEnvelopeLearner, LocalOutlierFactorLearner
from Orange.data import Table, Domain
from Orange.regression import RandomForestRegressionLearner
from Orange.widgets.tests.utils import simulate

from orangecontrib.explain.widgets.owexplainfeaturebase import VariableItem
from orangecontrib.explain.widgets.owpermutationimportance import \
    OWPermutationImportance, Results, FeatureImportancePlot, \
    FeatureImportanceItem


def dummy_run(data, model, *_):
    if not data or model is None:
        return None
    m, n = data.X.shape
    mask = np.ones(m, dtype=bool)
    mask[150:] = False
    return Results(x=np.ones((n, 1)), names=[str(i) for i in range(n)],
                   mask=mask)


class TestOWPermutationImportance(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")
        cls.heart = Table("heart_disease")
        cls.housing = Table("housing")
        cls.rf_cls = RandomForestLearner(random_state=0)(cls.iris)
        cls.rf_reg = RandomForestRegressionLearner(random_state=0)(cls.housing)

    def setUp(self):
        self.widget = self.create_widget(OWPermutationImportance)

    def test_classification_data_classification_model(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget.plot, self.iris.domain)

    def test_classification_data_regression_model(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)
        self.assertTrue(self.widget.Error.unknown_err.is_shown())

    def test_regression_data_regression_model(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget.plot, self.housing.domain)

    def test_regression_data_classification_model(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)
        self.assertTrue(self.widget.Error.unknown_err.is_shown())

    def test_data_with_no_features(self):
        domain = Domain([],
                        class_vars=self.iris.domain.class_vars,
                        metas=self.iris.domain.attributes)
        data = self.iris.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)
        self.assertTrue(self.widget.Error.unknown_err.is_shown())

    def test_missing_target(self):
        data = self.housing.copy()
        data.Y[0] = np.nan
        rf = RandomForestRegressionLearner(random_state=0)(data)

        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.model, rf)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget.plot, data.domain)
        self.assertTrue(self.widget.Warning.missing_target.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.missing_target.is_shown())

    def test_output_scores(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.scores)
        self.assertIsInstance(output, Table)
        self.assertListEqual([a.name for a in output.domain.attributes],
                             ["Mean", "Std"])
        self.assertListEqual([a.name for a in output.domain.metas],
                             ["Feature"])
        self.assertEqual(list(output.metas.flatten()),
                         [a.name for a in self.iris.domain.attributes])
        self.send_signal(self.widget.Inputs.model, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.scores))

    def test_selection(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()

        plot = self.widget.plot
        h = plot.layout().itemAt(0, plot.ITEM_COLUMN)
        pos = self.widget.view.mapFromScene(h.scenePos())
        QTest.mousePress(self.widget.view.viewport(), Qt.LeftButton,
                         pos=pos + QPoint(1, 1))
        mouseMove(self.widget.view.viewport(), Qt.LeftButton,
                  pos=pos + QPoint(200, 20))
        QTest.mouseRelease(self.widget.view.viewport(), Qt.LeftButton,
                           pos=pos + QPoint(200, 30))
        selection = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsInstance(selection, Table)
        self.assertEqual(len(selection), 150)
        self.assertEqual(len(selection.domain.attributes), 1)

        QTest.mouseClick(self.widget.view.viewport(), Qt.LeftButton,
                         pos=QPoint(10, 10))
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_saved_selection(self):
        log_reg = RandomForestLearner(random_state=0)(self.heart)

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.model, log_reg)
        self.wait_until_finished()

        plot = self.widget.plot
        h = plot.layout().itemAt(0, plot.ITEM_COLUMN)
        pos = self.widget.view.mapFromScene(h.scenePos())
        QTest.mousePress(self.widget.view.viewport(), Qt.LeftButton,
                         pos=pos + QPoint(0, 10))
        mouseMove(self.widget.view.viewport(), Qt.LeftButton,
                  pos=pos + QPoint(300, 20))
        QTest.mouseRelease(self.widget.view.viewport(), Qt.LeftButton,
                           pos=pos + QPoint(300, 30))
        saved_selection = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNotNone(saved_selection)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWPermutationImportance,
                               stored_settings=settings)
        self.send_signal(w.Inputs.data, self.heart, widget=w)
        log_reg = RandomForestLearner(random_state=0)(self.heart)
        self.send_signal(w.Inputs.model, log_reg, widget=w)
        self.wait_until_finished(widget=w)
        selection = self.get_output(w.Outputs.selected_data, widget=w)
        np.testing.assert_array_equal(selection.X, saved_selection.X)

    def test_all_models(self):
        def run(data):
            self.send_signal(self.widget.Inputs.data, data)
            if not issubclass(cls, Learner) or \
                    issubclass(cls, (EllipticEnvelopeLearner,
                                     LocalOutlierFactorLearner,
                                     IsolationForestLearner,
                                     OneClassSVMLearner)):
                return
            try:
                model = cls()(data)
            except:
                return
            self.send_signal(self.widget.Inputs.model, model)
            self.wait_until_finished(timeout=50000)

        for _, cls in itertools.chain(
                inspect.getmembers(Orange.regression, inspect.isclass),
                inspect.getmembers(Orange.modelling, inspect.isclass)):
            run(self.housing[::4])
        for _, cls in itertools.chain(
                inspect.getmembers(Orange.classification, inspect.isclass),
                inspect.getmembers(Orange.modelling, inspect.isclass)):
            run(self.iris[::4])

    def test_score_combo(self):
        score_cb: QComboBox = self.widget._score_combo
        simulate.combobox_run_through_all(score_cb)

        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_run_through_all(
            score_cb, callback=self.wait_until_finished)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        simulate.combobox_run_through_all(
            score_cb, callback=self.wait_until_finished)

        self.send_signal(self.widget.Inputs.data, self.housing)
        simulate.combobox_run_through_all(
            score_cb, callback=self.wait_until_finished)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        simulate.combobox_run_through_all(
            score_cb, callback=self.wait_until_finished)

        self.send_signal(self.widget.Inputs.data, None)
        simulate.combobox_run_through_all(
            score_cb, callback=self.wait_until_finished)

    @patch("orangecontrib.explain.widgets.owpermutationimportance."
           "permutation_feature_importance")
    def test_n_repeats(self, mocked_func: Mock):
        self.widget.controls.n_repeats.setValue(3)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertEqual(mocked_func.call_args[0][3], 3)

    @patch("orangecontrib.explain.widgets.owpermutationimportance."
           "MAX_N_ITEMS", 3)
    def test_n_attributes(self):
        self.widget.controls.n_attributes.setValue(3)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        domain = self.iris.domain
        domain = Domain(domain.attributes[:3], domain.class_vars)
        self.assertDomainInPlot(self.widget.plot, domain)

    def test_zoom_level(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.widget.controls.zoom_level.setValue(10)
        self.assertDomainInPlot(self.widget.plot, self.iris.domain)

    def test_plot(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget.plot, self.iris.domain)

        self.send_signal(self.widget.Inputs.model, None)
        self.assertPlotEmpty(self.widget.plot)

        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertPlotEmpty(self.widget.plot)

        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertDomainInPlot(self.widget.plot, self.iris.domain)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertPlotEmpty(self.widget.plot)

    def test_x_label(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        label: QGraphicsTextItem = self.widget.plot.bottom_axis.label
        self.assertEqual(label.toPlainText(), "Decrease in AUC ")

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.wait_until_finished()
        label: QGraphicsTextItem = self.widget.plot.bottom_axis.label
        self.assertEqual(label.toPlainText(), "Decrease in R2 ")

        score_cb: QComboBox = self.widget._score_combo
        simulate.combobox_activate_item(score_cb, "MSE")
        self.wait_until_finished()
        label: QGraphicsTextItem = self.widget.plot.bottom_axis.label
        self.assertEqual(label.toPlainText(), "Increase in MSE ")

    @unittest.mock.patch("orangecontrib.explain.widgets."
                         "owpermutationimportance.OWPermutationImportance.run")
    def test_data_sampled_info(self, mocked_run):
        mocked_run.side_effect = dummy_run
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

        self.send_signal(self.widget.Inputs.data, self.heart)
        log_reg = RandomForestLearner(random_state=0)(self.heart)
        self.send_signal(self.widget.Inputs.model, log_reg)
        self.wait_until_finished()
        self.assertTrue(self.widget.Information.data_sampled.is_shown())

        self.send_signal(self.widget.Inputs.data, self.heart)
        self.send_signal(self.widget.Inputs.model, None)
        self.wait_until_finished()
        self.assertFalse(self.widget.Information.data_sampled.is_shown())

    def test_sparse_data(self):
        data = self.heart
        sparse_data = data.to_sparse()
        with sparse_data.unlocked():
            sparse_data.X = sp.csr_matrix(sparse_data.X)

        sparse_model = RandomForestLearner(random_state=0)(sparse_data)
        self.send_signal(self.widget.Inputs.data, sparse_data)
        self.send_signal(self.widget.Inputs.model, sparse_model)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

        model = RandomForestLearner(random_state=0)(data)
        self.send_signal(self.widget.Inputs.data, sparse_data)
        self.send_signal(self.widget.Inputs.model, model)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.model, sparse_model)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.domain_transform_err.is_shown())
        self.assertFalse(self.widget.Error.unknown_err.is_shown())

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.send_signal(self.widget.Inputs.model, self.rf_reg)
        self.widget.send_report()

    def test_visual_settings(self):
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.model, self.rf_cls)
        self.wait_until_finished()
        setter = self.widget.plot.parameter_setter

        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis title", "Font size"), 14
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(14)
        for axis in setter.axis_items:
            self.assertFontEqual(axis.label.font(), font)

        key, value = ('Fonts', 'Axis ticks', 'Font size'), 15
        self.widget.set_visual_settings(key, value)
        key, value = ('Fonts', 'Axis ticks', 'Italic'), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(15)
        for axis in setter.axis_items:
            self.assertFontEqual(axis.style["tickFont"], font)

        key, value = ("Fonts", "Variable name", "Font size"), 19
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Variable name", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(19)
        self.assertFontEqual(setter.labels[0].item.items[0].font(), font)

        key, value = ("Fonts", "Variable value", "Font size"), 10
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Variable value", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(10)
        self.assertFontEqual(setter.labels[0].item.items[1].font(), font)

        key, value = ("Figure", "Label length", "Label length"), 50
        self.widget.set_visual_settings(key, value)
        self.assertLessEqual(setter.labels[0].item.boundingRect().width(), 50)

    def assertPlotEmpty(self, plot: FeatureImportancePlot):
        self.assertIsNone(plot)

    def assertDomainInPlot(self, plot: FeatureImportancePlot, domain: Domain):
        layout = plot.layout()  # type: QGraphicsGridLayout
        n_rows = layout.rowCount()
        self.assertEqual(n_rows, len(domain.attributes) + 1)
        self.assertEqual(layout.columnCount(), 2)
        for i in range(layout.rowCount() - 1):
            item0 = layout.itemAt(i, 0).item
            self.assertIsInstance(item0, VariableItem)
            self.assertIsInstance(layout.itemAt(i, 1), FeatureImportanceItem)
        self.assertIsNone(layout.itemAt(n_rows - 1, 0))
        self.assertIsInstance(layout.itemAt(n_rows - 1, 1), pg.AxisItem)

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    def test_orange_version(self):
        """
        This test serves as a reminder.

        When it starts to fail, remove it and remove the lines 18, 305 - 306 in
        owpermutationimportance.py
        """
        from Orange.version import version

        self.assertLess(version, "3.35.0")


if __name__ == "__main__":
    unittest.main()
