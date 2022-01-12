from itertools import chain
from types import SimpleNamespace
from typing import Optional, List

import numpy as np
from AnyQt.QtCore import Qt

import pyqtgraph as pg
from Orange.base import Model
from Orange.data import Table, Domain, ContinuousVariable
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, \
    PerfectDomainContextHandler
from Orange.widgets.utils.annotated_data import ANNOTATED_DATA_SIGNAL_NAME, \
    create_annotated_table
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.utils.plot import OWPlotGUI
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget, Msg
from orangecontrib.explain.explainer import explain_predictions, \
    prepare_force_plot_data_multi_inst, RGB_HIGH, RGB_LOW, INSTANCE_ORDERINGS


class RunnerResults(SimpleNamespace):
    values: Optional[List[np.ndarray]] = None
    predictions: Optional[np.ndarray] = None
    transformed_data: Optional[Table] = None
    base_value: Optional[float] = None


def run(data: Table, background_data: Table, model: Model, state: TaskState) \
        -> RunnerResults:
    if not data or not background_data or not model:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    values, pred, data, base_value = explain_predictions(
        model, data, background_data, callback)
    return RunnerResults(values=values, predictions=pred,
                         transformed_data=data, base_value=base_value)


class ForcePlot(pg.PlotWidget):
    def __init__(self, parent: OWWidget):
        super().__init__(parent, viewBox=pg.ViewBox(),
                         background="w", enableMenu=False,
                         axisItems={"bottom": pg.AxisItem("bottom"),
                                    "left": pg.AxisItem("left")})
        self.setAntialiasing(True)
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)

    def set_data(self, x_data, pos_y_data, neg_y_data):
        for rgb, data in ((RGB_HIGH, pos_y_data), (RGB_LOW, neg_y_data)):
            whiter_rgb = np.array(rgb) + (255 - np.array(rgb)) * 0.7
            pen = pg.mkPen(whiter_rgb, width=1)
            brush = pg.mkBrush(rgb)
            for y_top, y_bottom in data:
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(x=x_data, y=y_bottom),
                    pg.PlotDataItem(x=x_data, y=y_top), pen=pen, brush=brush
                )
                self.addItem(fill)

    def clear_all(self):
        self.clear()

    def zoom_button_clicked(self):
        pass

    def pan_button_clicked(self):
        pass

    def select_button_clicked(self):
        pass

    def reset_button_clicked(self):
        pass


class OWExplainPredictions(OWWidget, ConcurrentWidgetMixin):
    name = "Explain Predictions"
    description = "Predictions explanation widget."
    keywords = ["explain", "explain prediction", "explain model"]
    icon = "icons/ExplainPred.svg"
    priority = 120

    class Inputs:
        model = Input("Model", Model)
        background_data = Input("Background Data", Table)
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table, default=True)
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table)
        scores = Output("Scores", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")

    class Information(OWWidget.Information):
        multiple_instances = Msg("Explaining prediction for the first "
                                 "instance in 'Data'.")

    buttons_area_orientation = Qt.Vertical

    settingsHandler = PerfectDomainContextHandler()
    target_index = ContextSetting(0)
    order_index = ContextSetting(0)
    annot_index = ContextSetting(0)
    auto_send = Setting(True)

    graph_name = "graph.plotItem"

    ANNOTATIONS = ["Enumeration"]

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__results: Optional[RunnerResults] = None
        self.model: Optional[Model] = None
        self.background_data: Optional[Table] = None
        self.data: Optional[Table] = None
        self.selection: List[int] = []

        self.setup_gui()

    def setup_gui(self):
        self._add_plot()
        self._add_controls()
        self._add_buttons()

    def _add_plot(self):
        box = gui.vBox(self.mainArea)
        self.graph = ForcePlot(self)
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Target class")
        self._target_combo = gui.comboBox(box, self, "target_index",
                                          callback=self.__on_target_changed,
                                          contentsLength=12)

        box = gui.vBox(self.controlArea, "Instance order")
        self._order_combo = gui.comboBox(box, self, "order_index",
                                         callback=self.__on_order_changed,
                                         searchable=True, contentsLength=12)
        model = VariableListModel()
        model[:] = INSTANCE_ORDERINGS
        self._order_combo.setModel(model)

        box = gui.vBox(self.controlArea, "Annotation")
        self._annot_combo = gui.comboBox(box, self, "annot_index",
                                         callback=self.__on_annot_changed,
                                         searchable=True, contentsLength=12)
        model = VariableListModel()
        model[:] = self.ANNOTATIONS
        self._annot_combo.setModel(model)

        gui.rubber(self.controlArea)

    def __on_target_changed(self):
        self.setup_plot()

    def __on_order_changed(self):
        self.setup_plot()

    def __on_annot_changed(self):
        self.setup_plot()

    def _add_buttons(self):
        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_send")

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.data = data

    @Inputs.background_data
    @check_sql_input
    def set_background_data(self, data: Optional[Table]):
        self.background_data = data

    @Inputs.model
    def set_model(self, model: Optional[Model]):
        self.closeContext()
        self.model = model
        self.setup_controls()
        self.openContext(self.model.domain if self.model else None)

    def setup_controls(self):
        self._target_combo.clear()
        self._target_combo.setEnabled(True)

        self.order_index = 0
        self.annot_index = 0
        self._order_combo.clear()
        self._annot_combo.clear()
        orderings = INSTANCE_ORDERINGS
        annotations = self.ANNOTATIONS

        model = self.model
        if model is not None:
            if model.domain.has_discrete_class:
                self._target_combo.addItems(model.domain.class_var.values)
                self.target_index = 0
            elif model.domain.has_continuous_class:
                self.target_index = -1
                self._target_combo.setEnabled(False)
            else:
                raise NotImplementedError

            c_attrs = [a for a in model.domain.attributes if a.is_continuous]
            orderings = chain(
                INSTANCE_ORDERINGS,
                [VariableListModel.Separator] if c_attrs else [],
                c_attrs,
            )

            annotations = chain(
                self.ANNOTATIONS,
                [VariableListModel.Separator] if model.domain.metas else [],
                self.model.domain.metas,
                [VariableListModel.Separator],
                self.model.domain.class_vars,
                [VariableListModel.Separator],
                self.model.domain.attributes,
            )

        self._order_combo.model()[:] = orderings
        self._annot_combo.model()[:] = annotations

    def handleNewSignals(self):
        self.clear()
        self.start(run, self.data, self.background_data, self.model)
        self.commit()

    def clear(self):
        self.__results = None
        self.cancel()
        self.clear_messages()
        self.graph.clear_all()

    def setup_plot(self):
        self.graph.clear_all()
        if not self.__results or not self.data:
            return

        x_data, pos_y_data, neg_y_data, _, _ = \
            prepare_force_plot_data_multi_inst(
                self.__results.values,
                self.__results.base_value,
                self.__results.predictions,
                self.target_index,
                self.__results.transformed_data,
                self._order_combo.model()[self.order_index]
            )

        self.graph.set_data(x_data, pos_y_data, neg_y_data)

    def on_partial_result(self, _):
        pass

    def on_done(self, results):
        self.__results = results
        self.setup_plot()
        self.output_scores()

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def commit(self):
        selected = None
        if self.data and self.selection:
            selected = self.data[self.selection]
        annotated = create_annotated_table(self.data, self.selection)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def output_scores(self):
        scores = None
        if self.__results is not None:
            domain = self.__results.transformed_data.domain
            scores = self.__results.values[self.target_index]
            attrs = [ContinuousVariable(a.name) for a in domain.attributes]
            scores = Table(Domain(attrs), scores)
        self.Outputs.scores.send(scores)

    def send_report(self):
        if not self.data or not self.background_data or not self.model:
            return
        items = {"Target class": "None"}
        if self.model.domain.has_discrete_class:
            class_var = self.model.domain.class_var
            items["Target class"] = class_var.values[self.target_index]
        self.report_items(items)
        self.report_plot()


if __name__ == "__main__":
    from Orange.classification import RandomForestLearner
    from Orange.regression import RandomForestRegressionLearner

    table = Table("housing")
    kwargs = {"n_estimators": 10, "random_state": 0}
    if table.domain.has_continuous_class:
        model_ = RandomForestRegressionLearner(**kwargs)(table)
    else:
        model_ = RandomForestLearner(**kwargs)(table)
    WidgetPreview(OWExplainPredictions).run(
        set_background_data=table, set_data=table[:50], set_model=model_
    )
