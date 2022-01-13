from itertools import chain
from types import SimpleNamespace
from typing import Optional, List, Tuple

import numpy as np
from AnyQt.QtCore import QPointF, Qt, Signal, QRectF
from AnyQt.QtGui import QTransform, QPainter

import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent

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
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Input, Output, OWWidget, Msg
from orangecontrib.explain.explainer import explain_predictions, \
    prepare_force_plot_data_multi_inst, RGB_HIGH, RGB_LOW, \
    INSTANCE_ORDERINGS, get_instance_ordering


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


class SelectionRect(pg.GraphicsObject):
    def __init__(self, rect: QRectF):
        super().__init__()
        self.__rect = rect
        self.setZValue(1e9)

    def boundingRect(self) -> QRectF:
        return self.__rect

    def update_rect(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2)
        self.setPos(rect.topLeft())
        trans = QTransform.fromScale(rect.width(), rect.height())
        self.setTransform(trans)

    def paint(self, painter: QPainter, *_):
        painter.save()
        painter.setPen(pg.mkPen((255, 255, 0), width=1))
        painter.setBrush(pg.mkBrush(255, 255, 0, 100))
        painter.drawRect(self.__rect)
        painter.restore()


class ForcePlotViewBox(pg.ViewBox):
    sigSelectionChanged = Signal(QPointF, QPointF)
    sigDeselect = Signal()

    def __init__(self):
        super().__init__()
        self.__data_bounds: Optional[Tuple[Tuple[float, float],
                                           Tuple[float, float]]] = None
        self.__state: int = None
        self.set_state(SELECT)

        self.__selection_rect = SelectionRect(QRectF(0, 0, 1, 1))
        self.__selection_rect.hide()
        self.addItem(self.__selection_rect)

    def set_data_bounds(self, data_bounds: Optional[Tuple[
            Tuple[float, float], Tuple[float, float]]]):
        self.__data_bounds = data_bounds

    def set_state(self, state: int):
        self.__state = state
        self.setMouseMode(self.PanMode if state == PANNING else self.RectMode)

    def mouseDragEvent(self, ev: MouseDragEvent, axis: Optional[int] = None):
        if self.__state == SELECT and axis is None:
            ev.accept()

            if ev.button() == Qt.LeftButton and self.__data_bounds:
                p1, p2 = ev.buttonDownPos(), ev.pos()
                p1, p2 = self.mapToView(p1), self.mapToView(p2)

                (x1, x2), (y1, y2) = self.__data_bounds
                p1.setX(max(min(p1.x(), x2), x1))
                p2.setX(max(min(p2.x(), x2), x1))
                p1.setY(y1)
                p2.setY(y2)

                if ev.isFinish():
                    self.__selection_rect.hide()
                    self.sigSelectionChanged.emit(p1, p2)
                else:
                    self.__selection_rect.update_rect(p1, p2)
                    self.__selection_rect.show()

        elif self.__state == ZOOMING or self.__state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def mouseClickEvent(self, ev: MouseClickEvent):
        ev.accept()
        self.sigDeselect.emit()


class ForcePlot(pg.PlotWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: OWWidget):
        self.__data_bounds: Optional[Tuple[Tuple[float, float],
                                           Tuple[float, float]]] = None
        self.__selection: List = []
        self.__selection_rect_items: List[SelectionRect] = []

        view_box = ForcePlotViewBox()
        view_box.sigSelectionChanged.connect(self._update_selection)
        view_box.sigDeselect.connect(self._deselect)

        super().__init__(parent, viewBox=view_box,
                         background="w", enableMenu=False,
                         axisItems={"bottom": pg.AxisItem("bottom"),
                                    "left": pg.AxisItem("left")})
        self.setAntialiasing(True)
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().buttonsHidden = True

    def set_data(self, x_data: np.ndarray,
                 pos_y_data: List[Tuple[np.ndarray, np.ndarray]],
                 neg_y_data: List[Tuple[np.ndarray, np.ndarray]]):
        self.__data_bounds = ((np.nanmin(x_data), np.nanmax(x_data)),
                              (np.nanmin(pos_y_data), np.nanmax(neg_y_data)))
        self.getViewBox().set_data_bounds(self.__data_bounds)
        self._set_range()
        self._plot_data(x_data, pos_y_data, neg_y_data)

    def _plot_data(self, x_data: np.ndarray,
                   pos_y_data: List[Tuple[np.ndarray, np.ndarray]],
                   neg_y_data: List[Tuple[np.ndarray, np.ndarray]]):
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
        self.__data_bounds = None
        self.getViewBox().set_data_bounds(self.__data_bounds)
        self.clear()
        self._clear_selection()

    def _clear_selection(self):
        self.__selection = []
        for i in range(len(self.__selection_rect_items)):
            self.removeItem(self.__selection_rect_items[i])
        self.__selection_rect_items.clear()

    def _update_selection(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2).normalized()
        self.__selection.append((rect.topLeft().x(), rect.topRight().x()))

        sel_rect_item = SelectionRect(rect)
        self.addItem(sel_rect_item)
        self.__selection_rect_items.append(sel_rect_item)
        self.selectionChanged.emit(self.__selection)

    def _deselect(self):
        selection_existed = bool(self.__selection)
        self._clear_selection()
        if selection_existed:
            self.selectionChanged.emit([])

    def select_button_clicked(self):
        self.getViewBox().set_state(SELECT)

    def pan_button_clicked(self):
        self.getViewBox().set_state(PANNING)

    def zoom_button_clicked(self):
        self.getViewBox().set_state(ZOOMING)

    def reset_button_clicked(self):
        self._set_range()

    def _set_range(self):
        x_range, y_range = self.__data_bounds or ((0, 1), (0, 1))
        view_box: ForcePlotViewBox = self.getViewBox()
        view_box.setXRange(*x_range, padding=0)
        view_box.setYRange(*y_range, padding=0.1)


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
    selection = Setting([], schema_only=True)
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
        # cached instance indices after instance ordering
        self.__data_idxs: Optional[np.ndarray] = None
        self.__pending_selection: List[int] = self.selection

        self.setup_gui()

    def setup_gui(self):
        self._add_plot()
        self._add_controls()
        self._add_buttons()

    def _add_plot(self):
        box = gui.vBox(self.mainArea)
        self.graph = ForcePlot(self)
        self.graph.selectionChanged.connect(self.__on_selection_changed)
        box.layout().addWidget(self.graph)

    def __on_selection_changed(self, selection: List[Tuple[float, float]]):
        ordering = self._order_combo.model()[self.order_index]

        if ordering in INSTANCE_ORDERINGS:
            selection = list(set(
                chain.from_iterable(
                    range(int(np.ceil(start)), int(np.floor(stop) + 1))
                    for start, stop in selection)
            ))
            self.selection = sorted(self.__data_idxs[selection])

        else:  # variable
            data = self.__results.transformed_data
            column = data.get_column_view(ordering)[0]
            mask = np.zeros((len(column)), dtype=bool)
            for start, stop in selection:
                mask |= (start <= column) & (column <= stop)
            self.selection = list(np.flatnonzero(mask))

        self.commit()

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
        self.selection = []
        self.graph.clear_all()
        self.__data_idxs = None

    def setup_plot(self):
        self.graph.clear_all()
        self.__data_idxs = None
        if not self.__results or not self.data:
            return

        self.__data_idxs = get_instance_ordering(
            self.__results.values,
            self.__results.predictions,
            self.target_index,
            self.__results.transformed_data,
            self._order_combo.model()[self.order_index]
        )

        x_data, pos_y_data, neg_y_data, _, _ = \
            prepare_force_plot_data_multi_inst(
                self.__results.values,
                self.__results.base_value,
                self.target_index,
                self.__results.transformed_data,
                self._order_combo.model()[self.order_index],
                self.__data_idxs
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
