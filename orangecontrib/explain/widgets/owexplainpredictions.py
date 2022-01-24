from itertools import chain
from types import SimpleNamespace
from typing import Optional, List, Tuple, Any
from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import QPointF, Qt, Signal, QRectF
from AnyQt.QtGui import QTransform, QPainter
from AnyQt.QtWidgets import QToolTip, QGraphicsSceneHelpEvent, QComboBox

import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.base import Model
from Orange.data import Table, Domain, ContinuousVariable, Variable
from Orange.data.table import DomainTransformationError, RowInstance
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
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import HelpEventDelegate, \
    AxisItem
from Orange.widgets.widget import Input, Output, OWWidget, Msg

from orangecontrib.explain.explainer import explain_predictions, \
    prepare_force_plot_data_multi_inst, RGB_HIGH, RGB_LOW, \
    INSTANCE_ORDERINGS, get_instance_ordering


class RunnerResults(SimpleNamespace):
    values: Optional[List[np.ndarray]] = None
    predictions: Optional[np.ndarray] = None
    transformed_data: Optional[Table] = None
    mask: Optional[np.ndarray] = None
    base_value: Optional[float] = None


def run(data: Table, background_data: Table, model: Model, state: TaskState) \
        -> Optional[RunnerResults]:
    if not data or not background_data or not model:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    values, pred, data, sample_mask, base_value = explain_predictions(
        model, data, background_data, callback)
    return RunnerResults(values=values,
                         predictions=pred,
                         transformed_data=data,
                         mask=sample_mask,
                         base_value=base_value)


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


class ParameterSetter(CommonParameterSetter):
    BOTTOM_AXIS_LABEL = "Bottom axis"
    IS_VERTICAL_LABEL = "Vertical ticks"

    def __init__(self, parent):
        super().__init__()
        self.master: ForcePlot = parent

    def update_setters(self):
        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
            },
            self.PLOT_BOX: {
                self.BOTTOM_AXIS_LABEL: {self.IS_VERTICAL_LABEL: (None, True)}
            }
        }

        def update_bottom_axis(**settings):
            axis = self.master.getAxis("bottom")
            axis.setRotateTicks(settings[self.IS_VERTICAL_LABEL])

        self._setters[self.PLOT_BOX] = {
            self.BOTTOM_AXIS_LABEL: update_bottom_axis
        }

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]


class ForcePlot(pg.PlotWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: OWWidget):
        self.__data_bounds: Optional[Tuple[Tuple[float, float],
                                           Tuple[float, float]]] = None
        self.__tooltip_data: Optional[Table] = None

        self.__selection: List = []
        self.__selection_rect_items: List[SelectionRect] = []

        view_box = ForcePlotViewBox()
        view_box.sigSelectionChanged.connect(self._update_selection)
        view_box.sigDeselect.connect(self._deselect)

        super().__init__(parent, viewBox=view_box,
                         background="w", enableMenu=False,
                         axisItems={"bottom": AxisItem("bottom", True),
                                    "left": AxisItem("left")})
        self.setAntialiasing(True)
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().buttonsHidden = True

        self._tooltip_delegate = HelpEventDelegate(self.help_event)
        self.scene().installEventFilter(self._tooltip_delegate)

        self.parameter_setter = ParameterSetter(self)

    def set_data(self, x_data: np.ndarray,
                 pos_y_data: List[Tuple[np.ndarray, np.ndarray]],
                 neg_y_data: List[Tuple[np.ndarray, np.ndarray]],
                 tooltip_data: Table):

        self.__data_bounds = ((np.nanmin(x_data), np.nanmax(x_data)),
                              (np.nanmin(pos_y_data), np.nanmax(neg_y_data)))
        self.__tooltip_data = tooltip_data

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

    def set_axis(self, ticks: Optional[List]):
        ax: AxisItem = self.getAxis("bottom")
        ax.setTicks(ticks)

    def clear_all(self):
        self.__data_bounds = None
        self.__tooltip_data = None
        self.getViewBox().set_data_bounds(self.__data_bounds)
        self.clear()
        self._clear_selection()

    def _clear_selection(self):
        self.__selection = []
        for i in range(len(self.__selection_rect_items)):
            self.removeItem(self.__selection_rect_items[i])
        self.__selection_rect_items.clear()

    def _update_selection(self, p1: QPointF, p2: QPointF):
        self.__select_range(p1, p2)
        self.selectionChanged.emit(self.__selection)

    def __select_range(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2).normalized()
        self.__selection.append((rect.topLeft().x(), rect.topRight().x()))

        sel_rect_item = SelectionRect(rect)
        self.addItem(sel_rect_item)
        self.__selection_rect_items.append(sel_rect_item)

    def _deselect(self):
        selection_existed = bool(self.__selection)
        self._clear_selection()
        if selection_existed:
            self.selectionChanged.emit([])

    def apply_selection(self, selection: List[Tuple[float, float]]):
        if self.__data_bounds is None:
            return

        (x_min, x_max), (y_min, y_max) = self.__data_bounds
        for x_start, x_stop in selection:
            p1 = QPointF(x_start, y_min)
            p2 = QPointF(x_stop, y_max)
            p1.setX(max(min(p1.x(), x_max), x_min))
            p2.setX(max(min(p2.x(), x_max), x_min))
            self.__select_range(p1, p2)

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

    def help_event(self, event: QGraphicsSceneHelpEvent) -> bool:
        if self.__tooltip_data is None:
            return False

        point: QPointF = self.getViewBox().mapSceneToView(event.scenePos())
        index = int(round(point.x(), 0))

        if 0 <= index < len(self.__tooltip_data):
            text = self._instance_tooltip(self.__tooltip_data.domain,
                                          self.__tooltip_data[index])
            QToolTip.showText(event.screenPos(), text, widget=self)
            return True
        return False

    @staticmethod
    def _instance_tooltip(domain: Domain, instance: RowInstance) -> str:
        def show_part(singular, plural, max_shown, variables):
            cols = [escape(f"{var.name} = {instance[var]}")
                    for var in variables[:max_shown + 2]][:max_shown]
            if not cols:
                return ""

            n_vars = len(variables)
            if n_vars > max_shown:
                cols[-1] = f"... and {n_vars - max_shown + 1} others"

            tag = singular if n_vars < 2 else plural
            return f"<b>{tag}</b>:<br/>" + "<br/>".join(cols)

        parts = (("Class", "Classes", 4, domain.class_vars),
                 ("Meta", "Metas", 4, domain.metas),
                 ("Feature", "Features", 10, domain.attributes))

        return "<br/>".join(show_part(*columns) for columns in parts)


class OWExplainPredictions(OWWidget, ConcurrentWidgetMixin):
    name = "Explain Predictions"
    description = "Predictions explanation widget."
    keywords = ["explain", "explain prediction", "explain model"]
    icon = "icons/ExplainPredictions.svg"
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
        not_enough_data = Msg("At least two instances are needed.")

    class Information(OWWidget.Information):
        data_sampled = Msg("Data has been sampled.")

    buttons_area_orientation = Qt.Vertical

    settingsHandler = PerfectDomainContextHandler()
    target_index = ContextSetting(0)
    order_index = ContextSetting(0)
    annot_index = ContextSetting(0)
    selection_ranges = Setting([], schema_only=True)
    auto_send = Setting(True)
    visual_settings = Setting({}, schema_only=True)

    graph_name = "graph.plotItem"

    ANNOTATIONS = ["None", "Enumeration"]

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__results: Optional[RunnerResults] = None
        self.model: Optional[Model] = None
        self.background_data: Optional[Table] = None
        self.data: Optional[Table] = None
        # cached instance indices after instance ordering
        self.__data_idxs: Optional[np.ndarray] = None
        self.__pending_selection: List[Tuple[float, float]] = \
            self.selection_ranges

        self.graph: ForcePlot = None
        self._target_combo: QComboBox = None
        self._order_combo: ForcePlot = None
        self._annot_combo: ForcePlot = None

        self.setup_gui()

        initial_settings = self.graph.parameter_setter.initial_settings
        VisualSettingsDialog(self, initial_settings)

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
        self.selection_ranges = selection
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
        self.selection_ranges = []
        self.setup_plot()
        self.commit()

    def __on_order_changed(self):
        self.selection_ranges = []
        self.setup_plot()
        self.commit()

    def __on_annot_changed(self):
        if not self.__results or not self.data:
            return
        self._set_plot_annotations()

    def _add_buttons(self):
        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_send")

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.data = data
        self._check_data()
        self._setup_controls()
        self.openContext(self.data.domain if self.data else None)

    @Inputs.background_data
    @check_sql_input
    def set_background_data(self, data: Optional[Table]):
        self.background_data = data

    @Inputs.model
    def set_model(self, model: Optional[Model]):
        self.model = model

    def _check_data(self):
        self.Error.not_enough_data.clear()
        if self.data and len(self.data) < 2:
            self.data = None
            self.Error.not_enough_data()

    def _setup_controls(self):
        self._target_combo.clear()
        self._target_combo.setEnabled(True)

        self.order_index = 0
        self.annot_index = 0
        self._order_combo.clear()
        self._annot_combo.clear()
        orderings = INSTANCE_ORDERINGS
        annotations = self.ANNOTATIONS

        if self.data:
            domain = self.data.domain
            if domain.has_discrete_class:
                self._target_combo.addItems(domain.class_var.values)
                self.target_index = 0
            elif domain.has_continuous_class:
                self.target_index = -1
                self._target_combo.setEnabled(False)

            orderings = chain(
                INSTANCE_ORDERINGS,
                [VariableListModel.Separator] if domain.metas else [],
                domain.metas,
                [VariableListModel.Separator] if domain.class_vars else [],
                domain.class_vars,
                [VariableListModel.Separator] if domain.attributes else [],
                domain.attributes,
            )

            annotations = chain(
                self.ANNOTATIONS,
                [VariableListModel.Separator] if domain.metas else [],
                domain.metas,
                [VariableListModel.Separator] if domain.class_vars else [],
                domain.class_vars,
                [VariableListModel.Separator] if domain.attributes else [],
                domain.attributes,
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
        self.Error.domain_transform_err.clear()
        self.Error.unknown_err.clear()
        self.Information.data_sampled.clear()
        self.selection_ranges = []
        self.graph.clear_all()
        self.graph.set_axis(None)
        self.__data_idxs = None

    def setup_plot(self):
        self.graph.clear_all()
        self.__data_idxs = None
        if not self.__results or not self.data:
            return

        values_idxs = get_instance_ordering(
            self.__results.values[self.target_index],
            self.__results.predictions[self.__results.mask, self.target_index],
            self.data[self.__results.mask],
            self._order_combo.model()[self.order_index]
        )

        data_idxs = np.arange(len(self.data))
        self.__data_idxs = data_idxs[self.__results.mask][values_idxs]

        x_data, pos_y_data, neg_y_data = \
            prepare_force_plot_data_multi_inst(
                self.__results.values[self.target_index][values_idxs],
                self.__results.base_value[self.target_index]
            )

        self.graph.set_data(x_data, pos_y_data, neg_y_data,
                            self.data[self.__data_idxs])
        self._set_plot_annotations()

    def _set_plot_annotations(self):
        annotator = self._annot_combo.model()[self.annot_index]
        if isinstance(annotator, Variable):
            ticks = [[(i, str(row[annotator].value)) for i, row in
                      enumerate(self.data[self.__data_idxs])]]
            self.graph.set_axis(ticks)
        elif annotator == "None":
            self.graph.set_axis([])
        elif annotator == "Enumeration":
            ticks = [[(i, str(idx + 1)) for i, idx in
                      enumerate(self.__data_idxs)]]
            self.graph.set_axis(ticks)
        else:
            raise NotImplementedError(annotator)

    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[RunnerResults]):
        self.__results = results
        if results is not None and not all(results.mask):
            self.Information.data_sampled()
        self.setup_plot()
        self.apply_selection()
        self.output_scores()

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def apply_selection(self):
        selection_ranges = self.selection_ranges or self.__pending_selection
        if selection_ranges:
            self.graph.apply_selection(selection_ranges)
            self.__on_selection_changed(selection_ranges)
            self.__pending_selection = []

    def commit(self):
        selected = None
        selected_indices = []

        if self.__results:
            selection = list(set(
                chain.from_iterable(
                    range(int(np.ceil(start)), int(np.floor(stop) + 1))
                    for start, stop in self.selection_ranges)
            ))
            selected_indices = sorted(self.__data_idxs[selection])

        if self.data and selected_indices:
            selected = self.data[selected_indices]
        annotated = create_annotated_table(self.data, selected_indices)
        self.Outputs.selected_data.send(selected)
        self.Outputs.annotated_data.send(annotated)

    def output_scores(self):
        scores = None
        if self.__results is not None:
            mask = self.__results.mask
            data = self.__results.transformed_data[mask]
            domain = data.domain
            attrs = [ContinuousVariable(f"S({a.name})")
                     for a in domain.attributes]
            domain = Domain(attrs, domain.class_vars, domain.metas)
            scores = self.__results.values[self.target_index]
            scores = Table(domain, scores, data.Y, data.metas)
            scores.name = "Feature Scores"
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

    def set_visual_settings(self, key: Tuple[str, str, str], value: Any):
        self.visual_settings[key] = value
        self.graph.parameter_setter.set_parameter(key, value)


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
