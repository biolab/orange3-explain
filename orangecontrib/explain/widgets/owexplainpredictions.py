from itertools import chain
from types import SimpleNamespace
from typing import Optional, List, Tuple, Any
from xml.sax.saxutils import escape

import numpy as np

from AnyQt.QtCore import QPointF, Qt, Signal, QRectF, QEvent
from AnyQt.QtGui import QTransform, QPainter, QColor, QPainterPath, \
    QPolygonF, QMouseEvent
from AnyQt.QtWidgets import QComboBox, QApplication, QGraphicsSceneMouseEvent

import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.base import Model
from Orange.data import Table, Domain, ContinuousVariable, Variable
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
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import AxisItem
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

                (x1, x2), _ = self.__data_bounds
                p1.setX(max(min(p1.x(), x2), x1))
                p2.setX(max(min(p2.x(), x2), x1))
                _, (y1, y2) = self.viewRange()
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

    def mousePressEvent(self, ev: QGraphicsSceneMouseEvent):
        keys = QApplication.keyboardModifiers()
        if self.__state == SELECT and not keys & Qt.ShiftModifier:
            ev.accept()
            self.sigDeselect.emit()
        super().mousePressEvent(ev)

    def mouseClickEvent(self, ev: MouseClickEvent):
        if self.__state == SELECT:
            ev.accept()
            self.sigDeselect.emit()
        else:
            super().mouseClickEvent(ev)


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
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
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


class FillBetweenItem(pg.FillBetweenItem):
    def __init__(self, rgb: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__rgb = rgb

    @property
    def rgb(self) -> List[int]:
        return self.__rgb


class ForcePlot(pg.PlotWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: OWWidget):
        self.__data_bounds: Optional[Tuple[Tuple[float, float],
                                           Tuple[float, float]]] = None
        self.__pos_labels: Optional[List[str]] = None
        self.__neg_labels: Optional[List[str]] = None
        self.__tooltip_data: Optional[Table] = None
        self.__show_tooltips = True
        self.__highlight_feature = True
        self.__mouse_pressed = False

        self.__fill_items: List[pg.FillBetweenItem] = []
        self.__text_items: List[pg.TextItem] = []
        self.__dot_items: List[pg.ScatterPlotItem] = []
        self.__vertical_line_item: Optional[pg.InfiniteLine] = None
        self.__selection: List = []
        self.__selection_rect_items: List[SelectionRect] = []

        view_box = ForcePlotViewBox()
        view_box.sigSelectionChanged.connect(self._update_selection)
        view_box.sigDeselect.connect(self._deselect)
        view_box.sigRangeChangedManually.connect(self.__on_range_changed_man)
        view_box.sigRangeChanged.connect(self.__on_range_changed)

        super().__init__(parent, viewBox=view_box,
                         background="w", enableMenu=False,
                         axisItems={"bottom": AxisItem("bottom", True),
                                    "left": AxisItem("left")})
        self.setAntialiasing(True)
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().scene().sigMouseMoved.connect(self.__on_mouse_moved)

        self.parameter_setter = ParameterSetter(self)

    def __on_range_changed_man(self):
        scene: pg.GraphicsScene = self.getPlotItem().scene()
        if scene.lastHoverEvent is not None:
            self.__clear_tooltips()
            point = scene.lastHoverEvent.scenePos()
            self.__show_tooltip(self.getViewBox().mapSceneToView(point))

    def __on_range_changed(self):
        _, (y1, y2) = self.getViewBox().viewRange()
        for i in range(len(self.__selection_rect_items)):
            sel_rect_item = self.__selection_rect_items[i]
            self.removeItem(sel_rect_item)

            rect = sel_rect_item.boundingRect()
            rect.setTop(y1)
            rect.setBottom(y2)
            sel_rect_item = SelectionRect(rect)

            self.addItem(sel_rect_item)
            self.__selection_rect_items[i] = sel_rect_item

    def __on_mouse_moved(self, point: QPointF):
        self.__clear_hover()

        view_box: ForcePlotViewBox = self.getViewBox()
        view_pos: QPointF = view_box.mapSceneToView(point)
        (xmin, xmax), (ymin, ymax) = view_box.viewRange()
        in_view = xmin <= view_pos.x() <= xmax and ymin <= view_pos.y() <= ymax
        if not in_view or self.__mouse_pressed:
            return

        self.__hightlight(view_pos)
        self.__show_tooltip(view_pos)

    def set_data(self, x_data: np.ndarray,
                 pos_y_data: List[Tuple[np.ndarray, np.ndarray]],
                 neg_y_data: List[Tuple[np.ndarray, np.ndarray]],
                 pos_labels: List[str], neg_labels: List[str],
                 x_label: str, y_label: str,
                 tooltip_data: Table):

        self.__data_bounds = ((np.nanmin(x_data), np.nanmax(x_data)),
                              (np.nanmin(pos_y_data), np.nanmax(neg_y_data)))
        self.__pos_labels = pos_labels
        self.__neg_labels = neg_labels
        self.__tooltip_data = tooltip_data

        self.getViewBox().set_data_bounds(self.__data_bounds)
        self._set_range()
        self._set_axes(x_label, y_label)
        self._plot_data(x_data, pos_y_data, neg_y_data)

    def _plot_data(self, x_data: np.ndarray,
                   pos_y_data: List[Tuple[np.ndarray, np.ndarray]],
                   neg_y_data: List[Tuple[np.ndarray, np.ndarray]]):
        for rgb, data in ((RGB_HIGH, pos_y_data), (RGB_LOW, neg_y_data)):
            whiter_rgb = np.array(rgb) + (255 - np.array(rgb)) * 0.7
            pen = pg.mkPen(whiter_rgb, width=1)
            brush = pg.mkBrush(rgb)
            for y_top, y_bottom in data:
                fill = FillBetweenItem(
                    rgb, pg.PlotDataItem(x=x_data, y=y_bottom),
                    pg.PlotDataItem(x=x_data, y=y_top), pen=pen, brush=brush
                )
                self.__fill_items.append(fill)
                self.addItem(fill)

    def _set_axes(self, x_label: str, y_label: str):
        bottom_axis: AxisItem = self.getAxis("bottom")
        bottom_axis.setLabel(x_label)
        bottom_axis.resizeEvent(None)

        left_axis: AxisItem = self.getAxis("left")
        left_axis.setLabel(y_label)
        left_axis.resizeEvent(None)

    def set_axis(self, ticks: Optional[List]):
        ax: AxisItem = self.getAxis("bottom")
        ax.setTicks(ticks)

    def set_show_tooltip(self, show: bool):
        self.__show_tooltips = show

    def set_highlight_feature(self, highlight: bool):
        self.__highlight_feature = highlight

    def clear_all(self):
        self.__data_bounds = None
        self.__tooltip_data = None
        self.__fill_items.clear()
        self.__text_items.clear()
        self.__dot_items.clear()
        self.__vertical_line_item = None
        self.getViewBox().set_data_bounds(self.__data_bounds)
        self.clear()
        self._clear_selection()
        self._set_axes(None, None)

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

    def __hightlight(self, point: QPointF):
        if not self.__highlight_feature:
            return
        for index, item in enumerate(self.__fill_items):
            if self._contains_point(item, point):
                n = len(self.__neg_labels)
                if index < n:
                    name = self.__pos_labels[index]
                    index_other = self.__neg_labels.index(name) + n
                else:
                    name = self.__neg_labels[index - n]
                    index_other = self.__pos_labels.index(name)

                for i in (index, index_other):
                    item = self.__fill_items[i]
                    color = QColor(*item.rgb)
                    color = color.darker(120)
                    item.setBrush(pg.mkBrush(color))

                break

    def __show_tooltip(self, point: QPointF):
        if not self.__show_tooltips:
            return
        instance_index = int(round(point.x(), 0))
        if self.__tooltip_data is None or instance_index < 0 or \
                instance_index >= len(self.__tooltip_data):
            return

        instance = self.__tooltip_data[instance_index]
        n_features = len(self.__fill_items) // 2
        pos_fills = self.__fill_items[:n_features]
        neg_fills = self.__fill_items[n_features:]
        pos_labels = self.__pos_labels
        neg_labels = self.__neg_labels

        view_box: ForcePlotViewBox = self.getViewBox()
        px_width, px_height = view_box.viewPixelSize()
        pos = view_box.mapViewToScene(point)
        right_side = view_box.boundingRect().width() / 2 > pos.x()

        self.__vertical_line_item = pg.InfiniteLine(instance_index)
        self.addItem(self.__vertical_line_item)

        for rgb, labels, fill_items in ((RGB_HIGH, pos_labels, pos_fills),
                                        (RGB_LOW, neg_labels, neg_fills)):
            whiter_rgb = np.array(rgb) + (255 - np.array(rgb)) * 0.7
            for i, (label, fill_item) in enumerate(zip(labels, fill_items)):
                curve1, curve2 = fill_item.curves
                y_lower = curve1.curve.getData()[1][instance_index]
                y_upper = curve2.curve.getData()[1][instance_index]
                delta_y = y_upper - y_lower
                text_item = pg.TextItem(
                    text=escape(f"{label} = {instance[label]}"),
                    color=rgb, fill=pg.mkBrush(whiter_rgb)
                )
                height = text_item.boundingRect().height() * px_height * 2
                if height < delta_y or self._contains_point(fill_item, point):
                    if right_side:
                        x_pos = instance_index + px_width * 5
                    else:
                        x_pos = instance_index - px_width * 5
                        x_pos -= px_width * text_item.boundingRect().width()
                    y_pos = y_upper - delta_y / 2

                    text_item.setPos(x_pos, y_pos)
                    self.__text_items.append(text_item)
                    self.addItem(text_item)

                    dot_color = QColor(Qt.white)
                    dot_item = pg.ScatterPlotItem(
                        x=[instance_index], y=[y_pos], size=6,
                        pen=pg.mkPen(dot_color), brush=pg.mkBrush(dot_color)
                    )
                    self.__dot_items.append(dot_item)
                    self.addItem(dot_item)

    def __clear_hover(self):
        for item in self.__fill_items:
            item.setBrush(pg.mkBrush(*item.rgb))
        self.__clear_tooltips()

    def __clear_tooltips(self):
        for item in self.__text_items:
            self.removeItem(item)
        self.__text_items.clear()
        for item in self.__dot_items:
            self.removeItem(item)
        self.__dot_items.clear()
        if self.__vertical_line_item is not None:
            self.removeItem(self.__vertical_line_item)
        self.__vertical_line_item = None

    def mousePressEvent(self, ev: QMouseEvent):
        self.__mouse_pressed = True
        self.__clear_hover()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        super().mouseReleaseEvent(ev)
        self.__mouse_pressed = False

    def leaveEvent(self, ev: QEvent):
        super().leaveEvent(ev)
        self.__clear_hover()

    @staticmethod
    def _contains_point(item: pg.FillBetweenItem, point: QPointF) -> bool:
        curve1, curve2 = item.curves
        x_data_lower, y_data_lower = curve1.curve.getData()
        x_data_upper, y_data_upper = curve2.curve.getData()
        pts = [QPointF(x, y) for x, y in zip(x_data_lower, y_data_lower)]
        pts += [QPointF(x, y) for x, y in
                reversed(list(zip(x_data_upper, y_data_upper)))]
        pts += pts[:1]
        path = QPainterPath()
        path.addPolygon(QPolygonF(pts))
        return path.contains(point)


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
    show_tooltip = Setting(True)
    highlight_feature = Setting(True)
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
        self._order_combo: QComboBox = None
        self._annot_combo: QComboBox = None

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
        self.graph.set_show_tooltip(self.show_tooltip)
        self.graph.set_highlight_feature(self.highlight_feature)
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

        box = gui.vBox(self.controlArea, "", margin=True,
                       contentsMargins=(8, 4, 8, 4))
        gui.checkBox(box, self, "show_tooltip", "Show tooltips",
                     callback=self.__on_show_tooltip_changed)
        gui.checkBox(box, self, "highlight_feature",
                     "Highlight feature on hover",
                     callback=self.__on_highlight_feature_changed)

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

    def __on_show_tooltip_changed(self):
        self.graph.set_show_tooltip(self.show_tooltip)

    def __on_highlight_feature_changed(self):
        self.graph.set_highlight_feature(self.highlight_feature)

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

        order = self._order_combo.model()[self.order_index]
        values_idxs = get_instance_ordering(
            self.__results.values[self.target_index],
            self.__results.predictions[self.__results.mask, self.target_index],
            self.data[self.__results.mask],
            order
        )

        data_idxs = np.arange(len(self.data))
        self.__data_idxs = data_idxs[self.__results.mask][values_idxs]

        x_data, pos_y_data, neg_y_data, pos_labels, neg_labels = \
            prepare_force_plot_data_multi_inst(
                self.__results.values[self.target_index][values_idxs],
                self.__results.base_value[self.target_index],
                self.model.domain
            )

        if self.order_index == 0:
            order = "hierarhical clustering"
        elif self.order_index == 1:
            order = "output value"
        elif self.order_index == 2:
            order = "original ordering"
        x_label = f"Instances ordered by {order}"

        target = self.model.domain.class_var
        if self.model.domain.has_discrete_class:
            target = f"{target} = {target.values[self.target_index]}"
        y_label = f"Output value ({target})"

        self.graph.set_data(x_data, pos_y_data, neg_y_data,
                            pos_labels, neg_labels, x_label, y_label,
                            self.__results.transformed_data[self.__data_idxs])
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
