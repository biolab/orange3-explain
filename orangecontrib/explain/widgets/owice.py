import bisect
from types import SimpleNamespace
from typing import Optional, Dict, List
from xml.sax.saxutils import escape

import numpy as np
from AnyQt.QtCore import Qt, QSortFilterProxyModel, QSize, QModelIndex, \
    QItemSelection, QPointF
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QComboBox, QSizePolicy, QGraphicsSceneHelpEvent, \
    QToolTip

import pyqtgraph as pg

from orangecanvas.gui.utils import disconnected
from orangewidget.utils.listview import ListViewSearch

from Orange.base import Model, SklModel, RandomForestModel
from Orange.data import Table, ContinuousVariable, Variable, \
    DiscreteVariable
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, Setting, \
    PerfectDomainContextHandler
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import VariableListModel, DomainModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.plotutils import PlotWidget, \
    HelpEventDelegate
from Orange.widgets.widget import Input, OWWidget, Msg

from orangecontrib.explain.inspection import individual_condition_expectation


class RunnerResults(SimpleNamespace):
    x_data: Optional[np.ndarray] = None
    y_average: Optional[np.ndarray] = None
    y_individual: Optional[np.ndarray] = None


def run(
        data: Table,
        feature: Variable,
        model: Model,
        state: TaskState
) -> Optional[RunnerResults]:
    if not data or not model or not feature:
        return None

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    result = individual_condition_expectation(
        model, data, feature, progress_callback=callback
    )
    return RunnerResults(x_data=result["values"],
                         y_average=result["average"],
                         y_individual=result["individual"])


class SortProxyModel(QSortFilterProxyModel):
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        role = self.sortRole()
        l_score = left.data(role)
        r_score = right.data(role)
        return r_score is not None and (l_score is None or l_score < r_score)


class ICEPlot(PlotWidget):
    DEFAULT_COLOR = np.array([100, 100, 100])
    MAX_POINTS_IN_TOOLTIP = 5

    def __init__(self, parent: OWWidget):
        super().__init__(parent, enableMenu=False)
        self.setAntialiasing(True)
        self.setMouseEnabled(False, False)
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().scene().sigMouseMoved.connect(self.__on_mouse_moved)

        self.__data: Table = None
        self.__feature: ContinuousVariable = None
        self.__x_data: Optional[np.ndarray] = None
        self.__y_individual: Optional[np.ndarray] = None
        self.__lines_items: List[pg.PlotCurveItem] = []
        self.__mean_line_item: Optional[pg.PlotCurveItem] = None
        self.__hovered_lines_item: Optional[pg.PlotCurveItem] = None
        self.__hovered_scatter_item: Optional[pg.ScatterPlotItem] = None

        self._help_delegate = HelpEventDelegate(self._help_event)
        self.scene().installEventFilter(self._help_delegate)

    def __on_mouse_moved(self, point: QPointF):
        if self.__hovered_lines_item is None:
            return

        self.__hovered_lines_item.setData(None, None, connect=None)
        self.__hovered_scatter_item.setData(None, None)

        view_pos: QPointF = self.getViewBox().mapSceneToView(point)
        indices = self._indices_at(view_pos)
        if not indices:
            return

        # lines
        y_individual = self.__y_individual[indices]
        connect = np.ones(y_individual.shape)
        connect[:, -1] = 0
        self.__hovered_lines_item.setData(
            np.tile(self.__x_data, len(y_individual)),
            y_individual.flatten(),
            connect=connect.flatten()
        )

        # points
        x_data = self.__data.get_column_view(self.__feature)[0][indices]
        n_dec = self.__feature.number_of_decimals
        y_data = []
        for i, x in zip(indices, x_data):
            mask = np.round(self.__x_data, n_dec) == round(x, n_dec)
            idx = np.flatnonzero(mask)
            y = self.__y_individual[i, idx[0]] if len(idx) > 0 else np.nan
            y_data.append(y)

        y_data = np.array(y_data)
        mask = ~np.isnan(y_data)
        self.__hovered_scatter_item.setData(x_data[mask], y_data[mask])

    def set_data(
            self,
            data: Table,
            feature: ContinuousVariable,
            x_data: np.ndarray,
            y_average: np.ndarray,
            y_individual: np.ndarray,
            y_label: str,
            colors: Optional[np.ndarray],
            color_col: Optional[np.ndarray],
            show_mean: bool,
    ):
        self.__data = data
        self.__feature = feature
        self.__x_data = x_data
        self.__y_individual = y_individual
        self._add_lines(y_average, show_mean, colors, color_col)
        self._set_axes(feature.name, y_label)

    def _set_axes(self, x_label: str, y_label: str):
        self.getAxis("bottom").setLabel(x_label)
        self.getAxis("left").setLabel(y_label)

    def _add_lines(
            self,
            y_average: np.ndarray,
            show_mean: bool,
            colors: np.ndarray,
            color_col: np.ndarray,
    ):
        if colors is None:
            colors = self.DEFAULT_COLOR[None, :]
            color_col = np.zeros(len(self.__y_individual))

        x_data = self.__x_data
        for i, color in enumerate(colors):
            y_data = self.__y_individual[color_col == i]
            self.__add_curve_item(x_data, y_data, color)

        mask = np.isnan(color_col)
        if any(mask):
            y_data = self.__y_individual[mask]
            self.__add_curve_item(x_data, y_data, self.DEFAULT_COLOR)

        width = 3
        color = QColor("#1f77b4")
        self.__hovered_lines_item = pg.PlotCurveItem(
            pen=pg.mkPen(color, width=width), antialias=True
        )
        self.addItem(self.__hovered_lines_item)

        size = 8
        self.__hovered_scatter_item = pg.ScatterPlotItem(
            pen=color, brush=color, size=8, shape="o"
        )
        self.addItem(self.__hovered_scatter_item)

        self.__mean_line_item = pg.PlotCurveItem(
            x_data, y_average,
            pen=pg.mkPen(color=QColor("#ffbe00"), width=5),
            antialias=True
        )
        self.addItem(self.__mean_line_item)
        self.set_show_mean(show_mean)

        color = QColor(0, 0, 0, 0)
        dummy = pg.ScatterPlotItem(
            [np.min(x_data), np.max(x_data)],
            [np.min(self.__y_individual), np.max(self.__y_individual)],
            pen=color, brush=color, size=size, shape="o"
        )
        self.addItem(dummy)

    def set_show_mean(self, show: bool):
        if self.__mean_line_item is not None:
            self.__mean_line_item.setVisible(show)

    def clear_all(self):
        self.__data = None
        self.__feature = None
        self.__x_data = None
        self.__y_individual = None
        if self.__mean_line_item is not None:
            self.removeItem(self.__mean_line_item)
            self.__mean_line_item = None
        for lines in self.__lines_items:
            self.removeItem(lines)
        self.__lines_items.clear()
        if self.__hovered_lines_item is not None:
            self.removeItem(self.__hovered_lines_item)
            self.__hovered_lines_item = None
        if self.__hovered_scatter_item is not None:
            self.removeItem(self.__hovered_scatter_item)
            self.__hovered_scatter_item = None
        self.clear()
        self._set_axes(None, None)

    def __add_curve_item(self, x_data, y_data, color):
        connect = np.ones(y_data.shape)
        connect[:, -1] = 0
        lines = pg.PlotCurveItem(
            np.tile(x_data, len(y_data)), y_data.flatten(),
            connect=connect.flatten(), antialias=True,
            pen=pg.mkPen(color=QColor(*color, 100), width=1)
        )
        self.addItem(lines)
        self.__lines_items.append(lines)

    def _indices_at(self, pos: QPointF) -> List[int]:
        if not self.__x_data[0] <= pos.x() <= self.__x_data[-1]:
            return []

        index = bisect.bisect(self.__x_data, round(pos.x(), 2)) - 1
        assert 0 <= index < len(self.__x_data)
        if index < len(self.__x_data) - 1:
            x = pos.x()
            x_left = self.__x_data[index]
            x_right = self.__x_data[index + 1]
            y_left = self.__y_individual[:, index]
            y_right = self.__y_individual[:, index + 1]
            y = (x - x_left) * (y_right - y_left) / (x_right - x_left) + y_left

        else:
            y = self.__y_individual[:, -1]

        # TODO - eps depends on pixel-view ratio
        eps = (np.nanmax(self.__y_individual) -
               np.nanmin(self.__y_individual)) / 1000
        mask = np.abs(y - pos.y()) < eps
        return np.flatnonzero(mask).tolist()

    def _help_event(self, event: QGraphicsSceneHelpEvent):
        if self.__mean_line_item is None:
            return False

        pos = self.__mean_line_item.mapFromScene(event.scenePos())
        indices = self._indices_at(pos)
        text = self._get_tooltip(indices)
        if text:
            QToolTip.showText(event.screenPos(), text, widget=self)
            return True
        return False

    def _get_tooltip(self, indices: List[int]) -> str:
        text = "<hr/>".join(self.__instance_tooltip(self.__data, idx) for idx
                            in indices[:self.MAX_POINTS_IN_TOOLTIP])
        if len(indices) > self.MAX_POINTS_IN_TOOLTIP:
            text = f"{len(indices)} instances<hr/>{text}<hr/>..."
        return text

    @staticmethod
    def __instance_tooltip(data: Table, idx: int) -> str:
        def show_part(_point_data, singular, plural, max_shown, _vars):
            cols = [escape('{} = {}'.format(var.name, _point_data[var]))
                    for var in _vars[:max_shown + 2]][:max_shown]
            if not cols:
                return ""
            n_vars = len(_vars)
            if n_vars > max_shown:
                cols[-1] = "... and {} others".format(n_vars - max_shown + 1)
            return \
                "<b>{}</b>:<br/>".format(singular if n_vars < 2 else plural) \
                + "<br/>".join(cols)

        parts = (("Class", "Classes", 4, data.domain.class_vars),
                 ("Meta", "Metas", 4, data.domain.metas),
                 ("Feature", "Features", 10, data.domain.attributes))
        return "<br/>".join(show_part(data[idx], *cols) for cols in parts)


class OWICE(OWWidget, ConcurrentWidgetMixin):
    name = "ICE"
    description = "Dependence between a target and a feature of interest."
    keywords = ["ICE", "PDP", "partial", "dependence"]
    icon = "icons/ICE.svg"
    priority = 130

    class Inputs:
        model = Input("Model", (SklModel, RandomForestModel))
        data = Input("Data", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("{}")
        not_enough_data = Msg("At least two instances are needed.")
        no_cont_features = Msg("At least one numeric feature is required.")

    class Information(OWWidget.Information):
        data_sampled = Msg("Data has been sampled.")

    buttons_area_orientation = Qt.Vertical

    settingsHandler = PerfectDomainContextHandler()
    target_index = ContextSetting(0)
    feature = ContextSetting(None)
    order_by_importance = Setting(False)
    color_var = ContextSetting(None)
    centered = Setting(True)
    show_mean = Setting(True)
    # auto_send = Setting(True)

    graph_name = "graph.plotItem"
    MIN_INSTANCES = 2
    MAX_INSTANCES = 300

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.__results: Optional[RunnerResults] = None
        self.__results_avgs: Optional[Dict[ContinuousVariable, float]] = None
        self.model: Optional[Model] = None
        self.data: Optional[Table] = None
        self.graph: ICEPlot = None
        self._target_combo: QComboBox = None
        self._features_view: ListViewSearch = None
        self._features_model: VariableListModel = None
        self._color_model: DomainModel = None

        self.setup_gui()

    def setup_gui(self):
        self._add_plot()
        self._add_controls()
        # self._add_buttons()

    def _add_plot(self):
        box = gui.vBox(self.mainArea)
        self.graph = ICEPlot(self)
        box.layout().addWidget(self.graph)

    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Target class")
        self._target_combo = gui.comboBox(
            box, self, "target_index", contentsLength=12,
            callback=self.__on_target_changed
        )

        box = gui.vBox(self.controlArea, "Feature")
        self._features_model = VariableListModel()
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self._features_model)
        sorted_model.sort(0)
        self._features_view = ListViewSearch()
        self._features_view.setModel(sorted_model)
        self._features_view.setMinimumSize(QSize(30, 100))
        self._features_view.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Expanding)
        self._features_view.selectionModel().selectionChanged.connect(
            self.__on_feature_changed
        )
        box.layout().addWidget(self._features_view)
        gui.checkBox(box, self, "order_by_importance", "Order by importance",
                     callback=self.__on_order_changed)

        box = gui.vBox(self.controlArea, "Display")
        self._color_model = DomainModel(placeholder="None", separators=False,
                                        valid_types=DiscreteVariable)
        gui.comboBox(box, self, "color_var", label="Color:", searchable=True,
                     model=self._color_model, orientation=Qt.Horizontal,
                     contentsLength=12, callback=self.__on_color_var_changed)
        gui.checkBox(box, self, "centered", "Centered",
                     callback=self.__on_centered_changed)
        gui.checkBox(box, self, "show_mean", "Show mean",
                     callback=self.__on_show_mean_changed)

    def __on_target_changed(self):
        self._apply_feature_sorting()
        self.setup_plot()

    def __on_feature_changed(self, selection: QItemSelection):
        if not selection:
            return

        self.feature = selection.indexes()[0].data(gui.TableVariable)
        self._apply_feature_sorting()
        self._run()

    def __on_order_changed(self):
        self._apply_feature_sorting()

    def __on_color_var_changed(self):
        self.setup_plot()

    def __on_centered_changed(self):
        self.setup_plot()

    def __on_show_mean_changed(self):
        self.graph.set_show_mean(self.show_mean)

    def _add_buttons(self):
        gui.auto_send(self.buttonsArea, self, "auto_send")

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.data = data
        self._check_data()
        self._setup_controls()
        self.openContext(self.data.domain if self.data else None)
        self.set_list_view_selection()

    @Inputs.model
    def set_model(self, model: Optional[Model]):
        self.model = model

    def _check_data(self):
        self.Error.no_cont_features.clear()
        self.Error.not_enough_data.clear()
        self.Information.data_sampled.clear()
        if self.data is None:
            return

        if len(self.data) < self.MIN_INSTANCES:
            self.data = None
            self.Error.not_enough_data()

        if self.data and not self.data.domain.has_continuous_attributes():
            self.data = None
            self.Error.no_cont_features()

        if self.data and len(self.data) > self.MAX_INSTANCES:
            kwargs = {"size": self.MAX_INSTANCES, "replace": False}
            np.random.seed(0)
            indices = np.random.choice(len(self.data), **kwargs)
            self.data = self.data[indices]
            self.Information.data_sampled()

    def _setup_controls(self):
        domain = self.data.domain if self.data else None

        self._target_combo.clear()
        self._target_combo.setEnabled(True)
        self._features_model.clear()
        self._color_model.set_domain(domain)
        self.color_var = None

        if domain is not None:
            features = [var for var in domain.attributes if var.is_continuous
                        and not var.attributes.get("hidden", False)]
            self._features_model[:] = features
            if domain.has_discrete_class:
                self._target_combo.addItems(domain.class_var.values)
                self.target_index = 0
            elif domain.has_continuous_class:
                self._target_combo.setEnabled(False)
                self.target_index = -1
            if len(self._features_model) > 0:
                self.feature = self._features_model[0]

    def set_list_view_selection(self):
        model = self._features_view.model()
        sel_model = self._features_view.selectionModel()
        src_model = model.sourceModel()
        if self.feature not in src_model:
            return

        with disconnected(sel_model.selectionChanged,
                          self.__on_feature_changed):
            row = src_model.indexOf(self.feature)
            sel_model.select(model.index(row, 0), sel_model.ClearAndSelect)

        self._ensure_selection_visible(self._features_view)

    @staticmethod
    def _ensure_selection_visible(view):
        selection = view.selectedIndexes()
        if len(selection) == 1:
            view.scrollTo(selection[0])

    def handleNewSignals(self):
        self.__results_avgs = None
        self._apply_feature_sorting()
        self._run()

    def _apply_feature_sorting(self):
        if self.data is None or self.model is None:
            return

        order = list(range(len(self._features_model)))
        if self.order_by_importance:
            def compute_score(feature):
                values = self.__results_avgs[feature][self.target_index]
                return -np.sum(np.abs(values - np.mean(values)))

            try:
                if self.__results_avgs is None:
                    self.__results_avgs = {
                        feature: individual_condition_expectation(
                            self.model, self.data, feature, kind="average"
                        )["average"] for feature in self._features_model
                    }
                order = [compute_score(f) for f in self._features_model]
            except Exception:
                pass

        for i in range(self._features_model.rowCount()):
            self._features_model.setData(self._features_model.index(i),
                                         order[i], Qt.UserRole)

        self._ensure_selection_visible(self._features_view)

    def _run(self):
        self.clear()
        self.start(run, self.data, self.feature, self.model)

    def clear(self):
        self.__results = None
        self.cancel()
        self.Error.domain_transform_err.clear()
        self.Error.unknown_err.clear()
        self.graph.clear_all()

    def setup_plot(self):
        self.graph.clear_all()
        if not self.__results or not self.data:
            return

        x_data = self.__results.x_data
        y_average = self.__results.y_average[self.target_index]
        y_individual = self.__results.y_individual[self.target_index]

        if self.centered:
            y_average = y_average - y_average[0, None]
            y_individual = y_individual - y_individual[:, 0, None]

        class_var: Variable = self.model.original_domain.class_var
        postfix = f"={class_var.values[self.target_index]} probability" \
            if class_var.is_discrete else ""

        colors = None
        color_col = None
        if self.color_var and self.color_var.is_discrete:
            colors = self.color_var.colors
            color_col = self.data.get_column_view(self.color_var)[0]

        self.graph.set_data(self.data, self.feature,
                            x_data, y_average, y_individual,
                            f"Predicted {class_var.name}{postfix}",
                            colors, color_col, self.show_mean)

    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[RunnerResults]):
        self.__results = results
        self.setup_plot()

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def send_report(self):
        if not self.data or not self.model:
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

    table = Table("iris")
    kwargs_ = {"n_estimators": 100, "random_state": 0}
    if table.domain.has_continuous_class:
        model_ = RandomForestRegressionLearner(**kwargs_)(table)
    else:
        model_ = RandomForestLearner(**kwargs_)(table)
    WidgetPreview(OWICE).run(set_data=table, set_model=model_)
