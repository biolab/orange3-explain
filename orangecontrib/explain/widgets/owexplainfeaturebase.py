# pylint: disable=missing-docstring,no-name-in-module,invalid-name
# pylint: disable=too-few-public-methods
from typing import Tuple, Optional, List, Dict, Union
from types import SimpleNamespace

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSizeF, QSize, pyqtSignal as Signal
from AnyQt.QtGui import QPen, QPainter, QFont, QFontMetrics, QResizeEvent, \
    QColor
from AnyQt.QtWidgets import QGraphicsItemGroup, QGraphicsLineItem, \
    QGraphicsScene, QGraphicsWidget, QGraphicsGridLayout, QSizePolicy, \
    QGraphicsSimpleTextItem, QGraphicsSceneMouseEvent, QGraphicsRectItem

from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.base import Model
from Orange.data import Table
from Orange.data.table import DomainTransformationError
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.graphicslayoutitem import SimpleLayoutItem
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.state_summary import format_summary_details
from Orange.widgets.utils.stickygraphicsview import StickyGraphicsView
from Orange.widgets.visualize.utils.customizableplot import \
    CommonParameterSetter, Updater
from Orange.widgets.visualize.utils.plotutils import AxisItem
from Orange.widgets.widget import Input, Output, OWWidget, Msg

MAX_N_ITEMS = 100


class BaseResults(SimpleNamespace):
    x = None
    names = None
    mask = None


class SelectionRect(QGraphicsRectItem):
    COLOR = [255, 255, 0]

    def __init__(self, parent):
        super().__init__(parent)

        color = QColor(*self.COLOR)
        color.setAlpha(100)
        self.setBrush(color)

        color = QColor(*self.COLOR)
        self.setPen(color)


class VariableItem(QGraphicsItemGroup):
    MAX_ATTR_LEN = 25
    MAX_LABEL_LEN = 150
    VALUE_FONT_SETTING = {Updater.SIZE_LABEL: 12,
                          Updater.IS_ITALIC_LABEL: True}

    def __init__(self, parent, label: str):
        self.__name: Optional[str] = None
        self.__value: Optional[str] = None
        self.__name_item = QGraphicsSimpleTextItem()
        self.__value_item = QGraphicsSimpleTextItem()
        font = Updater.change_font(QFont(), self.VALUE_FONT_SETTING)
        self.__value_item.setFont(font)
        self.__max_len = self.MAX_LABEL_LEN
        super().__init__(parent)
        self._set_data(label)

    @property
    def items(self) -> Tuple[QGraphicsSimpleTextItem, QGraphicsSimpleTextItem]:
        return self.__name_item, self.__value_item

    def boundingRect(self) -> QRectF:
        name_br = self.__name_item.boundingRect()
        width = name_br.width()
        height = name_br.height()
        if self.__value_item is not None:
            value_br = self.__value_item.boundingRect()
            width = max(width, value_br.width())
            height += value_br.height()
        return QRectF(-width, 0, width, height)

    def updateGeometry(self):
        self.__elide()
        self.__align_center()
        self.__align_right()

    def set_max_len(self, length: int):
        self.__max_len = length
        self.updateGeometry()

    def _set_data(self, label: str):
        split = label.split("=")
        self.__name = split[0]
        self.__name_item.setToolTip(self.__name)
        self.updateGeometry()  # align before adding to group
        self.addToGroup(self.__name_item)
        if len(split) > 1:
            self.__value = split[1]
            self.__value_item.setToolTip(self.__value)
            self.updateGeometry()  # align before adding to group
            self.addToGroup(self.__value_item)

    def __elide(self):
        fm = QFontMetrics(self.__name_item.font())
        text = fm.elidedText(self.__name, Qt.ElideRight, self.__max_len)
        self.__name_item.setText(text)
        if self.__value is not None:
            fm = QFontMetrics(self.__value_item.font())
            text = fm.elidedText(self.__value, Qt.ElideRight, self.__max_len)
            self.__value_item.setText(text)

    def __align_center(self):
        if self.__value is not None:
            self.__value_item.setY(self.__name_item.boundingRect().height())

    def __align_right(self):
        self.__name_item.setX(-self.__name_item.boundingRect().width())
        if self.__value is not None:
            self.__value_item.setX(-self.__value_item.boundingRect().width())


class FeatureItem(QGraphicsWidget):
    HEIGHT = 50

    def __init__(self, parent, attr_name: str, x_range: Tuple[float],
                 width: int):
        super().__init__(parent)
        self._attr_name = attr_name
        self._width = width
        self._height = self.HEIGHT
        self._range = x_range
        self._x_data: Optional[Union[np.ndarray, float]] = None
        self._group = QGraphicsItemGroup(self)

    @property
    def attr_name(self) -> str:
        return self._attr_name

    def set_data(self, x_data: np.ndarray, *_):
        raise NotImplementedError

    def rescale(self, width: float):
        raise NotImplementedError

    def set_height(self, height: float):
        raise NotImplementedError

    def sizeHint(self, *_) -> QSizeF:
        return QSizeF(self._width, self._height)


class BaseParameterSetter(CommonParameterSetter):
    VAR_LABEL = "Variable name"
    VAL_LABEL = "Variable value"
    LABEL_LENGTH = "Label length"

    def __init__(self, parent):
        self.value_label_font = Updater.change_font(
            QFont(), VariableItem.VALUE_FONT_SETTING
        )
        self.label_len_setting = {
            self.LABEL_LENGTH: VariableItem.MAX_LABEL_LEN
        }
        super().__init__()
        self.master: FeaturesPlot = parent

    def update_setters(self):
        def _update_labels(font, index):
            for item in self.labels:
                var_items = item.item
                text_item = var_items.items[index]
                if text_item is not None:
                    text_item.setFont(font)
                    var_items.updateGeometry()
                    item.updateGeometry()
            self.master.set_vertical_line()

        def update_name_label(**settings):
            self.label_font = Updater.change_font(self.label_font, settings)
            _update_labels(self.label_font, 0)

        def update_value_label(**settings):
            self.value_label_font = \
                Updater.change_font(self.value_label_font, settings)
            _update_labels(self.value_label_font, 1)

        def update_label_len(**settings):
            self.label_len_setting.update(settings)
            max_len = self.label_len_setting[self.LABEL_LENGTH]
            for item in self.labels:
                var_items: VariableItem = item.item
                var_items.set_max_len(max_len)
                item.updateGeometry()
            self.master.resized.emit()

        font_size = VariableItem.VALUE_FONT_SETTING[Updater.SIZE_LABEL]
        is_italic = VariableItem.VALUE_FONT_SETTING[Updater.IS_ITALIC_LABEL]
        value_font_setting = {
            Updater.SIZE_LABEL: (range(4, 50), font_size),
            Updater.IS_ITALIC_LABEL: (None, is_italic)
        }
        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.VAR_LABEL: self.FONT_SETTING,
                self.VAL_LABEL: value_font_setting,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
            },
            self.PLOT_BOX: {
                self.LABEL_LENGTH: {
                    self.LABEL_LENGTH: (range(0, 500, 5),
                                        VariableItem.MAX_LABEL_LEN)
                },
            }
        }

        self._setters[self.LABELS_BOX][self.VAR_LABEL] = update_name_label
        self._setters[self.LABELS_BOX][self.VAL_LABEL] = update_value_label
        self._setters[self.PLOT_BOX] = {
            self.LABEL_LENGTH: update_label_len,
        }

    @property
    def axis_items(self) -> List[AxisItem]:
        return [self.master.bottom_axis]

    @property
    def labels(self) -> List[SimpleLayoutItem]:
        return self.master.labels


class FeaturesPlot(QGraphicsWidget):
    BOTTOM_AXIS_LABEL = "Feature Importance"
    LABEL_COLUMN, ITEM_COLUMN = range(2)
    ITEM_COLUMN_WIDTH, OFFSET = 300, 80
    selection_cleared = Signal()
    selection_changed = Signal(object)
    resized = Signal()

    def __init__(self):
        super().__init__()
        self._item_column_width = self.ITEM_COLUMN_WIDTH
        self._range: Optional[Tuple[float, float]] = None
        self._items: List[FeatureItem] = []
        self._variable_items: List[VariableItem] = []
        self._bottom_axis = AxisItem(parent=self, orientation="bottom",
                                     maxTickLength=7, pen=QPen(Qt.black))
        self._bottom_axis.setLabel(self.BOTTOM_AXIS_LABEL)
        self._vertical_line = QGraphicsLineItem(self._bottom_axis)
        self._vertical_line.setPen(QPen(Qt.gray))

        self._layout = QGraphicsGridLayout()
        self._layout.setVerticalSpacing(0)
        self.setLayout(self._layout)

        self.parameter_setter = BaseParameterSetter(self)

    @property
    def item_column_width(self) -> int:
        return self._item_column_width

    @item_column_width.setter
    def item_column_width(self, view_width: int):
        j = FeaturesPlot.LABEL_COLUMN
        w = max([self._layout.itemAt(i, j).item.boundingRect().width()
                 for i in range(len(self._items))] + [0])
        width = view_width - self.OFFSET - w
        self._item_column_width = max(self.ITEM_COLUMN_WIDTH, width)

    @property
    def x0_scaled(self) -> float:
        min_max = self._range[1] - self._range[0]
        return - self._range[0] * self.item_column_width / min_max

    @property
    def bottom_axis(self) -> AxisItem:
        return self._bottom_axis

    @property
    def labels(self) -> List[VariableItem]:
        return self._variable_items

    def set_data(self, x: np.ndarray, names: List[str], n_attrs: int,
                 view_width: int, *plot_args):
        self.item_column_width = view_width
        self._set_range(x, *plot_args)
        self._set_items(x, names, *plot_args)
        self._set_labels(names)
        self._set_bottom_axis()
        self.set_n_visible(n_attrs)

    def _set_range(self, *_):
        raise NotImplementedError

    def _set_items(self, *_):
        raise NotImplementedError

    def set_n_visible(self, n: int):
        for i in range(len(self._items)):
            item = self._layout.itemAt(i, FeaturesPlot.ITEM_COLUMN)
            item.setVisible(i < n)
            text_item = self._layout.itemAt(i, FeaturesPlot.LABEL_COLUMN).item
            text_item.setVisible(i < n)
        self.set_vertical_line()

    def rescale(self, view_width: int):
        self.item_column_width = view_width
        for item in self._items:
            item.rescale(self.item_column_width)

        self._bottom_axis.setWidth(self.item_column_width)
        x = self.x0_scaled
        self._vertical_line.setLine(x, 0, x, self._vertical_line.line().y2())
        self.updateGeometry()

    def set_height(self, height: float):
        for i in range(len(self._items)):
            item = self._layout.itemAt(i, FeaturesPlot.ITEM_COLUMN)
            item.set_height(height)
        self.set_vertical_line()
        self.updateGeometry()

    def _set_labels(self, labels: List[str]):
        for i, (label, _) in enumerate(zip(labels, self._items)):
            text = VariableItem(self, label)
            item = SimpleLayoutItem(text)
            item.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self._layout.addItem(item, i, FeaturesPlot.LABEL_COLUMN,
                                 Qt.AlignRight | Qt.AlignVCenter)
            self._variable_items.append(item)

    def _set_bottom_axis(self):
        self._bottom_axis.setRange(*self._range)
        self._layout.addItem(self._bottom_axis,
                             len(self._items),
                             FeaturesPlot.ITEM_COLUMN)

    def set_vertical_line(self):
        height = 0
        for i in range(len(self._items)):
            item = self._layout.itemAt(i, FeaturesPlot.ITEM_COLUMN)
            text_item = self._layout.itemAt(i, FeaturesPlot.LABEL_COLUMN).item
            if item.isVisible():
                height += max(text_item.boundingRect().height(),
                              item.preferredSize().height())
        self._vertical_line.setLine(self.x0_scaled, 0, self.x0_scaled, -height)

    def deselect(self):
        self.selection_cleared.emit()

    def select(self, *args):
        self.selection_changed.emit(*args)

    def select_from_settings(self, *_):
        raise NotImplementedError

    def apply_visual_settings(self, settings: Dict):
        for key, value in settings.items():
            self.parameter_setter.set_parameter(key, value)


class GraphicsScene(QGraphicsScene):
    mouse_clicked = Signal(object)

    def mousePressEvent(self, ev: QGraphicsSceneMouseEvent):
        self.mouse_clicked.emit(ev)
        super().mousePressEvent(ev)


class GraphicsView(StickyGraphicsView):
    resized = Signal()

    def resizeEvent(self, ev: QResizeEvent):
        if ev.size().width() != ev.oldSize().width():
            self.resized.emit()
        super().resizeEvent(ev)


class OWExplainFeatureBase(OWWidget, ConcurrentWidgetMixin, openclass=True):
    class Inputs:
        data = Input("Data", Table, default=True)
        model = Input("Model", Model)

    class Outputs:
        selected_data = Output("Selected Data", Table)
        scores = Output("Scores", Table)

    class Error(OWWidget.Error):
        domain_transform_err = Msg("{}")
        unknown_err = Msg("An error occurred.\n{}")

    class Information(OWWidget.Information):
        data_sampled = Msg("Data has been sampled.")

    settingsHandler = NotImplemented
    n_attributes = Setting(10)
    zoom_level = Setting(0)
    selection = Setting((), schema_only=True)
    auto_send = Setting(True)
    visual_settings = Setting({}, schema_only=True)

    graph_name = "scene"
    PLOT_CLASS = FeaturesPlot

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.results: Optional[BaseResults] = None
        self.data: Optional[Table] = None
        self.model: Optional[Model] = None
        self.plot: Optional[FeaturesPlot] = None
        self.scene: Optional[GraphicsScene] = None
        self.view: Optional[GraphicsView] = None
        self.setup_gui()
        self.__pending_selection = self.selection

        initial = self.PLOT_CLASS().parameter_setter.initial_settings
        VisualSettingsDialog(self, initial)

    # GUI setup
    def setup_gui(self):
        self._add_controls()
        self._add_plot()
        self._add_buttons()
        self.info.set_input_summary(self.info.NoInput)
        self.info.set_output_summary(self.info.NoOutput)

    def _add_plot(self):
        self.scene = GraphicsScene()
        self.view = GraphicsView(self.scene)
        self.view.resized.connect(self._update_plot)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.mainArea.layout().addWidget(self.view)

    def _add_controls(self):
        self.display_box = box = gui.vBox(self.controlArea, "Display")
        gui.spin(box, self, "n_attributes", 1, MAX_N_ITEMS,
                 label="Top features:", controlWidth=50,
                 callback=self.__n_spin_changed)
        gui.hSlider(box, self, "zoom_level", minValue=0, maxValue=200,
                    width=150, label="Zoom:", createLabel=False,
                    callback=self.__zoom_slider_changed)

    def _add_buttons(self):
        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_send")

    def __zoom_slider_changed(self):
        if self.plot is not None:
            self.plot.set_height(self.zoom_level)

    def __n_spin_changed(self):
        if self.plot is not None:
            self.plot.set_n_visible(self.n_attributes)

    # Inputs
    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.data = data
        summary = len(data) if data else self.info.NoInput
        details = format_summary_details(data) if data else ""
        self.info.set_input_summary(summary, details)
        self._check_data()

    def _check_data(self):
        pass

    @Inputs.model
    def set_model(self, model: Optional[Model]):
        self.closeContext()
        self.model = model
        self.setup_controls()
        self.openContext(self.model)

    def setup_controls(self):
        pass

    def handleNewSignals(self):
        self.clear()
        self.start(self.run, *self.get_runner_parameters())

    def get_runner_parameters(self) -> Tuple:
        return self.data, self.model

    def clear(self):
        self.results = None
        self.cancel()
        self._clear_selection()
        self._clear_scene()
        self.Error.domain_transform_err.clear()
        self.Error.unknown_err.clear()
        self.Information.data_sampled.clear()

    # Plot setup
    def _clear_scene(self):
        self.scene.clear()
        self.scene.setSceneRect(QRectF())
        self.view.setSceneRect(QRectF())
        self.view.setHeaderSceneRect(QRectF())
        self.view.setFooterSceneRect(QRectF())
        self.plot = None

    def update_scene(self):
        self._clear_scene()

    def setup_plot(self, values: np.ndarray, names: List[str], *plot_args):
        width = int(self.view.viewport().rect().width())
        self.plot = self.PLOT_CLASS()
        self.plot.set_data(values, names, self.n_attributes, width, *plot_args)
        self.plot.apply_visual_settings(self.visual_settings)
        self.plot.selection_cleared.connect(self._clear_selection)
        self.plot.selection_changed.connect(self.update_selection)
        self.plot.layout().activate()
        self.plot.geometryChanged.connect(self._update_scene_rect)
        self.plot.resized.connect(self._update_plot)
        self.scene.addItem(self.plot)
        self.scene.mouse_clicked.connect(self.plot.deselect)
        self._update_scene_rect()
        self._update_plot()

    def _update_scene_rect(self):
        def extend_horizontal(rect):
            rect = QRectF(rect)
            rect.setLeft(geom.left())
            rect.setRight(geom.right())
            return rect

        geom = self.plot.geometry()
        self.scene.setSceneRect(geom)
        self.view.setSceneRect(geom)

        footer_geom = self.plot.bottom_axis.geometry()
        footer = extend_horizontal(footer_geom.adjusted(0, -3, 0, 10))
        self.view.setFooterSceneRect(footer)

    def _update_plot(self):
        if self.plot is not None:
            width = int(self.view.viewport().rect().width())
            self.plot.rescale(width)

    # Selection
    def _clear_selection(self):
        if self.selection:
            self.selection = ()
            self.commit()

    def update_selection(self, *_):
        raise NotImplementedError

    def select_pending(self, pending_selection: Tuple):
        self.__pending_selection = pending_selection
        self.unconditional_commit()

    # Outputs
    def commit(self):
        selected_data = self.get_selected_data()
        if not selected_data:
            self.info.set_output_summary(self.info.NoOutput)
        else:
            detail = format_summary_details(selected_data)
            self.info.set_output_summary(len(selected_data), detail)
        self.Outputs.selected_data.send(selected_data)

    def get_selected_data(self) -> Optional[Table]:
        raise NotImplementedError

    def update_scores(self):
        scores = None
        if self.results is not None:
            scores = self.get_scores_table()
        self.Outputs.scores.send(scores)

    def get_scores_table(self) -> Table:
        raise NotImplementedError

    # Concurrent
    def on_partial_result(self, _):
        pass

    def on_done(self, results: Optional[BaseResults]):
        self.results = results
        if self.data and results is not None and not all(results.mask):
            self.Information.data_sampled()
        self.update_scene()
        self.update_scores()
        self.select_pending(self.__pending_selection)

    def on_exception(self, ex: Exception):
        if isinstance(ex, DomainTransformationError):
            self.Error.domain_transform_err(ex)
        else:
            self.Error.unknown_err(ex)

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    # Misc
    def sizeHint(self) -> QSizeF:
        sh = self.controlArea.sizeHint()
        return sh.expandedTo(QSize(800, 520))

    def send_report(self):
        if not self.data or not self.model:
            return
        self.report_plot()

    def set_visual_settings(self, key, value):
        self.visual_settings[key] = value
        if self.plot is not None:
            self.plot.parameter_setter.set_parameter(key, value)

    @staticmethod
    def run(data: Table, model: Model, *, state: TaskState) -> BaseResults:
        raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    from Orange.classification import RandomForestLearner
    from Orange.widgets.settings import DomainContextHandler
    from Orange.widgets.utils.widgetpreview import WidgetPreview


    class Item(FeatureItem):
        def set_data(self, value):
            self._x_data = value
            item = QGraphicsRectItem()
            item.setX(0)
            item.setY(self._height / 2 - self._height * 0.5 / 2)
            width = value * self._width / (self._range[1] - self._range[0])
            item.setRect(0, 0, width, self._height * 0.5)
            self._group.addToGroup(item)

        def rescale(self, width):
            self._width = width
            for item in self._group.childItems():
                rect: QRectF = item.rect()
                rect.setX(self._width / 2)
                rect.setWidth(self._x_data * self._width /
                              (self._range[1] - self._range[0]))
                item.setRect(rect)
            self.updateGeometry()

        def set_height(self, height):
            self._height = height + self.HEIGHT
            for item in self._group.childItems():
                rect: QRectF = item.rect()
                rect.setHeight(self._height / 2)
                item.setRect(rect)
            self.updateGeometry()


    class Plot(FeaturesPlot):
        def _set_range(self, x: np.ndarray):
            abs_max = np.max(np.abs(x)) * 1.05
            self._range = -abs_max, abs_max

        def _set_items(self, x, labels):
            for i in range(x.shape[0]):
                item = Item(self, labels[i], self._range,
                            self.item_column_width)
                item.set_data(x[i])
                self._items.append(item)
                self._layout.addItem(item, i, FeaturesPlot.ITEM_COLUMN)
                if i == MAX_N_ITEMS:
                    break

        def select_from_settings(self, *_):
            pass


    class Widget(OWExplainFeatureBase):
        name = "Explain"
        PLOT_CLASS = Plot
        settingsHandler = DomainContextHandler()

        def update_selection(self, *_):
            pass

        def get_selected_data(self):
            return None

        def get_scores_table(self):
            return None

        def update_scene(self):
            super().update_scene()
            if self.results is not None:
                values = np.mean(self.results.x, axis=0)
                indices = np.argsort(values)[::-1]
                names = [self.results.names[i] for i in indices]
                self.setup_plot(values[indices], names)

        @staticmethod
        def run(data, model, state):
            if not data or not model:
                return None

            def callback(i: float, status=""):
                state.set_progress_value(i * 100)
                if status:
                    state.set_status(status)
                if state.is_interruption_requested():
                    raise Exception

            data = model.data_to_model_domain(data, callback)
            names = [attr.name for attr in data.domain.attributes]
            mask = np.ones(len(data), dtype=bool)
            return BaseResults(x=data.X, names=names, mask=mask)


    table = Table("iris")
    rf_model = RandomForestLearner(random_state=0)(table)
    WidgetPreview(Widget).run(set_data=table, set_model=rf_model)
