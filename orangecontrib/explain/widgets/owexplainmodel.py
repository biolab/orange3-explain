# pylint: disable=missing-docstring,no-name-in-module,invalid-name
# pylint: disable=too-few-public-methods
from typing import Tuple, Optional, List, Callable

import numpy as np

from AnyQt.QtCore import Qt, QRectF, QSizeF, pyqtSignal as Signal
from AnyQt.QtGui import QColor, QPen, QBrush, QLinearGradient, QFont
from AnyQt.QtWidgets import QGraphicsItemGroup, QGraphicsWidget, \
    QGraphicsEllipseItem, QGraphicsSimpleTextItem, QGraphicsRectItem, \
    QGraphicsSceneMouseEvent, QComboBox

from Orange.base import Model
from Orange.classification import RandomForestLearner
from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.regression import RandomForestRegressionLearner
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, \
    ClassValuesContextHandler
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.customizableplot import Updater

from orangecontrib.explain.explainer import \
    get_shap_values_and_colors, RGB_LOW, RGB_HIGH, temp_seed
from orangecontrib.explain.widgets.owexplainfeaturebase import \
    OWExplainFeatureBase, BaseResults, FeaturesPlot, BaseParameterSetter, \
    FeatureItem, SelectionRect as BaseSelectionRect, MAX_N_ITEMS


class Results(BaseResults):
    colors: Optional[List[np.ndarray]] = None


class Legend(QGraphicsWidget):
    BAR_WIDTH = 7
    BAR_HEIGHT = 150
    FONT_SIZE = 11

    def __init__(self, parent):
        super().__init__(parent)
        self.__offset = 2
        self.__group = QGraphicsItemGroup(self)
        self._add_bar()
        self._add_high_label()
        self._add_low_label()
        self._add_feature_label()
        font = self.font()
        font.setPointSize(self.FONT_SIZE)
        self.set_font(font)

    def _add_bar(self):
        self._bar_item = QGraphicsRectItem()
        self._bar_item.setPen(QPen(Qt.NoPen))
        self._set_bar(self.BAR_HEIGHT)
        self.__group.addToGroup(self._bar_item)

    def _add_high_label(self):
        self.__high_label = item = QGraphicsSimpleTextItem("High")
        item.setX(self.BAR_WIDTH + self.__offset)
        item.setY(0)
        self.__group.addToGroup(item)

    def _add_low_label(self):
        self.__low_label = item = QGraphicsSimpleTextItem("Low")
        item.setX(self.BAR_WIDTH + self.__offset)
        self.__group.addToGroup(item)

    def _add_feature_label(self):
        self.__feature_label = item = QGraphicsSimpleTextItem("Feature value")
        item.setRotation(-90)
        item.setX(self.BAR_WIDTH + self.__offset * 2)
        self.__group.addToGroup(item)

    def _set_font(self):
        for label in (self.__high_label,
                      self.__low_label,
                      self.__feature_label):
            label.setFont(self.font())

        bar_height = self._bar_item.boundingRect().height()

        height = self.__low_label.boundingRect().height()
        self.__low_label.setY(bar_height - height)

        width = self.__feature_label.boundingRect().width()
        self.__feature_label.setY(bar_height / 2 + width / 2)

    def set_font(self, font: QFont):
        self.setFont(font)
        self._set_font()

    def _set_bar(self, height: int):
        self._bar_item.setRect(0, 0, self.BAR_WIDTH, height)
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(*RGB_HIGH))
        gradient.setColorAt(1, QColor(*RGB_LOW))
        self._bar_item.setBrush(gradient)

    def set_bar(self, height: int):
        self._set_bar(height)
        self._set_font()

    def sizeHint(self, *_):
        width = self.__high_label.boundingRect().width()
        width += self._bar_item.boundingRect().width() + self.__offset
        return QSizeF(width, ViolinItem.HEIGHT)


class ViolinItem(FeatureItem):
    POINT_R = 6
    SCALE_FACTOR = 0.5
    selection_changed = Signal(float, float, str)

    class SelectionRect(BaseSelectionRect):
        def __init__(self, parent, width: int):
            super().__init__(parent)
            self.parent_width = width

    def __init__(self, parent, attr_name: str, x_range: Tuple[float],
                 width: int):
        super().__init__(parent, attr_name, x_range, width)
        assert x_range[0] == -x_range[1]
        self._range = x_range[1] if x_range[1] else 1
        self._selection_rect: Optional[QGraphicsRectItem] = None
        parent.selection_cleared.connect(self.__remove_selection_rect)

    def set_data(self, x_data: np.ndarray, color_data: np.ndarray):
        def put_point(_x, _y):
            item = QGraphicsEllipseItem()
            item.setX(_x)
            item.setY(_y)
            item.setRect(0, 0, self.POINT_R, self.POINT_R)
            color = QColor(*colors.pop().astype(int))
            item.setPen(QPen(color))
            item.setBrush(QBrush(color))
            self._group.addToGroup(item)

        self._x_data = x_data

        x_data_unique, dist, x_data = self.prepare_data()
        for x, d in zip(x_data_unique, dist):
            colors = color_data[x_data == np.round(x, 3)]
            colors = list(colors[np.random.choice(len(colors), 11)])
            y = self._height / 2 - self.POINT_R / 2
            self.plot_data(put_point, x, y, d)

    def prepare_data(self) -> \
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x_data = self._values_to_pixels(self._x_data)
        x_data = x_data[~np.isnan(x_data)]
        if len(x_data) == 0:
            return None

        x_data = np.round(x_data - self.POINT_R / 2, 3)

        # remove duplicates and get counts (distribution) to set y
        x_data_unique, counts = np.unique(x_data, return_counts=True)
        min_count, max_count = np.min(counts), np.max(counts)
        dist = (counts - min_count) / (max_count - min_count)
        if min_count == max_count:
            dist[:] = 1
        dist = dist ** 0.7

        # plot rarest values first
        indices = np.argsort(counts)
        return x_data_unique[indices], dist[indices], x_data

    @staticmethod
    def plot_data(func: Callable, x: float, y: float, d: float):
        func(x, y)  # y = 0
        if d > 0:
            offset = d * 10
            func(x, y + offset)  # y = (0, 10]
            func(x, y - offset)  # y = (-10, 0]
            for i in range(2, int(offset), 2):
                func(x, y + i)  # y = [2, 8]
                func(x, y - i)  # y = [-8, -2]

    def _values_to_pixels(self, x: np.ndarray) -> np.ndarray:
        # scale data to [-0.5, 0.5]
        x = x / self._range * self.SCALE_FACTOR
        # round data to 3. decimal for sampling
        x = np.round(x, 3)
        # convert to pixels
        return x * self._width + self._width / 2

    def _values_from_pixels(self, p: np.ndarray) -> np.ndarray:
        # convert from pixels
        x = (p - self._width / 2) / self._width
        # rescale data from [-0.5, 0.5]
        return np.round(x * self._range / self.SCALE_FACTOR, 3)

    def rescale(self, width):
        def move_point(_x, *_):
            item = next(points)
            item.setX(_x)

        self._width = width
        self.updateGeometry()
        points = (item for item in self._group.childItems())

        x_data_unique, dist, _ = self.prepare_data()
        for x, d in zip(x_data_unique, dist):
            self.plot_data(move_point, x, 0, d)

        if self._selection_rect is not None:
            old_width = self._selection_rect.parent_width
            rect = self._selection_rect.rect()
            x1 = self._width * rect.x() / old_width
            x2 = self._width * (rect.x() + rect.width()) / old_width
            rect = QRectF(x1, rect.y(), x2 - x1, rect.height())
            self._selection_rect.setRect(rect)
            self._selection_rect.parent_width = self._width

    def set_height(self, height: float):
        self._height = height + self.HEIGHT
        # TODO

    def __remove_selection_rect(self):
        if self._selection_rect is not None:
            self._selection_rect.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(self._selection_rect)
            self._selection_rect = None

    def add_selection_rect(self, x1, x2):
        x1, x2 = self._values_to_pixels(np.array([x1, x2]))
        rect = QRectF(x1, 0, x2 - x1, self._height)
        self._selection_rect = ViolinItem.SelectionRect(self, self._width)
        self._selection_rect.setRect(rect)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() & Qt.LeftButton:
            if self._selection_rect is None:
                self._selection_rect = ViolinItem.SelectionRect(
                    self, self._width)
            x = event.buttonDownPos(Qt.LeftButton).x()
            rect = QRectF(x, 0, event.pos().x() - x, self._height).normalized()
            rect = rect.intersected(self.contentsRect())
            self._selection_rect.setRect(rect)
            event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        x1 = event.buttonDownPos(Qt.LeftButton).x()
        x2 = event.pos().x()
        if x1 > x2:
            x2, x1 = x1, x2
        x1, x2 = self._values_from_pixels(np.array([x1, x2]))
        self.selection_changed.emit(x1, x2, self._attr_name)
        event.accept()


class ParameterSetter(BaseParameterSetter):
    LEGEND_HEIGHT = "Legend height"

    def update_setters(self):
        super().update_setters()

        def update_legend(**settings):
            font = Updater.change_font(self.legend.font(), settings)
            self.legend.set_font(font)

        def update_legend_bar(**settings):
            self.legend.set_bar(settings[self.LEGEND_HEIGHT])

        self.initial_settings[self.LABELS_BOX][self.LEGEND_LABEL] = \
            self.FONT_SETTING
        self.initial_settings[self.PLOT_BOX][self.LEGEND_HEIGHT] = {
            self.LEGEND_HEIGHT: (range(100, 500, 5), Legend.BAR_HEIGHT)
        }

        self._setters[self.LABELS_BOX][self.LEGEND_LABEL] = update_legend
        self._setters[self.PLOT_BOX][self.LEGEND_HEIGHT] = update_legend_bar

    @property
    def legend(self) -> Legend:
        return self.master.legend


class ViolinPlot(FeaturesPlot):
    BOTTOM_AXIS_LABEL = "Impact on model output"
    LEGEND_COLUMN = 2
    selection_changed = Signal(float, float, str)

    def __init__(self):
        super().__init__()
        self.__legend = Legend(self)
        self.layout().addItem(self.__legend, 0, ViolinPlot.LEGEND_COLUMN)

        self.parameter_setter = ParameterSetter(self)

    @FeaturesPlot.item_column_width.setter
    def item_column_width(self, view_width: int):
        j = FeaturesPlot.LABEL_COLUMN
        w = max([self._layout.itemAt(i, j).item.boundingRect().width()
                 for i in range(len(self._items))] + [0])
        width = view_width - self.legend.sizeHint().width() - self.OFFSET - w
        self._item_column_width = max(self.ITEM_COLUMN_WIDTH, width)

    @property
    def legend(self):
        return self.__legend

    def show_legend(self, show: bool):
        self.__legend.setVisible(show)
        self._bottom_axis.setWidth(self.item_column_width)
        x = self._item_column_width / 2
        self._vertical_line.setLine(x, 0, x, self._vertical_line.line().y2())

    def select_from_settings(self, x1: float, x2: float, attr_name: str):
        point_r_diff = 2 * self._range[1] / (self.item_column_width / 2)
        for item in self._items:
            if item.attr_name == attr_name:
                item.add_selection_rect(x1 - point_r_diff, x2 + point_r_diff)
                break
        self.select(x1, x2, attr_name)

    def _set_range(self, x: np.ndarray, *_):
        abs_max = np.max(np.abs(x)) * 1.05
        self._range = -abs_max, abs_max

    def _set_items(self, x: np.ndarray, labels: List[str], colors: np.ndarray):
        with temp_seed(0):
            for i in range(x.shape[1]):
                item = ViolinItem(self, labels[i], self._range,
                                  self.item_column_width)
                item.set_data(x[:, i], colors[:, i])
                item.selection_changed.connect(self.select)
                self._items.append(item)
                self._layout.addItem(item, i, FeaturesPlot.ITEM_COLUMN)
                if i == MAX_N_ITEMS - 1:
                    break


class OWExplainModel(OWExplainFeatureBase):
    name = "Explain Model"
    description = "Model explanation widget."
    keywords = ["explain", "explain prediction", "explain model"]
    icon = "icons/ExplainModel.svg"
    priority = 100
    replaces = [
        "orangecontrib.prototypes.widgets.owexplainmodel.OWExplainModel"
    ]

    settingsHandler = ClassValuesContextHandler()
    target_index = ContextSetting(0)
    show_legend = Setting(True)

    PLOT_CLASS = ViolinPlot

    def __init__(self):
        self._target_combo: QComboBox = None
        super().__init__()

    # GUI setup
    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Target class")
        self._target_combo = gui.comboBox(
            box, self, "target_index",
            callback=self.__target_combo_changed,
            contentsLength=12)

        super()._add_controls()

        # TODO - implement zoom - ViolinItem.set_height()
        self.controls.zoom_level.parent().setVisible(False)

        gui.checkBox(self.display_box, self, "show_legend", "Show legend",
                     callback=self.__show_check_changed)

    def __target_combo_changed(self):
        self.update_scene()
        self.update_scores()
        self._clear_selection()

    def __show_check_changed(self):
        if self.plot is not None:
            self.plot.show_legend(self.show_legend)

    def openContext(self, model: Optional[Model]):
        super().openContext(model.domain.class_var if model else None)

    def setup_controls(self):
        self._target_combo.clear()
        self._target_combo.setEnabled(True)
        if self.model is not None:
            if self.model.domain.has_discrete_class:
                self._target_combo.addItems(self.model.domain.class_var.values)
                self.target_index = 0
            elif self.model.domain.has_continuous_class:
                self.target_index = -1
                self._target_combo.setEnabled(False)
            else:
                raise NotImplementedError

    # Plot setup
    def update_scene(self):
        super().update_scene()
        if self.results is not None:
            assert isinstance(self.results.x, list)
            x = self.results.x[self.target_index]
            scores_x = np.mean(np.abs(x), axis=0)
            indices = np.argsort(scores_x)[::-1]
            colors = self.results.colors
            names = [self.results.names[i] for i in indices]
            if x.shape[1]:
                self.setup_plot(x[:, indices], names, colors[:, indices])

    def setup_plot(self, values, names, *plot_args):
        super().setup_plot(values, names, *plot_args)
        self.plot.show_legend(self.show_legend)

    # Selection
    def update_selection(self, min_val: float, max_val: float, attr_name: str):
        assert self.results is not None
        x = self.results.x[self.target_index]
        column = self.results.names.index(attr_name)
        mask = self.results.mask.copy()
        mask[self.results.mask] = np.logical_and(x[:, column] <= max_val,
                                                 x[:, column] >= min_val)
        if not self.selection and not any(mask):
            return
        self.selection = (attr_name, list(np.flatnonzero(mask)))
        self.commit()

    def select_pending(self, pending_selection: Tuple):
        if not pending_selection or not pending_selection[1] \
                or self.results is None:
            return

        attr_name, row_indices = pending_selection
        names = self.results.names
        if not names or attr_name not in names:
            return
        col_index = names.index(attr_name)
        mask = np.zeros(self.results.mask.shape, dtype=bool)
        mask[row_indices] = True
        mask = np.logical_and(self.results.mask, mask)
        row_indices = np.flatnonzero(mask[self.results.mask])
        column = self.results.x[self.target_index][row_indices, col_index]
        x1, x2 = np.min(column), np.max(column)
        self.plot.select_from_settings(x1, x2, attr_name)
        super().select_pending(())

    # Outputs
    def get_selected_data(self):
        return self.data[self.selection[1]] \
            if self.selection and self.selection[1] else None

    def get_scores_table(self) -> Table:
        x = self.results.x[self.target_index]
        scores = np.mean(np.abs(x), axis=0)
        domain = Domain([ContinuousVariable("Score")],
                        metas=[StringVariable("Feature")])
        scores_table = Table(domain, scores[:, None],
                             metas=np.array(self.results.names)[:, None])
        scores_table.name = "Feature Scores"
        return scores_table

    # Misc
    def send_report(self):
        if not self.data or not self.model:
            return
        items = {"Target class": "None"}
        if self.model.domain.has_discrete_class:
            class_var = self.model.domain.class_var
            items["Target class"] = class_var.values[self.target_index]
        self.report_items(items)
        super().send_report()

    @staticmethod
    def run(data: Table, model: Model, state: TaskState) -> Results:
        if not data or not model:
            return None

        def callback(i: float, status=""):
            state.set_progress_value(i * 100)
            if status:
                state.set_status(status)
            if state.is_interruption_requested():
                raise Exception

        x, names, mask, colors = get_shap_values_and_colors(model, data,
                                                            callback)
        return Results(x=x, colors=colors, names=names, mask=mask)


if __name__ == "__main__":  # pragma: no cover
    table = Table("heart_disease")
    if table.domain.has_continuous_class:
        rf_model = RandomForestRegressionLearner(random_state=42)(table)
    else:
        rf_model = RandomForestLearner(random_state=42)(table)
    WidgetPreview(OWExplainModel).run(set_data=table, set_model=rf_model)
