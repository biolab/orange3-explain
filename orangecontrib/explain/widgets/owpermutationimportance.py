# pylint: disable=missing-docstring,no-name-in-module,invalid-name
# pylint: disable=too-few-public-methods
from typing import Tuple, Optional, List, Type, Set

import numpy as np

from AnyQt.QtCore import QRectF, Qt, QPointF
from AnyQt.QtGui import QColor, QPen, QBrush
from AnyQt.QtWidgets import QGraphicsRectItem, QGraphicsSceneMouseEvent, \
    QApplication, QComboBox

from Orange.base import Model
from Orange.classification import RandomForestLearner
from Orange.data import Table, DiscreteVariable, Domain, ContinuousVariable, \
    StringVariable, HasClass
from Orange.evaluation.scoring import Score
from Orange.regression import RandomForestRegressionLearner
from Orange.version import version
from Orange.widgets import gui
from Orange.widgets.evaluate.utils import BUILTIN_SCORERS_ORDER, usable_scorers
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler
from Orange.widgets.utils.concurrent import TaskState
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg

from orangecontrib.explain.inspection import permutation_feature_importance
from orangecontrib.explain.widgets.owexplainfeaturebase import \
    OWExplainFeatureBase, BaseResults as Results, FeaturesPlot, \
    FeatureItem, SelectionRect, MAX_N_ITEMS


class FeatureImportanceItem(FeatureItem):
    MEAN_RATIO = 0.6
    STD_RATIO = 0.1

    def __init__(self, parent, attr_name: str, x_range: Tuple[float],
                 width: int):
        super().__init__(parent, attr_name, x_range, width)
        self._std: Optional[float] = None
        self._pen = QPen(Qt.NoPen)
        self._pen_selected = QPen(QColor("#ffbe00"), 4)

    def set_data(self, x_data: float, std: float):
        self._x_data = x_data
        self._std = std

        item = QGraphicsRectItem(0, 0, 0, self._height * self.MEAN_RATIO)
        item.setPen(self._pen)
        item.setBrush(QBrush(QColor("#1f77b4")))
        self._group.addToGroup(item)

        item = QGraphicsRectItem(0, 0, 0, self._height * self.STD_RATIO)
        item.setPen(self._pen)
        item.setBrush(QBrush(QColor("#444")))
        self._group.addToGroup(item)

        self.set_height(0)

    def rescale(self, width: int):
        self._width = width

        min_max = self._range[1] - self._range[0]
        scale = self._width / min_max
        x0_px = - self._range[0] * scale
        x_px = self._x_data * scale
        std_px = self._std * scale

        child_items = self._group.childItems()

        item = child_items[0]
        rect: QRectF = item.rect()
        rect.setX(x0_px)
        rect.setWidth(x_px)
        item.setRect(rect)

        item = child_items[1]
        rect: QRectF = item.rect()
        rect.setX(x0_px + x_px - std_px)
        rect.setWidth(std_px * 2)
        item.setRect(rect)

        self.updateGeometry()

    def set_height(self, height: int):
        self._height = height + self.HEIGHT
        ratios = [self.MEAN_RATIO, self.STD_RATIO]

        for item, ratio in zip(self._group.childItems(), ratios):
            rect: QRectF = item.rect()
            item.setY(self._height / 2 - self._height * ratio / 2)
            rect.setHeight(self._height * ratio)
            item.setRect(rect)

        self.updateGeometry()

    def set_selected(self, selected: bool):
        mean_item = self._group.childItems()[0]
        mean_item.setPen(self._pen_selected if selected else self._pen)


class FeatureImportancePlot(FeaturesPlot):
    class SelectAction:
        ClearSelect, Select, Deselect, Toggle = range(4)

    def __init__(self):
        super().__init__()
        self.__sel_action: Optional[int] = None
        self.__selection_rect = SelectionRect(self)
        self.__selection_rect.setVisible(False)
        self.__selection_rect.setZValue(100)
        self.__selected_attrs = set()
        self.__selected_attrs_current = set()

    def _set_range(self, x: np.ndarray, std: np.ndarray, *_):
        self._range = (min(np.min(x - std) * 1.05, 0),
                       max(np.max(x + std) * 1.05, 0))

    def _set_items(self, x: np.ndarray, labels: List[str], std: np.ndarray,
                   x_label: str):
        for i in range(x.shape[0]):
            item = FeatureImportanceItem(self, labels[i], self._range,
                                         self.item_column_width)
            item.set_data(x[i], std[i])
            self._items.append(item)
            self._layout.addItem(item, i, FeaturesPlot.ITEM_COLUMN)
            if i == MAX_N_ITEMS - 1:
                break
        self._bottom_axis.setLabel(x_label)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() & Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self.__sel_action = self.SelectAction.Toggle
            elif event.modifiers() & Qt.AltModifier:
                self.__sel_action = self.SelectAction.Deselect
            elif event.modifiers() & Qt.ShiftModifier:
                self.__sel_action = self.SelectAction.Select
            else:
                self.__sel_action = self.SelectAction.ClearSelect
            self.select(event.pos())
            event.accept()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if event.buttons() & Qt.LeftButton:
            rect = QRectF(event.buttonDownPos(Qt.LeftButton),
                          event.pos()).normalized()
            self.__selection_rect.setRect(rect)
            self.__selection_rect.setVisible(True)
            if rect.width():
                self.select(None)
            event.accept()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.LeftButton:
            self.__selection_rect.setVisible(False)
            self.__sel_action = None
            self.__selected_attrs = self.__selected_attrs_current
            self.selection_changed.emit(self.__selected_attrs)
            event.accept()

    def select(self, point: Optional[QPointF]):
        def add_to_selection(item_):
            item_.set_selected(True)
            item_.update()
            self.__selected_attrs_current.add(item_.attr_name)

        def remove_from_selection(item_):
            item_.set_selected(False)
            item_.update()
            if item_.attr_name in self.__selected_attrs_current:
                self.__selected_attrs_current.remove(item_.attr_name)

        self.__selected_attrs_current = self.__selected_attrs.copy()

        for item in self._items:
            item: FeatureImportanceItem
            if point is not None:
                rect = item.sceneBoundingRect()
                collides = rect.adjusted(0, 1, 0, 0).contains(point)
            else:
                collides = item.collidesWithItem(self.__selection_rect)

            if self.__sel_action == self.SelectAction.ClearSelect:
                if collides:
                    add_to_selection(item)
                else:
                    remove_from_selection(item)

            elif self.__sel_action == self.SelectAction.Select:
                if collides:
                    add_to_selection(item)
                else:
                    if item.attr_name not in self.__selected_attrs_current:
                        remove_from_selection(item)

            elif self.__sel_action == self.SelectAction.Deselect:
                if collides:
                    remove_from_selection(item)
                else:
                    if item.attr_name in self.__selected_attrs_current:
                        add_to_selection(item)

            elif self.__sel_action == self.SelectAction.Toggle:
                if collides:
                    if item.attr_name in self.__selected_attrs_current:
                        remove_from_selection(item)
                    else:
                        add_to_selection(item)
                else:
                    if item.attr_name in self.__selected_attrs_current:
                        add_to_selection(item)
                    else:
                        remove_from_selection(item)

            else:
                raise NotImplementedError

    def deselect(self):
        keys = QApplication.keyboardModifiers()
        if keys & (Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier):
            return
        for item in self._items:
            item.set_selected(False)
        self.__selected_attrs.clear()
        super().deselect()

    def select_from_settings(self, attr_names: Tuple[str]):
        self.__selected_attrs = set()
        for item in self._items:
            item: FeatureImportanceItem
            if item.attr_name in attr_names:
                item.set_selected(True)
                self.__selected_attrs.add(item.attr_name)
        self.selection_changed.emit(self.__selected_attrs)


class OWPermutationImportance(OWExplainFeatureBase):
    name = "Feature Importance"
    description = "Inspect model using Permutation Feature " \
                  "Importance technique."
    keywords = ["explain", "model", "permutation", "feature", "importance"]
    icon = "icons/PermutationImportance.svg"
    priority = 50

    settingsHandler = PerfectDomainContextHandler()
    score_index = ContextSetting(0)
    n_repeats = Setting(5)

    PLOT_CLASS = FeatureImportancePlot

    class Warning(OWExplainFeatureBase.Warning):
        missing_target = Msg("Instances with unknown target values "
                             "were removed from data.")

    # GUI setup
    def _add_controls(self):
        box = gui.vBox(self.controlArea, "Parameters")
        self._score_combo: QComboBox = gui.comboBox(
            box, self, "score_index", label="Score:",
            items=BUILTIN_SCORERS_ORDER[DiscreteVariable],
            orientation=Qt.Horizontal, contentsLength=12,
            callback=self.__parameter_changed
        )
        gui.spin(
            box, self, "n_repeats", 1, 1000, label="Permutations:",
            controlWidth=50, callback=self.__parameter_changed
        )

        super()._add_controls()

    def __parameter_changed(self):
        self.clear()
        self.start(self.run, *self.get_runner_parameters())

    def _check_data(self):
        self.Warning.missing_target.clear()
        if self.data and np.isnan(self.data.Y).any():
            self.Warning.missing_target()
            self.data = HasClass()(self.data)

    def openContext(self, model: Optional[Model]):
        super().openContext(model.domain if model else None)

    def setup_controls(self):
        if self.model and self.model.domain.has_continuous_class:
            class_type = ContinuousVariable
        else:
            class_type = DiscreteVariable
        self._score_combo.clear()
        items = BUILTIN_SCORERS_ORDER[class_type]
        self._score_combo.addItems(items)
        self.score_index = items.index("R2") if "R2" in items else 0

    def get_runner_parameters(self) -> Tuple[Optional[Table], Optional[Model],
                                             Optional[Type[Score]], int]:
        score = None
        if self.model:
            if version > "3.31.1":
                # Eventually, keep this line (remove lines 305-306) and
                # upgrade minimal Orange version to 3.32.0.
                # Also remove the Orange.version import
                score = usable_scorers(self.model.domain)[self.score_index]
            else:
                var = self.model.domain.class_var
                score = usable_scorers(var)[self.score_index]
        return self.data, self.model, score, self.n_repeats

    # Plot setup
    def update_scene(self):
        super().update_scene()
        if self.results is not None:
            importance = self.results.x
            mean = np.mean(importance, axis=1)
            std = np.std(importance, axis=1)
            indices = np.argsort(mean)[::-1]
            names = [self.results.names[i] for i in indices]
            score = self._score_combo.itemText(self.score_index)
            txt = "Increase" if score in ("MSE", "RMSE", "MAE") else "Decrease"
            x_label = f"{txt} in {score}"
            self.setup_plot(mean[indices], names, std[indices], x_label)

    # Selection
    def update_selection(self, attr_names: Set[str]):
        if set(self.selection) == attr_names:
            return
        assert self.results is not None
        self.selection = tuple(attr_names)
        self.commit()

    def select_pending(self, pending_selection: Tuple):
        if not pending_selection or self.results is None:
            return

        self.plot.select_from_settings(pending_selection)
        super().select_pending(())

    # Outputs
    def get_selected_data(self) -> Optional[Domain]:
        if not self.selection or not self.data:
            return None
        domain = self.data.domain
        attrs = [a for a in domain.attributes if a.name in self.selection]
        return self.data[:, attrs + list(domain.class_vars + domain.metas)]

    def get_scores_table(self) -> Table:
        domain = Domain([ContinuousVariable("Mean"),
                         ContinuousVariable("Std")],
                        metas=[StringVariable("Feature")])
        x = self.results.x
        X = np.vstack((np.mean(x, axis=1), np.std(x, axis=1))).T
        M = np.array(self.results.names)[:, None]
        scores_table = Table(domain, X, metas=M)
        scores_table.name = "Feature Scores"
        return scores_table

    # Misc
    def send_report(self):
        if not self.data or not self.model or not self.data.domain.class_var:
            return
        var_type = type(self.data.domain.class_var)
        items = {
            "Score": BUILTIN_SCORERS_ORDER[var_type][self.score_index],
            "Permutations": self.n_repeats,
        }
        self.report_items(items)
        super().send_report()

    @staticmethod
    def run(data: Table, model: Model, score_class: Type[Score],
            n_repeats: int, state: TaskState) -> Optional[Results]:
        if not data or not model or not score_class:
            return None

        def callback(i: float, status=""):
            state.set_progress_value(i * 100)
            if status:
                state.set_status(status)
            if state.is_interruption_requested():
                raise Exception

        importance, names = permutation_feature_importance(
            model, data, score_class(), n_repeats, callback)
        mask = np.ones(importance.shape[0], dtype=bool)
        return Results(x=importance, names=names, mask=mask)


if __name__ == "__main__":  # pragma: no cover
    table = Table("heart_disease")
    if table.domain.has_continuous_class:
        rf_model = RandomForestRegressionLearner(random_state=0)(table)
    else:
        rf_model = RandomForestLearner(random_state=0)(table)
    WidgetPreview(OWPermutationImportance).run(set_data=table,
                                               set_model=rf_model)
