

import numpy as np
import sys

from PySide6 import QtCore
import sys
from typing import Callable
from PySide6.QtWidgets import (QApplication, QWidget, QLineEdit, QSlider, QGridLayout, QLabel, QVBoxLayout)
import threading

def round_to_multiple(value, multiple=1):
    """ Rounds value to nearest multiple. Multiple can be greater or less than 1.
        Example:
            round_to_float(7.77777, 1e-3) --> 7.778
            round_to_float(7.77777, 3)    --> 9.0    
    """

    invmul = 1/multiple

    r1 = value/multiple
    w = r1//1
    w = np.where((r1%1) >= 0.5, w+1, w)
    
    mlog10 = np.log10(multiple)
    
    if mlog10 > 0:
        return w/invmul
    else:
        return np.around(w/invmul, int(np.abs(mlog10)))


class TunerThread(QtCore.QThread):

    def __init__(self, plot: Callable):

        self._wait = QtCore.QWaitCondition()
        self.mutex = QtCore.QMutex()
        self.queue = []
        self.plot = plot
        super().__init__()

    @QtCore.Slot(str)
    def push(self):

        self._wait.wakeAll()

    def run(self):

        while 1:
            self.mutex.lock()
            self._wait.wait(self.mutex)
            self.mutex.unlock()

            self.plot()

    
class DoubleSlider(QSlider):

    def __init__(self, step_size, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._min = 0
        self._max = 1
        self._step_size = 0.1
        self.setSingleStep(step_size)
        super().setSingleStep(1)

    def value(self):
        value = super().value()* self._step_size
        return round_to_multiple(value, self._step_size)

    def setMinimum(self, value):
        rmin = round_to_multiple(value, self._step_size)
        self._min = int(rmin / self._step_size)
        return super().setMinimum(self._min)

    def setMaximum(self, value):
        rmax = round_to_multiple(value, self._step_size)
        self._max = int(rmax / self._step_size)
        return super().setMaximum(self._max)

    def setSingleStep(self, value):
        self._step_size = value

    def singleStep(self):
        value = super().singleStep() * self._step_size
        return round_to_multiple(value, self._step_size)

    def setValue(self, value):
        rval = round_to_multiple(value, self._step_size)
        super().setValue(int(rval / self._step_size))

    def setBounds(self, min_, max_, step):
        val = self.value()
        self.setSingleStep(step)
        self.setMinimum(min_)
        self.setMaximum(max_)
        self.setValue(round_to_multiple(val, step))


class FloatTuner(QWidget):
    def __init__(
        self, 
        label: str, 
        lower: float, 
        upper: float, 
        initial: float, 
        multiplier: float, 
        callback: Callable, 
        processor: QtCore.QThread
    ):
        super(FloatTuner, self).__init__()
        name = QLabel(label)

        self.multiplier = multiplier

        self.processor = processor
        
        self.lower_limit = QLineEdit(str(lower))
        self.lower_limit.setFixedWidth(50)

        self.upper_limit = QLineEdit(str(upper))
        self.upper_limit.setFixedWidth(50)
        
        self.value = QLineEdit(str(initial))
        self.value.setFixedWidth(50)

        self.step = (upper - lower) / 100
        self.slider = DoubleSlider(self.step, QtCore.Qt.Horizontal)
        self.slider.setValue(initial)
        self.slider.setBounds(lower, upper, self.step)
        
        self.setFixedWidth(500)

        mainLayout = QGridLayout()
        mainLayout.addWidget(name,             0, 0)
        mainLayout.addWidget(self.value,       0, 1)
        mainLayout.addWidget(self.slider,      0, 2)
        mainLayout.addWidget(self.lower_limit, 0, 3)
        mainLayout.addWidget(self.upper_limit, 0, 4)

        mainLayout.setColumnStretch(2, 1)
        mainLayout.setColumnMinimumWidth(1, 20)
        mainLayout.setColumnMinimumWidth(2, 20)
        mainLayout.setColumnMinimumWidth(3, 20)
        mainLayout.setColumnMinimumWidth(4, 20)

        self.setLayout(mainLayout)

        self.slider.setFocus()   
        self.value.setText(str(self.slider.value()))

        self.callback = callback
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.value.returnPressed.connect(self.value_box_changed)
        self.lower_limit.returnPressed.connect(self.set_scale)
        self.upper_limit.returnPressed.connect(self.set_scale)
    
    def value_box_changed(self):
        value = float(self.value.text())
        self.slider.setValue(value)
        value = self.slider.value()
        self.value.setText(str(value))

    def slider_value_changed(self, value):
        value = self.slider.value()
        prev_value = float(self.value.text())

        if np.abs(value - prev_value) > (self.step / 2):
            self.value.setText(str(value))
            self.callback(value * self.multiplier)
            self.processor.push()

    def set_scale(self):
        min_ = float(self.lower_limit.text())
        max_ = float(self.upper_limit.text())
        self.step = (max_ - min_) / 100
        self.slider.setBounds(min_, max_, self.step)

class TunerGroup(QWidget):
    def __init__(self, tuners, plot):
        super(TunerGroup,self).__init__()

        self.processor = TunerThread(plot)
        self.processor.start()

        mainLayout = QVBoxLayout()
        self.tuners = []
        for i, (k, v) in enumerate(tuners.items()):
            self.tuners.append(FloatTuner(**v, processor=self.processor))
            mainLayout.addWidget(self.tuners[i], i)

        mainLayout.setSpacing(1)
        self.setLayout(mainLayout)
