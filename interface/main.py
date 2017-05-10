#!/usr/bin/env python3

import os
import kivy
import numpy as np
import threading
from kivy.app import App
from kivy.loader import Loader
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image, AsyncImage
from kivy.graphics import Color, Rectangle
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.layout import Layout
from kivy.uix.button import Button
from kivy.properties import ListProperty
from kivy.uix.treeview import TreeView, TreeViewNode, TreeViewLabel

from functools import partial
from coactive import Domain, CoactiveModel
from draw import draw_tables, draw_rooms
from gridui import GridUI, Object

kivy.resources.resource_add_path('/usr/share/fonts')


class Problem:

    _sliders = {
        'tables': [
            {'label': 'Min X distance', 'index': 1, 'min': 0.0, 'max': 10.0},
            {'label': 'Max X distance', 'index': 0, 'min': 0.0, 'max': 10.0},
            {'label': 'Min Y distance', 'index': 3, 'min': 0.0, 'max': 10.0},
            {'label': 'Max Y distance', 'index': 2, 'min': 0.0, 'max': 10.0},
            {'label': 'Min distance from walls', 'index': 5, 'min': 0.0, 'max': 10.0},
            {'label': 'Max distance from walls', 'index': 4, 'min': 0.0, 'max': 10.0},
            {'label': 'Min distance from borders', 'index': 7, 'min': 0.0, 'max': 10.0},
            {'label': 'Max distance from borders', 'index': 6, 'min': 0.0, 'max': 10.0},
            {'label': 'Number of square tables', 'index': 8, 'min': 0.0, 'max': 10.0},
            {'label': 'Number of desks', 'index': 9, 'min': 0.0, 'max': 10.0},
        ],
        'rooms': []}

    _features = {'tables': 10, 'rooms': 45}
    _context = {
        'tables': {
            'SIDE' : 10,
            'N_TABLES' : 5,
            'N_WALLS' : 1,
            'door_x' : [1,10],
            'door_y' : [5,5],
            'wall_x': [1],
            'wall_y' : [1],
            'wall_dx' : [3],
            'wall_dy' : [3]},
        'rooms': {}
    }
    _draw = {'tables': draw_tables, 'rooms': draw_rooms}

    def __init__(self, domain):
        phi = os.path.join(domain, 'phi.mzn')
        infer = os.path.join(domain, 'infer.mzn')
        improve = os.path.join(domain, 'improve.mzn')
        self.domain = Domain(phi, infer, improve, self._features[domain])
        self.context = self._context[domain]
        self.draw = self._draw[domain]
        self.sliders = self._sliders[domain]


class LayoutImage(Image):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(1., 1., 1.)
            self.bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.track, size=self.track)

    def track(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size


class TreeViewSlider(Slider, TreeViewNode):

    def __init__(self, idx, text, **kwargs):
        super().__init__(**kwargs)
        self.index = idx
        self.text = text


class FeatureTree(TreeView):

    def __init__(self, problem, **kwargs):
        super().__init__(hide_root=True, **kwargs)

        self.problem = problem
        self.labels = {}
        self.sliders = {}
        for feat in self.problem.sliders:
            label = TreeViewLabel(text = '{} = {}'.format(feat['label'], 0),
                                  font_size='18pt', color=[0, 0, 0, 1],
                                  #font_name='cantarell/Cantarell-Regular.otf',
                                  no_selection=True)
            self.add_node(label)
            slider = TreeViewSlider(feat['index'], feat['label'],
                                    min=feat['min'], max=feat['max'], step=1.0)
            self.add_node(slider, label)
            slider.bind(value=self.on_slider_update)
            self.labels[feat['index']] = label
            self.sliders[feat['index']] = slider

    def set_sliders(self, phi):
        for idx, value in enumerate(phi):
            self.sliders[idx].value = int(value)

    def on_slider_update(self, slider, value):
        label = self.labels[slider.index]
        label.text = '{} = {}'.format(slider.text, int(value))
        self.changed = slider.index

    def on_node_expand(self, node):
        for n in self.iterate_open_nodes():
            if n is not node and n.is_open:
                self.toggle_node(n)

    def bind_sliders(self, f):
        for slider in self.sliders.values():
            slider.bind(value=f)


class MainWindow(GridLayout):


    def __init__(self, domain, **kwargs):
        super().__init__(cols=2, **kwargs)
        self.problem = Problem(domain)
        self.domain = self.problem.domain
        self.model = CoactiveModel(self.domain)

        self._x = self.problem.context
        self._y = self.model.infer(self._x)
        self.phi = list(self.model.phi(self._x, self._y))

        n_cells = (self._x['SIDE'],)*2
        # TODO: move this somewhere else
        palette = {'Table 2x1' : lambda pos : Object(pos, (2,1), GridUI.TEX_TABLE_2_1),
                   'Table 1x2' : lambda pos : Object(pos, (1,2), GridUI.TEX_TABLE_1_2),
                   'Table 1x1' : lambda pos : Object(pos, (1,1), GridUI.TEX_TABLE_1_1)}

        self.grid_ui = GridUI(n_cells, palette)
        self.add_widget(self.grid_ui)
        self.grid_ui.update(self._x, self._y)

        self.side = GridLayout(cols=1, rows=2, size_hint=(.45, 1))
        with self.side.canvas.before:
            Color(.9, .9, .9)
            self.side_bg = Rectangle()
            self.side.bind(pos=self.track, size=self.track)
        self.add_widget(self.side)

        self.tree = FeatureTree(self.problem, size_hint=(.45, .8))
        self.side.add_widget(self.tree)
        self.tree.bind_sliders(self.on_slider_update)
        self.tree.set_sliders(self.phi)

        self.submit = Button(text='Submit', size_hint=(.2, .1))
        self.side.add_widget(self.submit)
        self.submit.bind(on_press=self.on_submit_press)

    def on_slider_update(self, slider, value):
        _phi = list(self.phi)
        _phi[slider.index] = slider.value

        y_bar = self.model.improve(self._x, np.array(_phi, dtype=np.int32),
                                   self.tree.changed)

        self.phi = self.model.phi(self._x, y_bar)
        self.grid_ui.update(self._x, y_bar)

    def on_submit_press(self, *args):
        y_bar = self.grid_ui.get_output()
        n_tables = len(y_bar['x'])
        if n_tables != self._x['N_TABLES']:
            phi = self.model.phi(self._x, self._y)
            x = dict(self._x)
            x['N_TABLES'] = n_tables
            phi_bar = self.model.phi(x, y_bar)
            self.model.phi_update(phi, phi_bar)
            self._x['N_TABLES'] = n_tables
        else:
            self.model.update(self._x, self._y, y_bar)
        self._y = self.model.infer(self._x)
        self.phi = self.model.phi(self._x, self._y)
        self.tree.set_sliders(self.phi)
        self.grid_ui.update(self._x, self._y)

    def track(self, *args):
        self.side_bg.pos = self.side.pos
        self.side_bg.size = self.side.size


class LayoutSynthesisApp(App):
    def build(self):
        return MainWindow('tables')


if __name__ == '__main__':
    LayoutSynthesisApp().run()

