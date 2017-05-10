
from kivy.app import App
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Rectangle, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from random import random


class Object:

    def __init__(self, pos, size, texture, editable=True):
        self.pos = pos
        self.size = size
        self.texture = CoreImage(texture).texture
        self.editable = editable


class GridDrawingBoard(Widget):

    _BG_COLOR = lambda _ : Color(.75, .75, .75)
    _LINE_COLOR = lambda _ : Color(.5, .5, .5)

    def __init__(self, size, controller):
        super(GridDrawingBoard, self).__init__()
        if size[0] >= size[1]:
            self.size_hint = (size[1]/size[0], 1.0)
        else:
            self.size_hint = (1.0, size[0]/size[1])

        self.n_cols, self.n_rows  = size
        self.controller = controller
        self.bind(size=self.draw)
        self.bind(pos=self.draw)
        self.draw()

    def _get_relative_touch(self, touch):
        rel_x = (touch.x - self.pos[0]) / self.size[0]
        rel_y = (touch.y - self.pos[1]) / self.size[1]
        return rel_x, rel_y

    def _get_lower_left_node(self, touch):
        rx, ry = self._get_relative_touch(touch)
        cx = int(self.n_cols * rx)
        cy = int(self.n_rows * ry)
        return cx, cy

    def _get_nearest_node(self, touch):
        rx, ry = self._get_relative_touch(touch)
        cx = int(round(self.n_cols * rx))
        cy = int(round(self.n_rows * ry))
        return cx, cy

    def _draw_objects(self):
        delta_x = self.size[0] / self.n_cols
        delta_y = self.size[1] / self.n_rows

        for obj in self.controller.objects:
            obj_pos = (self.pos[0] + obj.pos[0] * delta_x,
                       self.pos[1] + obj.pos[1] * delta_y)
            size = (obj.size[0] * delta_x, obj.size[1] * delta_y)
            with self.canvas:
                Rectangle(pos=obj_pos, size=size, texture=obj.texture)

    def _draw_grid(self):
        delta_x = self.size[0] / self.n_cols
        delta_y = self.size[1] / self.n_rows
        with self.canvas:
            self._BG_COLOR()
            Rectangle(pos=self.pos, size=self.size)
            self._LINE_COLOR()
            # horizontal lines
            x1, x2 = self.pos[0], self.pos[0] + self.size[0]
            for i in range(1, self.n_rows):
                y = self.pos[1] + i * delta_y
                Line(points=(x1, y, x2, y))
            # vertical lines
            y1, y2 = self.pos[1], self.pos[1] + self.size[1]
            for i in range(1, self.n_cols):
                x = self.pos[0] + i * delta_x
                Line(points=(x, y1, x, y2))

            Color(1,1,1) # this seems necessary

    def draw(self, *args):
        self.canvas.clear()
        self._draw_grid()
        self._draw_objects()

    def on_touch_down(self, touch):
        if self._is_valid_touch(touch):
            self.controller.perform_action(self._get_lower_left_node(touch))
            self.draw()

    def _is_valid_touch(self, touch):
        return ((self.pos[0] <= touch.x) and
                (touch.x <= self.pos[0] + self.size[0]) and
                (self.pos[1] <= touch.y) and
                (touch.y <= self.pos[1] + self.size[1]))


class GridUI(BoxLayout):

    messages = {
    'default' : "Select an action by clicking on the buttons.",
    'add' : "Click on the grid to position the object (bottom-left corner).",
    'remove' : "Click on an object to remove it.",
    'move1' : "Click on the object to move.",
    'move2' : "Click on the new location (bottom-left corner).",
    'move_retry' : "The selected object cannot be moved there, try again.",
    'not_editable' : "The selected object is not editable."}

    TEX_WALL = "./resources/nice_wall.png"
    TEX_DOOR_LEFT = "./resources/nice_door_left.png"
    TEX_DOOR_RIGHT = "./resources/nice_door_right.png"
    TEX_DOOR_TOP = "./resources/nice_door_top.png"
    TEX_DOOR_BOTTOM = "./resources/nice_door_bottom.png"
    TEX_TABLE_2_1 = "./resources/nice_table_2_1.png"
    TEX_TABLE_1_1 = "./resources/nice_table_1_1.png"
    TEX_TABLE_1_2 = "./resources/nice_table_1_2.png"
    TEX_TABLE_GENERIC = TEX_TABLE_2_1

    WALL_BORDER = 0.5

    def __init__(self, size, object_palette):
        super(GridUI, self).__init__(orientation='vertical')
        self.n_cols, self.n_rows = size
        self.object_palette = object_palette
        self.objects = []
        self.output = None
        self.current_action = None

        self.grid = GridDrawingBoard((self.n_cols, self.n_rows), self)

        gl = FloatLayout()
        self.grid.pos_hint = {'center_x': .5, 'center_y': .5}
        self.grid.size_hint = (1.0 - (2/size[0]) * self.WALL_BORDER,
                               1.0 - (2/size[1]) * self.WALL_BORDER)
        gl.add_widget(self.grid)
        self.add_widget(gl)

        toolbar = BoxLayout(orientation='vertical', size_hint = (1, .1))
        self.info_label = Label(text=self.messages['default'])
        toolbar.add_widget(self.info_label)


        button_bar = BoxLayout(orientation='horizontal')
        add_btn = Button(text="Add ...")
        self.dropdown = DropDown()

        for obj_type in self.object_palette.keys():
            btn = Button(text=obj_type, size_hint_y=None, height=44)
            btn.bind(on_release=self._select_obj)
            self.dropdown.add_widget(btn)

        add_btn.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=
                      lambda inst, x: setattr(add_btn, 'text', "Add " + x))

        rem_btn = Button(text="Remove")
        rem_btn.bind(on_release=self._select_rem)

        move_btn = Button(text="Move")
        move_btn.bind(on_release=self._select_move)

        button_bar.add_widget(add_btn)
        button_bar.add_widget(rem_btn)
        button_bar.add_widget(move_btn)

        toolbar.add_widget(button_bar)
        self.add_widget(toolbar)

    def get_output(self):
        output = {key : [] for key in ['x','y','dx','dy']}
        print("========================================")
        for obj in self.objects:
            if obj.editable:
                x, y = obj.pos
                dx, dy = obj.size
                print("output -- ({},{}) ({},{})".format(x + 1,y + 1,dx, dy))
                output['x'].append(x + 1)
                output['y'].append(y + 1)
                output['dx'].append(dx)
                output['dy'].append(dy)

        return output

    def update(self, x, y):
        self.objects = []
        tables = list(zip(y['x'], y['dx'], y['y'], y['dy']))
        walls = list(zip(x['wall_x'], x['wall_dx'], x['wall_y'], x['wall_dy']))
        doors = list(zip(x['door_x'], x['door_y']))
        print("========================================")
        for x, dx, y, dy in tables:
            size = (dx, dy)
            if size == (2, 1):
                texture = self.TEX_TABLE_2_1
            elif size == (1, 2):
                texture = self.TEX_TABLE_1_2
            elif size == (1, 1):
                texture = self.TEX_TABLE_1_1
            else:
                texture = self.TEX_TABLE_GENERIC

            table = Object((x - 1, y - 1), size, texture)
            print("update -- {} {}".format((x,y),(dx,dy)))
            self.objects.append(table)

        for x, dx, y, dy in walls:
            wall = Object((x - 1, y - 1), (dx, dy), self.TEX_WALL, editable=False)
            self.objects.append(wall)

        for x, y in doors:
            if x == 1:
                texture = self.TEX_DOOR_LEFT
            elif x == self.n_cols:
                texture = self.TEX_DOOR_RIGHT
            elif y == 1:
                texture = self.TEX_DOOR_BOTTOM
            elif x == self.n_rows:
                texture = self.TEX_DOOR_TOP

            door = Object((x - 1, y - 1), (1, 1), texture, editable=False)
            self.objects.append(door)

        self.grid.draw()

    def perform_action(self, cell):
        action = self.current_action
        if action == None:
            pass
        elif action[0] == 'add':
            obj = self.object_palette[action[1]](cell)
            self.add_object(obj)
        elif action[0] == 'remove':
            self.remove_object(cell)
        elif action[0] == 'move':
            if action[1] != None:
                if self.move_object(cell, action[1]):
                    self.current_action = ('move', None)
                    self.info_label.text = self.messages['move1']
                else:
                    self.info_label.text = self.messages['move_retry']
            else:
                selected_obj = self.select_object(cell)
                if selected_obj != None:
                    if selected_obj.editable:
                        self.current_action = ('move', selected_obj)
                        self.info_label.text = self.messages['move2']
                    else:
                        self.info_label.text += " " + self.messages['not_editable']
        else:
            self.current_action = None
            self.info_label = self.messages['default']

    def move_object(self, cell, obj):
        if self.is_free_area(cell, obj.size, ignore=[obj]):
            obj.pos = cell
            return True
        else:
            self.info_label.text += " " + self.messages['move_retry']
            return False

    def select_object(self, cell):
        x, y = cell
        for obj in self.objects:
            if (x in range(obj.pos[0], obj.pos[0] + obj.size[0]) and
                y in range(obj.pos[1], obj.pos[1] + obj.size[1])):
                return obj
        return None

    def add_object(self, obj):
        if self.is_free_area(obj.pos, obj.size):
            self.objects.append(obj)
            return True
        else:
            return False

    def remove_object(self, cell):
        selected_obj = self.select_object(cell)
        if selected_obj != None and selected_obj.editable:
            self.objects.remove(selected_obj)
            return True

        return False

    def is_free_area(self, pos, size, ignore=[]):
        delta_x = pos[0] + size[0]
        delta_y = pos[1] + size[1]
        if delta_x > self.n_cols or delta_y > self.n_rows:
            return False
        for x in range(pos[0], delta_x):
            for y in range(pos[1], delta_y):
                obstacle = self.select_object((x,y))
                if obstacle != None and not obstacle in ignore:
                    return False
        return True

    def _select_obj(self, btn):
        self.dropdown.select(btn.text)
        self.current_action = ('add', btn.text)
        self.info_label.text = self.messages['add']

    def _select_rem(self, btn):
        self.current_action = ('remove', None)
        self.info_label.text = self.messages['remove']

    def _select_move(self, btn):
        self.current_action = ('move', None)
        self.info_label.text = self.messages['move1']


class TestApp(App):

    def __init__(self, n_cells, object_palette):
        super(TestApp, self).__init__()
        self.n_cells = n_cells
        self.object_palette = object_palette

    def build(self):
        layout = BoxLayout(orientation='horizontal')
        layout.add_widget(GridUI(self.n_cells, self.object_palette))
        return layout


if __name__ == "__main__":

    size = (20,15)
    pic_folder = './resources/'
    object_palette = {'Table 2x1' : lambda pos : Object(pos, (2,1), pic_folder + 'table_2_1.png'),
                      'Table 1x2' : lambda pos : Object(pos, (1,2), pic_folder + 'table_2_1.png'),
                      'Chair 1x1' : lambda pos : Object(pos, (1,1), pic_folder + 'chair_1_1.png')}

    TestApp(size, object_palette).run()

