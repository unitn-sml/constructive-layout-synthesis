

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_tables(x, y, output_file):
    side = x['SIDE']
    tables = list(zip(y['x'], y['dx'], y['y'], y['dy']))
    doors = list(zip(x['door_x'], x['door_y']))
    walls = list(zip(x['wall_x'], x['wall_dx'], x['wall_y'], x['wall_dy']))

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xbound((0, side))
    ax.set_ybound((0, side))
    ax.set_xticks(range(side))
    ax.set_yticks(range(side))
    ax.grid(True)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis='both',
        which='both',
        top='off',
        left='off',
        right='off',
        bottom='off',
        labelleft='off',
        labelbottom='off')

    for door_x, door_y in doors:
        door_x -= 1
        door_y -= 1
        width = 0.5 if door_x == 0 else 1
        door_dx = 0.5 if door_x == side - 1 else 0
        height = 0.5 if door_y == 0 else 1
        door_dy = 0.5 if door_y == side - 1 else 0
        r = patches.Rectangle((door_x + door_dx, door_y + door_dy), width, height, edgecolor='none', facecolor='#E5292B')
        ax.add_patch(r)

    for table_x, table_dx, table_y, table_dy in tables:
        table_x -= 1
        table_y -= 1
        r = patches.Rectangle((table_x, table_y), table_dx, table_dy, hatch='\\', edgecolor='#7f4224', facecolor='#A0522D')
        ax.add_patch(r)

    for wall_x, wall_dx, wall_y, wall_dy in walls:
        wall_x -= 1
        wall_y -= 1
        r = patches.Rectangle((wall_x, wall_y), wall_dx, wall_dy, hatch='/', edgecolor='#121212', facecolor='#FFFFFF')
        ax.add_patch(r)

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

class ColorPicker:

    KITCHEN = 1
    DININGROOM = 2
    LIVINGROOM = 3
    BEDROOM = 4
    BATHROOM = 5
    CORRIDOR = 6
    colors = {
        KITCHEN : [(1 - (i * 0.1), 0, 0, 0.7) for i in range(5)],
        DININGROOM : [(0, 1 - (i * 0.1), 0, 0.7) for i in range(5)],
        LIVINGROOM : [(0, 0, 1 - (i * 0.1), 0.7) for i in range(5)],
        BEDROOM : [(0.5 - (i * 0.05), 0.5 - (i * 0.05), 0, 0.7) for i in range(5)],
        BATHROOM : [(0, 0.5 - (i * 0.05), 0.5 - (i * 0.05), 0.7) for i in range(5)],
        CORRIDOR : [(0.5 - (i * 0.05), 0, 0.5 - (i * 0.05), 0.7) for i in range(5)],
    }


    def __init__(self):
        self.used = {i: {} for i in range(1,7)}

    def getc(self, type_r, id_s):
        if id_s not in self.used[type_r]:
            l = len(self.used[type_r])
            self.used[type_r][id_s] = ColorPicker.colors[type_r][l]
        return self.used[type_r][id_s]


def draw_rooms(x, y, output_file):
    side = x['SIDE']
    subrooms = list(zip(y['x'], y['y'], y['dx'], y['dy']))
    walls = list(zip(x['wall_x'], x['wall_y'], x['wall_dx'], x['wall_dy']))

    subtype = x['sub_type']
    roomtype = x['room_type']
    valid = y['valid']

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xbound((0, side))
    ax.set_ybound((0, side))
    ax.set_xticks(range(side))
    ax.set_yticks(range(side))
    ax.grid(True)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis='both',
        which='both',
        top='off',
        left='off',
        right='off',
        bottom='off',
        labelleft='off',
        labelbottom='off')

    cp = ColorPicker()

    for i, (subroom_x, subroom_dx, subroom_y, subroom_dy) in enumerate(subrooms):
        subroom_x -= 1
        subroom_y -= 1
        if valid[i]:
            color = cp.getc(roomtype[subtype[i]], subtype[i])
            r = patches.Rectangle((subroom_x, subroom_y), subroom_dx, subroom_dy, edgecolor=color, facecolor=color)
            ax.add_patch(r)

    for wall_x, wall_dx, wall_y, wall_dy in walls:
        wall_x -= 1
        wall_y -= 1
        r = patches.Rectangle((wall_x, wall_y), wall_dx, wall_dy, hatch='/', edgecolor='#121212', facecolor=(1,1,1,0.8))
        ax.add_patch(r)

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)



