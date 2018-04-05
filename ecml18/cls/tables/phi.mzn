include "globals.mzn";

%% DOMAIN %%

int: SIDE;
int: N_TABLES;
set of int: EDGE = 1..SIDE;
set of int: TABLES = 1..N_TABLES;
set of int: DISTN = -SIDE..SIDE;

int: MAX_TABLE_SIDE = 2;
set of int: TABLE_SIDE = 1..MAX_TABLE_SIDE;

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

%% DOORS %%
int: MAX_PASS = 2*SIDE;
set of int: PASS = 1..MAX_PASS;

set of int: TABLES0 = 0..N_TABLES;
int: N_DOORS = 2;

set of int: DOORS = 1..N_DOORS;
array[DOORS] of EDGE: door_x;
array[DOORS] of EDGE: door_y;
array[PASS] of var EDGE: pass_x;
array[PASS] of var EDGE: pass_y;


%% WALLS %%

int: N_WALLS ;
set of int: WALLS = 1..N_WALLS;

array[WALLS] of int: wall_x;
array[WALLS] of int: wall_y;
array[WALLS] of int: wall_dx;
array[WALLS] of int: wall_dy;

array[WALLS, TABLES] of DISTN: input_wall_x_dists = array2d(WALLS, TABLES, [max([wall_x[w] - input_x[t] - input_dx[t], input_x[t] - wall_x[w] - wall_dx[w]]) | w in WALLS, t in TABLES]);
array[WALLS, TABLES] of DISTN: input_wall_y_dists = array2d(WALLS, TABLES, [max([wall_y[w] - input_y[t] - input_dy[t], input_y[t] - wall_y[w] - wall_dy[w]]) | w in WALLS, t in TABLES]);


array[TABLES] of EDGE: input_x;
array[TABLES] of EDGE: input_y;
array[TABLES] of TABLE_SIDE: input_dx;
array[TABLES] of TABLE_SIDE: input_dy;

array[TABLES_COMBS] of int: input_xdists = [max(0, input_x[t2] - input_x[t1] - input_dx[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists_above = [input_y[t2] - input_y[t1] - input_dy[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists_below = [input_y[t1] - input_y[t2] - input_dy[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists = [max([0, input_ydists_above[t], input_ydists_below[t]]) | t in TABLES_COMBS];

int: input_max_xdist = max(input_xdists);
int: input_min_xdist = min(input_xdists);
int: input_max_ydist = max(input_ydists);
int: input_min_ydist = min(input_ydists);

int: N_FEATURES = 10;
set of int: FEATURES = 1..N_FEATURES;
int: input_left_side_dist = min(input_x);
int: input_right_side_dist = SIDE + 1 - max([input_x[t] + input_dx[t] | t in TABLES]);
int: input_lower_side_dist = min(input_y);
int: input_upper_side_dist = SIDE + 1 - max([input_y[t] + input_dy[t] | t in TABLES]);

array[TABLES] of int: input_min_dist_sides = [min([input_x[t], input_y[t], SIDE + 1 - input_x[t] - input_dx[t], SIDE + 1 - input_y[t] - input_dy[t]]) | t in TABLES];
array[TABLES] of int: input_min_dist_walls = [min([max(input_wall_x_dists[i,j], input_wall_y_dists[i,j]) | i in WALLS]) | j in TABLES];

int: input_max_dist_from_walls_sides = max([min(input_min_dist_sides[i], input_min_dist_walls[i]) | i in TABLES]);
int: input_max_dist_from_walls = max(input_min_dist_walls);
int: input_max_dist_from_sides = max(input_min_dist_sides);

int: input_min_dist_from_walls = min(input_min_dist_walls);
int: input_min_dist_from_sides = min(input_min_dist_sides);

array[TABLES] of 1..2: input_table_types = [input_dx[t] + input_dy[t] - 1 | t in TABLES];
array[1..2] of TABLES0: input_table_types_count = [sum(t in TABLES)(input_table_types[t] = i) | i in 1..2];

array[FEATURES] of int: phi = array1d(FEATURES, [input_max_xdist, input_min_xdist, input_max_ydist, input_min_ydist, input_max_dist_from_walls, input_min_dist_from_walls, input_max_dist_from_sides, input_min_dist_from_sides, input_table_types_count[1], input_table_types_count[2]]);

array[FEATURES] of float: normalizers = array1d(FEATURES, [1.0/SIDE | i in 1..8] ++ [1.0/N_TABLES | i in 1..2]);

%% SOLVE %%


solve satisfy;

%output ["phi = " ++ show(phi) ++ ";"];

