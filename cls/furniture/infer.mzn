include "globals.mzn";


function var int: manhattan_distance(var int: x1, var int: y1, var int: dx1, var int: dy1, var int: x2, var int: y2, var int: dx2, var int: dy2) =
    max([x1 - x2 - dx2, x2 - x1 - dx1, 0]) + max([y1 - y2 - dy2, y2 - y1 - dy1, 0]);


function var int: manhattan_distance_one(var int: x1, var int: y1, var int: dx1, var int: dy1, var int: x2, var int: y2) =
    max([x1 - x2 - 1, x2 - x1 - dx1, 0]) + max([y1 - y2 - 1, y2 - y1 - dy1, 0]);


function var int: manhattan_distance_one(var int: x1,var int: y1,var int: dx1,var int: dy1,int: x2,int: y2) =
    max([x1 - x2 - 1, x2 - x1 - dx1, 0]) + max([y1 - y2 - 1, y2 - y1 - dy1, 0]);

function var int: flat_pos(var int: x, var int: y) =
    x + SIDE * (y - 1);



%% DOMAIN %%


int: SIDE;
set of int: EDGE = 1..SIDE;
set of int: DISTS = 0..SIDE;
set of int: DISTN = -SIDE..SIDE;

int: N_TABLES;
set of int: TABLES = 1..N_TABLES;

int: MAX_TABLE_SIDE = 2;
set of int: TABLE_SIDE = 1..MAX_TABLE_SIDE;

array[TABLES] of var EDGE: x;
array[TABLES] of var TABLE_SIDE: dx;
array[TABLES] of var EDGE: y;
array[TABLES] of var TABLE_SIDE: dy;

constraint increasing(x);

int: MAX_PASS = 2*SIDE;
set of int: PASS = 1..MAX_PASS;

int: N_DOORS = 2;
set of int: DOORS = 1..N_DOORS;
array[DOORS] of EDGE: door_x ;
array[DOORS] of EDGE: door_y ;
array[PASS] of var EDGE: pass_x;
array[PASS] of var EDGE: pass_y;
var PASS: pass_lenght;

constraint forall(t in TABLES)(x[t] + dx[t] - 1 <= SIDE);
constraint forall(t in TABLES)(y[t] + dy[t] - 1 <= SIDE);
constraint forall(t in TABLES)(dx[t] + dy[t] <= 3);



%% DOORS AND PATH BETWEEN DOORS %%


constraint pass_x[1] = door_x[1];
constraint pass_y[1] = door_y[1];
constraint forall(i in 1..pass_lenght-1)(abs(pass_x[i] - pass_x[i+1]) + abs(pass_y[i] - pass_y[i+1]) = 1);
constraint forall(i in pass_lenght..MAX_PASS)(pass_x[i] = door_x[2]);
constraint forall(i in pass_lenght..MAX_PASS)(pass_y[i] = door_y[2]);
constraint forall(i in DOORS, j in TABLES)(manhattan_distance_one(x[j], y[j], dx[j], dy[j], door_x[i], door_y[i]) >= 2);

int: MAX_FLAT_POS = SIDE*SIDE+SIDE;
set of int: FLAT_POS = 1..MAX_FLAT_POS;
set of int: POS_SIZES = 1..MAX_TABLE_SIDE*MAX_TABLE_SIDE*N_TABLES; 

array[PASS] of var FLAT_POS: pass_pos = [flat_pos(pass_x[i], pass_y[i]) | i in PASS];
array[POS_SIZES] of var int: occupied_indices = [min([dx[t] + 1 - i, dy[t] + 1 - j, 1]) * flat_pos(x[t] + i - 1, y[t] + j - 1) | i, j in TABLE_SIDE, t in TABLES];
constraint forall(i in 1..pass_lenght)(forall(j in  POS_SIZES)(pass_pos[i] != occupied_indices[j]));


%% WALLS %%

int: N_WALLS;
set of int: WALLS = 1..N_WALLS;

array[WALLS] of int: wall_x;
array[WALLS] of int: wall_y;
array[WALLS] of int: wall_dx;
array[WALLS] of int: wall_dy;

array[WALLS, TABLES] of var DISTN: wall_x_dists = array2d(WALLS, TABLES, [max([wall_x[w] - x[t] - dx[t], x[t] - wall_x[w] - wall_dx[w]]) | w in WALLS, t in TABLES]);
array[WALLS, TABLES] of var DISTN: wall_y_dists = array2d(WALLS, TABLES, [max([wall_y[w] - y[t] - dy[t], y[t] - wall_y[w] - wall_dy[w]]) | w in WALLS, t in TABLES]);

constraint forall(i in WALLS, j in TABLES)(not (wall_x_dists[i, j] < 0 /\ wall_y_dists[i, j] < 0));
constraint forall(i in WALLS, j in PASS)(not (pass_x[j] >= wall_x[i] /\ pass_x[j] < wall_x[i] + wall_dx[i] /\ pass_y[j] >= wall_y[i] /\ pass_y[j] < wall_y[i] + wall_dy[i]));



%% PHI %%

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var DISTS: xdists = [max(0, x[t2] - x[t1] - dx[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_above = [y[t2] - y[t1] - dy[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_below = [y[t1] - y[t2] - dy[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTS: ydists = [max([0, ydists_above[t], ydists_below[t]]) | t in TABLES_COMBS];

constraint forall(t1, t2 in TABLES where t1 < t2)(xdists[t1 + t2 - 1] + ydists[t1 + t2 - 1] >= dx[t1] + dx[t2] + dy[t1] + dy[t2] - 3);
constraint forall (t in TABLES_COMBS)(xdists[t] + ydists[t] >= 1);

var DISTS: max_xdist = max(xdists);
var DISTS: min_xdist = min(xdists);
var DISTS: max_ydist = max(ydists);
var DISTS: min_ydist = min(ydists);

array[TABLES] of var int: min_dist_sides = [min([x[t], y[t], SIDE + 1 - x[t] - dx[t], SIDE + 1 - y[t] - dy[t]]) | t in TABLES];
array[TABLES] of var int: min_dist_walls = [min([max(wall_x_dists[i, j], wall_y_dists[i, j]) | i in WALLS]) | j in TABLES];

var int: max_dist_from_walls = max(min_dist_walls);
var int: max_dist_from_sides = max(min_dist_sides);

var int: min_dist_from_walls = min(min_dist_walls);
var int: min_dist_from_sides = min(min_dist_sides);


int: N_FEATURES = 8;
set of int: FEATURES = 1..N_FEATURES;
array[FEATURES] of var int: phi = array1d(FEATURES, [max_xdist, min_xdist, max_ydist, min_ydist, max_dist_from_walls, min_dist_from_walls, max_dist_from_sides, min_dist_from_sides]);



%% UTILITY %%

array[FEATURES] of float: w;
array[FEATURES] of int: w1 = array1d(FEATURES, [round(w[f]*10000) | f in FEATURES]);
array[FEATURES] of var int: w_phi = array1d(FEATURES, [w1[f] * phi[f] | f in FEATURES]);
var int: utility = sum(w_phi);


solve maximize utility;

output [
        "x = " ++ show(x) ++ ";\n" ++
        "dx = " ++ show(dx) ++ ";\n" ++
        "y = " ++ show(y) ++ ";\n" ++
        "dy = " ++ show(dy) ++ ";\n" ++
        "pass_lenght = " ++ show(pass_lenght) ++ ";\n" ++
        "pass_x = " ++ show(pass_x) ++ ";\n" ++
        "pass_y = " ++ show(pass_y) ++ ";\n" ++
        "door_x = " ++ show(door_x) ++ ";\n" ++
        "door_y = " ++ show(door_y) ++ ";\n" ++
        "wall_dx = " ++ show(wall_dx) ++ ";\n" ++
        "wall_dy = " ++ show(wall_dy) ++ ";\n" ++
        "wall_x = " ++ show(wall_x) ++ ";\n" ++
        "wall_y = " ++ show(wall_y) ++ ";\n" ++
        "side = " ++ show(SIDE) ++ ";\n" ++
        "wall_x_dists = " ++ show(wall_x_dists) ++ ";\n" ++
        "wall_y_dists = " ++ show(wall_y_dists) ++ ";\n" ++
        "phi = " ++ show(phi) ++ ";\n" ++
        "utility = " ++ show(utility) ++ ";"
        ];

