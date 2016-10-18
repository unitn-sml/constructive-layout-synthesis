include "globals.mzn";


function var int: xdist(var int: x1, var int: x2, var int: dx1) =
    x2 - x1 - dx1;


function var int: ydist_above(var int: y1, var int: y2, var int: dy1) =
    y2 - y1 - dy1;


function var int: ydist_below(var int: y1, var int: y2, var int: dy2) =
    y1 - y2 - dy2;


%% DOMAIN %%

int: SIDE = 100;
float: F_SIDE = int2float(SIDE);
set of int: EDGE = 0..SIDE;
set of float: F_EDGE = 0.0..F_SIDE;

int: N_TABLES = 4;
set of int: TABLES = 1..N_TABLES;

% int: N_TYPES = 2;
% int: TYPES = 1..N_TYPES;
% array[TYPES] of int: table_type_d1 = [2, 1];
% array[TYPES] of int: table_type_d2 = [3, 4];
% array[TABLES] of var TYPES: table_type;

array[TABLES] of var EDGE: x;
array[TABLES] of var 1..1: dx;
array[TABLES] of var EDGE: y;
array[TABLES] of var 1..1: dy;

constraint increasing(x);
constraint diffn(x, y, dx, dy);

constraint forall(t in TABLES)(x[t] + dx[t] <= SIDE);
constraint forall(t in TABLES)(y[t] + dy[t] <= SIDE);



%% UTILITY %%

int: N_TABLES_COMBS = round(N_TABLES * (N_TABLES - 1) / 2);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var 0..SIDE: xdists = [xdist(x[t1], x[t2], dx[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_above = [ydist_above(y[t1], y[t2], dy[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_below = [ydist_below(y[t1], y[t2], dy[t2]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists = [max(ydists_above[t1 + t2 - 1], ydists_below[t1 + t2 - 1]) | t1, t2 in TABLES where t1 < t2];

int: MAX_SUM_DIST = N_TABLES_COMBS * SIDE;
set of int: SUM_SPAN = 0..MAX_SUM_DIST;

var SUM_SPAN: sum_xdist = sum(xdists);
var SUM_SPAN: sum_ydist = sum(ydists);
var F_EDGE: mean_xdist = sum_xdist / N_TABLES_COMBS;
var F_EDGE: mean_ydist = sum_ydist / N_TABLES_COMBS;

%array[TABLES_COMBS] of var float: xdist_diffs = [abs(xdists[t1 + t2 - 1] - mean_xdist) | t1, t2 in TABLES where t1 < t2];
%array[TABLES_COMBS] of var float: ydist_diffs = [abs(ydists[t1 + t2 - 1] - mean_ydist) | t1, t2 in TABLES where t1 < t2];
%var float: sd_xdist = sum(xdist_diffs) / N_TABLES_COMBS;
%var float: sd_ydist = sum(ydist_diffs) / N_TABLES_COMBS;

int: N_FEATURES = 2;
set of int: FEATURES = 1..N_FEATURES;

array[FEATURES] of float: w = [1.0, 1.0];
array[FEATURES] of var float: phi = [mean_xdist, mean_ydist];
array[FEATURES] of var float: w_phi = [w[f] * phi[f] | f in FEATURES];
var float: obj = sum(w_phi);

solve
    :: seq_search([
        int_search(xdists ++ ydists_above ++ ydists_below, dom_w_deg, indomain_max, complete),
    ])
    maximize obj;

output [
        "x = " ++ show(x) ++ "\n" ++
        "y = " ++ show(y) ++ "\n" ++
        "mean_xdist = " ++ show(mean_xdist) ++ "\n" ++
        "mean_ydist = " ++ show(mean_ydist) ++ "\n" ++
        "obj = " ++ show(obj)
        ];


