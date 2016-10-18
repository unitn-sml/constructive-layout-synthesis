include "globals.mzn";

%% DOMAIN %%

int: SIDE = 100;
set of int: EDGE = 0..SIDE;

int: N_TABLES = 10;
set of int: TABLES = 1..N_TABLES;

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

array[TABLES_COMBS] of var int: xdists = [x[t2] - x[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_above = [y[t2] - y[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_below = [y[t1] - y[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists = [max(ydists_above[t1 + t2 - 1], ydists_below[t1 + t2 - 1]) | t1, t2 in TABLES where t1 < t2];

constraint forall(i in TABLES_COMBS)(ydists_above[i] >= 0 \/ ydists_below[i] >= 0);

var int: sum_xdist = sum(xdists);
var int: sum_ydist = sum(ydists);
var float: mean_xdist = sum_xdist / N_TABLES_COMBS;
var float: mean_ydist = sum_ydist / N_TABLES_COMBS;


int: N_FEATURES = 2;
set of int: FEATURES = 1..N_FEATURES;

array[FEATURES] of float: w = [1.0, 1.0];
array[FEATURES] of var float: phi = [mean_xdist, mean_ydist];
array[FEATURES] of var float: w_phi = [w[f] * phi[f] | f in FEATURES];
var float: obj = sum(w_phi);

solve
    :: seq_search([
        int_search(ydists_above ++ ydists_below, dom_w_deg, indomain_max, complete),
        int_search([sum_xdist, sum_ydist], first_fail, indomain_max, complete),
    ])
    maximize obj;

output [
        "x = " ++ show(x) ++ "\n" ++
        "y = " ++ show(y) ++ "\n" ++
        "mean_xdist = " ++ show(mean_xdist) ++ "\n" ++
        "mean_ydist = " ++ show(mean_ydist) ++ "\n" ++
        "obj = " ++ show(obj)
        ];


