include "globals.mzn";


predicate fdiffn(array [int] of var float: x, array [int] of var float: y, array [int] of var float: dx, array [int] of var float: dy) =
    assert(index_set(x) == index_set(y) /\ index_set(x) == index_set(dx) /\ 
           index_set(x) == index_set(dy), "fdiffn: index set mismatch",
           forall (i, j in index_set(x) where i < j)(
               x[i] + dx[i] <= x[j] \/ 
               y[i] + dy[i] <= y[j] \/
               x[j] + dx[j] <= x[i] \/
               y[j] + dy[j] <= y[i]
           ));


function var float: xdist(var float: x1, var float: x2, var float: dx1) =
    x2 - x1 - dx1;


function var float: ydist_above(var float: y1, var float: y2, var float: dy1) =
    y2 - y1 - dy1;


function var float: ydist_below(var float: y1, var float: y2, var float: dy2) =
    y1 - y2 - dy2;


%% DOMAIN %%

float: SIDE = 100.0;

int: N_TABLES = 3;
set of int: TABLES = 1..N_TABLES;

% int: N_TYPES = 2;
% int: TYPES = 1..N_TYPES;
% array[TYPES] of int: table_type_d1 = [2, 1];
% array[TYPES] of int: table_type_d2 = [3, 4];
% array[TABLES] of var TYPES: table_type;

array[TABLES] of var 0.0..SIDE: x;
array[TABLES] of var 1.0..1.0: dx;
array[TABLES] of var 0.0..SIDE: y;
array[TABLES] of var 1.0..1.0: dy;

constraint increasing(x);
constraint fdiffn(x, y, dx, dy);


%% UTILITY %%

int: N_FEATURES = 3;
set of int: FEATURES = 1..N_FEATURES;

int: N_TABLES_COMBS = floor(N_TABLES * (N_TABLES - 1) / 2);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var 0.0..SIDE: xdists = [xdist(x[t1], x[t2], dx[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var -SIDE..SIDE: ydists_above = [ydist_above(y[t1], y[t2], dy[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var -SIDE..SIDE: ydists_below = [ydist_below(y[t1], y[t2], dy[t2]) | t1, t2 in TABLES where t1 < t2];

var 0.0..SIDE: mean_xdist = sum(xdists) / N_TABLES_COMBS;
var -SIDE..SIDE: mean_ydist_above = sum(ydists_above) / N_TABLES_COMBS;
var -SIDE..SIDE: mean_ydist_below = sum(ydists_below) / N_TABLES_COMBS;

array[FEATURES] of float: w = [1.0, 1.0, 1.0];
array[FEATURES] of var float: phi = [mean_xdist, mean_ydist_above, mean_ydist_below];

var float: obj = sum(f in FEATURES)(w[f] * phi[f]);
solve
    :: seq_search([
        float_search([obj], 0.1, first_fail, indomain_reverse_split, complete),
%        float_search([mean_xdist], 0, first_fail, indomain_reverse_split, complete),
%        float_search([mean_ydist_above], 0, first_fail, indomain_reverse_split, complete),
%        float_search(ydists_below, 0, first_fail, indomain_reverse_split, complete),
%        float_search(ydists_below, 0, first_fail, indomain_reverse_split, complete),
%        float_search(x, 0.1, first_fail, indomain_reverse_split, complete),
%        float_search(y, 0.1, first_fail, indomain_split, complete),
    ])
    maximize obj;

output [
        "x = " ++ show(x) ++ "\n" ++
        "y = " ++ show(y) ++ "\n" ++
        "mean_xdist = " ++ show(mean_xdist) ++ "\n" ++
        "mean_ydist_above = " ++ show(mean_ydist_above) ++ "\n" ++
        "mean_ydist_below = " ++ show(mean_ydist_below) ++ "\n" ++
        "obj = " ++ show(obj)
        ];


