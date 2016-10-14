include "globals.mzn";


predicate fdiffn(array [int] of var float: x, array [int] of var float: y, array [int] of var float: dx, array [int] of var float: dy) =
    assert(index_set(x)==index_set(y) /\ index_set(x)==index_set(dx) /\ 
           index_set(x)==index_set(dy), "diffn: index set mismatch",
           forall (i, j in index_set(x) where i < j)(
               x[i] + dx[i] <= x[j] \/ 
               y[i] + dy[i] <= y[j] \/
               x[j] + dx[j] <= x[i] \/
               y[j] + dy[j] <= y[i]
           ));


function var float: pwdist(var float: x1, var float: x2, var float: dx1,
                           var float: y1, var float: y2, var float: dy1, 
                           var float: dy2) =
    max(0.0, x2 - x1 - dx1) + max(0, min(y2 - y1 - dy1, y1 - y2 - dy2));



%% DOMAIN %%

float: SIDE = 100.0;

int: N_TABLES = 2;
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

int: N_FEATURES = 1;
set of int: FEATURES = 1..N_FEATURES;

float: combs = N_TABLES * (N_TABLES - 1) / 2;

var float: mean_dist = sum(t1, t2 in TABLES where t1 < t2)(
                           pwdist(x[t1], x[t2], dx[t1],
                                  y[t1], y[t2], dy[t1], dy[t2])) / combs;

array[FEATURES] of float: w;
array[FEATURES] of var float: phi = [mean_dist];

var float: obj = sum(f in FEATURES)(w[f] * phi[f]);
solve
    :: float_search([mean_dist], 0.1, input_order, indomain_reverse_split, complete)
      maximize obj;

% output [
%         "mean_dist = " ++ show(mean_dist) ++ "\n" ++
%         "x = " ++ show(x)
%         ];


