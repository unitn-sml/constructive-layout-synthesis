include "globals.mzn";

%% DOMAIN %%

int: SIDE = 100;
set of int: EDGE = 0..SIDE;

int: N_TABLES = 20;
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
array[TABLES_COMBS] of var int: ydists = [max(ydists_above[t1 + t2 - 2], ydists_below[t1 + t2 - 2]) | t1, t2 in TABLES where t1 < t2];

%var int: sum_xdist = sum(xdists);
%var int: sum_ydist = sum(ydists);
%var float: mean_xdist = sum_xdist / N_TABLES_COMBS;
%var float: mean_ydist = sum_ydist / N_TABLES_COMBS;

var int: max_xdist = max(xdists);
var int: min_xdist = min(xdists);
var int: max_ydist = max(ydists);
var int: min_ydist = min(ydists);

int: N_FEATURES = 4;
set of int: FEATURES = 1..N_FEATURES;

array[FEATURES] of float: w;
array[FEATURES] of int: w0 = [round(w[f]*100) | f in FEATURES];

int: W_LB = lb_array(w0);
int: W_UB = ub_array(w0);
int: PHI_LB = 0;
int: PHI_UB = SIDE;
int: W_PHI_LB = min(W_LB*PHI_LB, W_UB*PHI_UB);
int: W_PHI_UB = max([W_LB*PHI_LB, W_UB*PHI_LB, W_LB*PHI_UB, W_UB*PHI_UB]);
int: OBJ_LB = N_FEATURES * W_PHI_LB;
int: OBJ_UB = N_FEATURES * W_PHI_UB;
set of int: PHI_DOM = PHI_LB..PHI_UB;
set of int: W_PHI_DOM = W_PHI_LB..W_PHI_UB;
set of int: OBJ_DOM = OBJ_LB..OBJ_UB;

array[FEATURES] of var PHI_DOM: phi = [max_xdist, min_xdist, max_ydist, min_ydist];
array[FEATURES] of var W_PHI_DOM: w_phi = [w0[f] * phi[f] | f in FEATURES];
var OBJ_DOM: obj = sum(w_phi);

solve
    :: seq_search([
        int_search(ydists_above ++ ydists_below, dom_w_deg, indomain_max, complete),
        int_search([max_xdist, min_xdist, max_ydist, min_ydist], dom_w_deg, indomain_max, complete),
        %float_search(w_phi, 0.1, first_fail, indomain_reverse_split, complete),
        int_search(xdists, dom_w_deg, indomain_max, complete),
        int_search(w_phi, first_fail, indomain_max, complete),
        %float_search([w_phi[1], w_phi[3]], 0.1, first_fail, indomain_reverse_split, complete),
        %float_search([w_phi[2], w_phi[4]], 0.1, first_fail, indomain_split, complete)
    ])
    maximize obj;

output [
        "x = " ++ show(x) ++ "\n" ++
        "y = " ++ show(y) ++ "\n" ++
        "obj = " ++ show(obj)
        ];


