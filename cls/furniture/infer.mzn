include "globals.mzn";



%% DOMAIN %%


int: SIDE;
int: N_TABLES;
int: MAX_TABLE_SIDE = 3;
set of int: EDGE = 1..SIDE;
set of int: TABLES = 1..N_TABLES;

array[TABLES] of var EDGE: x;
array[TABLES] of var 1..MAX_TABLE_SIDE: dx;
array[TABLES] of var EDGE: y;
array[TABLES] of var 1..MAX_TABLE_SIDE: dy;

constraint increasing(x);
constraint diffn(x, y, dx, dy);

constraint forall(t in TABLES)(x[t] + dx[t] <= SIDE);
constraint forall(t in TABLES)(y[t] + dy[t] <= SIDE);



%% PHI %%

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var int: xdists = [x[t2] - x[t1]| t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_above = [y[t2] - y[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists_below = [y[t1] - y[t2]  | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var int: ydists = [max(ydists_above[t1 + t2 - 2], ydists_below[t1 + t2 - 2]) | t1, t2 in TABLES where t1 < t2];

%var int: sum_xdist = sum(xdists);
%var int: sum_ydist = sum(ydists);

var int: max_dx = max(dx);
var int: max_dy = max(dy);
var int: max_xdist = max(xdists);
var int: min_xdist = min(xdists);
var int: max_ydist = max(ydists);
var int: min_ydist = min(ydists);

int: PHI_LB = 0;
int: PHI_UB = SIDE;
set of int: PHI_DOM = PHI_LB..PHI_UB;

int: N_FEATURES = 6;
set of int: FEATURES = 1..N_FEATURES;
array[FEATURES] of var PHI_DOM: phi = [max_xdist, min_xdist, max_ydist, min_ydist, max_dx, max_dy];



%% UTILITY %%

array[FEATURES] of float: w;
array[FEATURES] of int: w0 = [round(w[f]*100.0) | f in FEATURES];
%array[FEATURES] of int: w0;

int: W_LB = lb_array(w0);
int: W_UB = ub_array(w0);
int: W_PHI_LB = min(W_LB*PHI_LB, W_UB*PHI_UB);
int: W_PHI_UB = max([W_LB*PHI_LB, W_UB*PHI_LB, W_LB*PHI_UB, W_UB*PHI_UB]);
int: U_LB = N_FEATURES * W_PHI_LB;
int: U_UB = N_FEATURES * W_PHI_UB;
set of int: W_PHI_DOM = W_PHI_LB..W_PHI_UB;
set of int: U_DOM = U_LB..U_UB;

array[FEATURES] of var W_PHI_DOM: w_phi = [w0[f] * phi[f] | f in FEATURES];
var U_DOM: utility = sum(w_phi);


solve
    :: seq_search([
        int_search(ydists_above ++ ydists_below, dom_w_deg, indomain_max, complete),
        int_search([max_xdist, min_xdist, max_ydist, min_ydist], dom_w_deg, indomain_max, complete),
        int_search([max_dx, max_dy], dom_w_deg, indomain_max, complete),
        int_search(xdists, dom_w_deg, indomain_max, complete),
        int_search(w_phi, first_fail, indomain_max, complete),
    ])
    maximize utility;

output [
        "x = " ++ show(x) ++ ";\n" ++
        "y = " ++ show(y) ++ ";\n" ++
        "dx = " ++ show(dx) ++ ";\n" ++
        "dy = " ++ show(dy) ++ ";\n" ++
        "dy = " ++ show(utility) ++ ";"
        ];

