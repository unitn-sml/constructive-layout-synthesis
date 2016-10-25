include "globals.mzn";



%% DOMAIN %%


int: SIDE;
int: N_TABLES;
set of int: DISTS = 0..SIDE;
set of float: F_DISTS = 0.0..int2float(SIDE);
set of int: DISTN = -SIDE..SIDE;
set of int: EDGE = 1..SIDE-1;
set of int: TABLES = 1..N_TABLES;

array[TABLES] of var EDGE: x;
array[TABLES] of var EDGE: y;

constraint increasing(x);
constraint diffn(x, y, [3 | i in TABLES], [3 | i in TABLES]);

%% PHI %%

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var DISTS: xdists = [x[t2] - x[t1]| t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_above = [y[t2] - y[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_below = [y[t1] - y[t2]  | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTS: ydists = [max(ydists_above[t1 + t2 - 2], ydists_below[t1 + t2 - 2]) | t1, t2 in TABLES where t1 < t2];

var F_DISTS: mean_xdist = sum(xdists) / N_TABLES_COMBS;
var F_DISTS: mean_ydist = sum(ydists) / N_TABLES_COMBS;

var DISTS: max_xdist = max(xdists);
var DISTS: min_xdist = min(xdists);
var DISTS: max_ydist = max(ydists);
var DISTS: min_ydist = min(ydists);

float: PHI_LB = 0.0;
float: PHI_UB = int2float(SIDE);
set of float: PHI_DOM = PHI_LB..PHI_UB;

int: N_FEATURES = 6;
set of int: FEATURES = 1..N_FEATURES;
array[FEATURES] of var PHI_DOM: phi = [mean_xdist, mean_ydist, max_xdist, min_xdist, max_ydist, min_ydist];



%% UTILITY %%

array[FEATURES] of float: w;

float: W_LB = lb_array(w);
float: W_UB = ub_array(w);
float: W_PHI_UB = max(W_LB*PHI_LB, W_UB*PHI_UB);
float: W_PHI_LB = min([W_LB*PHI_LB, W_UB*PHI_LB, W_LB*PHI_UB, W_UB*PHI_UB]);
float: U_LB = N_FEATURES * W_PHI_LB;
float: U_UB = N_FEATURES * W_PHI_UB;
set of float: W_PHI_DOM = W_PHI_LB..W_PHI_UB;
set of float: U_DOM = U_LB..U_UB;

array[FEATURES] of var float: w_phi = [w[f] * phi[f] | f in FEATURES];
var float: utility = sum(w_phi);


solve
    :: seq_search([
        float_search([mean_ydist], 0.1, first_fail, indomain_reverse_split, complete),
        int_search(x ++ y, dom_w_deg, indomain_median, complete),
    ])
    maximize utility;

output [
        "x = " ++ show(x) ++ ";\n" ++
        "y = " ++ show(y) ++ ";\n" ++
        "utility = " ++ show(utility) ++ ";"
        ];

