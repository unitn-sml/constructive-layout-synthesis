include "globals.mzn";



%% DOMAIN %%


int: SIDE;
set of int: EDGE = 1..SIDE;
set of int: DISTS = 0..SIDE;
set of int: DISTN = -SIDE..SIDE;

int: N_TABLES;
set of int: TABLES = 1..N_TABLES;

int: MAX_TABLE_SIDE = 1;
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
array[DOORS] of EDGE: door_x = [10, 1];
array[DOORS] of EDGE: door_y = [1, 20];
array[PASS] of var EDGE: pass_x;
array[PASS] of var EDGE: pass_y;
var PASS: pass_lenght;

constraint pass_x[1] = door_x[1];
constraint pass_y[1] = door_y[1];
constraint forall(i in 1..pass_lenght-1)(abs(pass_x[i] - pass_x[i+1]) + abs(pass_y[i] - pass_y[i+1]) = 1);
constraint forall(i in pass_lenght..MAX_PASS)(pass_x[i] = door_x[2]);
constraint forall(i in pass_lenght..MAX_PASS)(pass_y[i] = door_y[2]);

int: MAX_FLAT_POS = SIDE*SIDE+SIDE;
set of int: FLAT_POS = 1..MAX_FLAT_POS;

array[TABLES] of var FLAT_POS: tables_pos = [y[t]*SIDE + x[t] | t in TABLES];
array[PASS] of var FLAT_POS: pass_pos = [pass_y[i]*SIDE + pass_x[i] | i in PASS];
constraint forall(i in 1..pass_lenght)(forall(t in TABLES)(pass_pos[i] != tables_pos[t]));

%% PHI %%

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of var DISTS: xdists = [max(0, x[t2] - x[t1] - dx[t1]) | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_above = [y[t2] - y[t1] - dy[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTN: ydists_below = [y[t1] - y[t2] - dy[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of var DISTS: ydists = [max([0, ydists_above[t], ydists_below[t]]) | t in TABLES_COMBS];

constraint forall(t in TABLES_COMBS)(xdists[t] + ydists[t] >= 1);

var DISTS: max_xdist = max(xdists);
var DISTS: min_xdist = min(xdists);
var DISTS: max_ydist = max(ydists);
var DISTS: min_ydist = min(ydists);

int: N_FEATURES = 4;
set of int: FEATURES = 1..N_FEATURES;
array[FEATURES] of var DISTS: phi = array1d(FEATURES, [max_xdist, min_xdist, max_ydist, min_ydist]);


%% UTILITY %%

array[FEATURES] of float: w;
array[FEATURES] of int: w1 = array1d(FEATURES, [round(w[f]*100) | f in FEATURES]);

int: W_LB = lb_array(w1);
int: W_UB = ub_array(w1);
int: PHI_LB = 0;
int: PHI_UB = SIDE;
int: W_PHI_UB = max(W_LB*PHI_LB, W_UB*PHI_UB);
int: W_PHI_LB = min([W_LB*PHI_LB, W_UB*PHI_LB, W_LB*PHI_UB, W_UB*PHI_UB]);
set of int: W_PHI_DOM = W_PHI_LB..W_PHI_UB;

int: U_LB = N_FEATURES * W_PHI_LB;
int: U_UB = N_FEATURES * W_PHI_UB;
set of int: U_DOM = U_LB..U_UB;

array[FEATURES] of var W_PHI_DOM: w_phi = array1d(FEATURES, [w1[f] * phi[f] | f in FEATURES]);
var U_DOM: utility = sum(w_phi);


solve maximize utility;

output [
        "x = " ++ show(x) ++ ";\n" ++
        "dx = " ++ show(dx) ++ ";\n" ++
        "y = " ++ show(y) ++ ";\n" ++
        "dy = " ++ show(dy) ++ ";\n" ++
        "pass_lenght = " ++ show(pass_lenght) ++ ";\n" ++
        "pass_x = " ++ show(pass_x) ++ ";\n" ++
        "pass_y = " ++ show(pass_y) ++ ";\n" ++
        "utility = " ++ show(utility) ++ ";"
        ];

