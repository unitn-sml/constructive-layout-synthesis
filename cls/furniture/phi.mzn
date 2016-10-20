include "globals.mzn";

%% DOMAIN %%

int: SIDE;
int: N_TABLES;
set of int: EDGE = 1..SIDE;
set of int: TABLES = 1..N_TABLES;

array[TABLES] of EDGE: input_x;
array[TABLES] of 1..1: input_dx;
array[TABLES] of EDGE: input_y;
array[TABLES] of 1..1: input_dy;



%% PHI %%

int: N_TABLES_COMBS = round(N_TABLES * (N_TABLES - 1) / 2);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;

array[TABLES_COMBS] of int: xdists = [input_x[t2] - input_x[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: ydists_above = [input_y[t2] - input_y[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: ydists_below = [input_y[t1] - input_y[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: ydists = [max(ydists_above[t1 + t2 - 2], ydists_below[t1 + t2 - 2]) | t1, t2 in TABLES where t1 < t2];

int: max_xdist = max(xdists);
int: min_xdist = min(xdists);
int: max_ydist = max(ydists);
int: min_ydist = min(ydists);

int: PHI_LB = 0;
int: PHI_UB = SIDE;
set of int: PHI_DOM = PHI_LB..PHI_UB;

int: N_FEATURES = 4;
set of int: FEATURES = 1..N_FEATURES;
array[FEATURES] of PHI_DOM: phi = [max_xdist, min_xdist, max_ydist, min_ydist];


solve satisfy;

output ["phi = " ++ show(phi) ++ ";"];

