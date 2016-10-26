include "globals.mzn";

%% DOMAIN %%

int: SIDE;
int: N_TABLES;
set of int: EDGE = 1..SIDE;
set of int: TABLES = 1..N_TABLES;

int: MAX_TABLE_SIDE = 2;
set of int: TABLE_SIDE = 1..MAX_TABLE_SIDE;

int: N_TABLES_COMBS = round(int2float(N_TABLES) * (int2float(N_TABLES) - 1.0) / 2.0);
set of int: TABLES_COMBS = 1..N_TABLES_COMBS;


array[TABLES] of EDGE: input_x;
array[TABLES] of EDGE: input_y;
array[TABLES] of TABLE_SIDE: input_dx;
array[TABLES] of TABLE_SIDE: input_dy;

array[TABLES_COMBS] of int: input_xdists = [input_x[t2] - input_x[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists_above = [input_y[t2] - input_y[t1] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists_below = [input_y[t1] - input_y[t2] | t1, t2 in TABLES where t1 < t2];
array[TABLES_COMBS] of int: input_ydists = [max(input_ydists_above[t1 + t2 - 2], input_ydists_below[t1 + t2 - 2]) | t1, t2 in TABLES where t1 < t2];

int: input_max_xdist = max(input_xdists);
int: input_min_xdist = min(input_xdists);
int: input_max_ydist = max(input_ydists);
int: input_min_ydist = min(input_ydists);

int: N_FEATURES = 8;
set of int: FEATURES = 1..N_FEATURES;
int: input_left_side_dist = min(input_x);
int: input_right_side_dist = SIDE - max([input_x[t] +input_dx[t] | t in TABLES]);
int: input_lower_side_dist = min(input_y);
int: input_upper_side_dist = SIDE - max([input_y[t] +input_dy[t] | t in TABLES]);

array[FEATURES] of int: phi = [input_max_xdist, input_min_xdist,input_max_ydist,input_min_ydist,input_left_side_dist,input_right_side_dist,input_lower_side_dist,input_upper_side_dist];

%% SOLVE %%



solve satisfy;

output ["phi = " ++ show(phi) ++ ";"];

