include "globals.mzn";
include "alldifferent.mzn";


%% DOMAIN %%


int: SIDE;
set of int: EDGE = 1..SIDE;
set of int: DISTS = 0..SIDE;
set of int: DISTN = -SIDE..SIDE;

int: N_TABLES;
set of int: TABLES = 1..N_TABLES;

int: MAX_TABLE_SIDE = 2;
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
array[DOORS] of EDGE: door_x = [1, 5];
array[DOORS] of EDGE: door_y = [5, SIDE];
array[PASS] of var EDGE: pass_x;
array[PASS] of var EDGE: pass_y;
var PASS: pass_lenght;



constraint forall(t in TABLES)(x[t] + dx[t] - 1 <= SIDE);
constraint forall(t in TABLES)(y[t] + dy[t] - 1 <= SIDE);


% DOORS AND PATH BETWEEN DOORS CONSTRAINTS

constraint forall(d in DOORS)( assert((door_x[d] = 1) \/ (door_x[d] = SIDE) \/ (door_y[d] = 1) \/ (door_y[d] = SIDE), "door nr. "++ show(d) ++  " isn't attached to a wall "  ));

constraint pass_x[1] = door_x[1];
constraint pass_y[1] = door_y[1];
constraint forall(i in 1..pass_lenght-1)(abs(pass_x[i] - pass_x[i+1]) + abs(pass_y[i] - pass_y[i+1]) = 1);
constraint forall(i in pass_lenght..MAX_PASS)(pass_x[i] = door_x[2]);
constraint forall(i in pass_lenght..MAX_PASS)(pass_y[i] = door_y[2]);

int: MAX_FLAT_POS = SIDE*SIDE+SIDE;
set of int: FLAT_POS = 1..MAX_FLAT_POS;
set of int: POS_SIZES = 1..MAX_TABLE_SIDE*MAX_TABLE_SIDE*N_TABLES; 

% idea : store array of  flattened positions ( row + SIDE*column), one for the path between doors and another
%        for the squares covered by the tables. Impose then a non overlapping constraint 

array[PASS] of var FLAT_POS: pass_pos = [pass_x[i]*SIDE + pass_y[i] | i in PASS];
array[POS_SIZES] of var int: occuped_indexes = [ min([dx[t] + 1 - i,dy[t] + 1 -j,1])*(x[t] + i - 1 + SIDE*(y[t] + j - 2) )  | i,j in TABLE_SIDE, t in TABLES  ];

constraint forall(i in 1..pass_lenght)( forall( o in  1..MAX_TABLE_SIDE*MAX_TABLE_SIDE*N_TABLES )( pass_pos[i] != occuped_indexes[o]  )   );

% WALLS
int: N_WALLS = 1;
set of int: WALLS = 1..N_WALLS;

array[WALLS] of int: wall_x = [1];
array[WALLS] of int: wall_y = [1];
array[WALLS] of int: wall_dx = [4];
array[WALLS] of int: wall_dy = [4];

array[1..N_WALLS*N_TABLES] of var -1..SIDE: wall_x_dists = [max([-1, wall_x[w] - x[t] - dx[t],x[t] - wall_x[w] - wall_dx[w]]) | t in TABLES, w in WALLS];
array[1..N_WALLS*N_TABLES] of var -1..SIDE: wall_y_dists = [max([-1, wall_y[w] - y[t] - dy[t],y[t] - wall_y[w] - wall_dy[w]]) | t in TABLES, w in WALLS];

constraint forall(i in 1..N_WALLS*N_TABLES)( (wall_x_dists[i] >= 0) /\ (wall_y_dists[i] >= 0)  );




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
var U_DOM: utility = sum(w_phi) ;


solve maximize utility;

output [
        "x = " ++ show(x) ++ ";\n" ++
        "dx = " ++ show(dx) ++ ";\n" ++
        "y = " ++ show(y) ++ ";\n" ++
        "dy = " ++ show(dy) ++ ";\n" ++
        "pass_lenght = " ++ show(pass_lenght) ++ ";\n" ++
        "pass_x = " ++ show(pass_x) ++ ";\n" ++
        "pass_y = " ++ show(pass_y) ++ ";\n" ++
        "door_x = " ++ show(door_x) ++ ";\n" ++
        "door_y = " ++ show(door_y) ++ ";\n" ++
        "wall_dx = " ++ show(wall_dx) ++ ";\n" ++
        "wall_dy = " ++ show(wall_dy) ++ ";\n" ++
        "wall_x = " ++ show(wall_x) ++ ";\n" ++
        "wall_y = " ++ show(wall_y) ++ ";\n" ++
        "side = " ++ show(SIDE) ++ ";\n" ++
%        "mat = " ++ show(presence_mat) ++ ";\n" ++
        "utility = " ++ show(utility) ++ ";"
        ];

