#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

S=1 # domain seed
N=4 # gorups of users and users seed
U=5 # users per group
E=8 #  number of tables
A=0.3 # alpha value

pycmd="python3.5 $DIR/main.py"


domain="domains/tables_n"$E".pickle"

# SIMULATE 

for i in  $(seq 1 $N); do
	$pycmd -s $S -v simulate -D $domain -U 'users/user_tables_s'$i'_n'$E'.pickle' -O 'outputs/output_tables_s'$i'_n'$E'_a'$A'.pickle' coactive  --alpha $A  > 'outputs/output_tables_s'$i'_n'$R'_a'$A'.out'  &
	sleep 1
done




