#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

S=1 # domain seed
N=4 # gorups of users and users seed
U=5 # users per group
E=8 #  number of tables
A=0.3 # alpha value

pycmd="python3.5 $DIR/main.py"



# GENERATE DOMAIN

domain="domains/tables_n"$E".pickle"
#rm -f $domain
$pycmd -s $S generate domain -D $domain  Tables --n-tables $E

# GENERATE USERS

#rm -rf $users
for i in  $(seq 1 $N); do
	$pycmd -s $i generate users -D $domain -U 'users/user_tables_s'$i'_n'$E'.pickle' -n $U
done





