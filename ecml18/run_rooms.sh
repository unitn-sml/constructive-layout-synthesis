#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

S=1 # domain seed
N=4 # gorups of users and users seed
U=5 # users per group
R=5 #  number of rooms
A=0.3 # alpha value

pycmd="python3.5 $DIR/main.py"


users="$(for i in $(seq	 1 $N); do echo 'users/user_s'$i'_n'$R'.pickle' ; done;)"

# GENERATE DOMAIN

domain="domains/rooms_n"$R".pickle"
#rm -f $domain
#$pycmd -s $S generate domain -D $domain  Rooms --n-rooms $R

# GENERATE USERS

#rm -rf $users
#for i in  $(seq 1 $N); do
#	$pycmd -s $i generate users -D $domain -U 'users/user_s'$i'_n'$R'.pickle' -n $U
#done

# SIMULATE 

for i in  $(seq 1 $N); do
	rm -f 'outputs/output_rooms_s'$i'_n'$R'_a'$A'.pickle'
	$pycmd -s $S -v simulate -D $domain -U 'users/user_s'$i'_n'$R'.pickle' -O 'outputs/output_rooms_s'$i'_n'$R'_a'$A'.pickle' coactive  --alpha $A  > 'outputs/output_rooms_s'$i'_n'$R'_a'$A'.out'  &
	sleep 1
done




