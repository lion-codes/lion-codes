#!/bin/bash

cp inc/equations.txt tmp_equations.txt


# Get the k indices for which k is equal to zero
grep -nE "^0$" inc/k_vals.csv | awk 'BEGIN{FS="\:"}{print $1}' > zerok_idx

# Replace in equation file the expressions that involve zero k's with nothing
while read kidx; do
	sed -i -r "s/(\+|\-|^)1\.0\*k\($kidx\)(\*y\([0-9]*\))*//g" tmp_equations.txt
done < "zerok_idx"

# Find empty equation lines in the new equation set
grep -nE "^$" tmp_equations.txt | awk 'BEGIN{FS="\:"}{print $1}' > empty_eqs

# Find the y indices in these empty equations, based on original equation file
while read linenum; do
	sed -n "{$linenum,$linenum p}" inc/equations.txt
done < "empty_eqs" > foo
grep -E -o "y\([0-9]*\)" foo > foo2
grep -E -o "[0-9]*" foo2 > foo3
sort foo3 | uniq > y_idx

# Find if the y's involved still appear somewhere else in the new set of equations
if [[ -f y_status ]]; then
	rm y_status
else
	touch y_status
fi 

while read yidx; do
	var=`sed -nr "/y\($yidx\)/ p" tmp_equations.txt`
	if [[ -z $var ]]; then
		sed -n -r "/^$yidx/ s/$/\:0/p" y_idx >> y_status
	else
		sed -n -r "/^$yidx/ s/$/\:1/p" y_idx >> y_status
	fi
done < "y_idx"

cat y_status | grep -E "\:0" > y_todelete


# Check that the number of deleted equations is the same as the y's
deleq=`cat empty_eqs | wc -l`
delvar=`cat y_todelete | wc -l`

BADSET=10

if [[ $deleq -eq $delvar ]]; then
	echo "Number of equations to delete $deleq is equal to number of unknowns to delete $delvar"
else
	echo "Number of equations to delete $deleq is NOT equal to number of unknowns to delete $delvar"
	exit $BADSET
fi

# Delete the equations completely from the set
sed -i -r "/^$/ d" tmp_equations.txt

# TODO: Reorganize the indices


