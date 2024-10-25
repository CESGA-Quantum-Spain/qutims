#!/bin/bash

# Define the range
start=34165
end=34174


# Loop over the range
for i in $(seq $start $end); do
    echo 
    echo "jobID $i"
    python visfit_santafe2.py "$i"
    python ../loss_curves_santafe.py "$i"
done
