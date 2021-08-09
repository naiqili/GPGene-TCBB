#!/bin/bash
for i in `seq 1 10`;
do
    unset DISPLAY 
    python synnet_andnot.py --kern Poly2
    java -jar ~/Lab/GNW/sandbox/gnw3-standalone.jar --evaluate --goldstandard ./data/synthetic/synnet_andnot.tsv --prediction ./res/synthetic_andnot.txt &>> ./res/synthetic_andnot_all2.txt
done