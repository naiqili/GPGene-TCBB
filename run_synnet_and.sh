#!/bin/bash
for i in `seq 1 10`;
do
    unset DISPLAY 
    python synnet_and.py --kern Poly2
    java -jar ~/Lab/GNW/sandbox/gnw3-standalone.jar --evaluate --goldstandard ./data/synthetic/synnet_and.tsv --prediction ./res/synthetic_and.txt &>> ./res/synthetic_and_all2.txt
done