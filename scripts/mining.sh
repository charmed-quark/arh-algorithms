#!/bin/bash

FILEPATH=/home/charlie/thesis/mushrooms/vidpso/measurements/
JARPATH=/home/charlie/spmf.jar
SUP=40
CONF=70

for file in ${FILEPATH}*.txt; do
    echo $file
    java -jar ${JARPATH} run FPGrowth_association_rules ${file} ${file%.*}_mined.txt ${SUP}% ${CONF}%
done