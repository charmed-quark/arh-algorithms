#!/bin/bash

for dataset in "chess" "mushrooms" "bms1" "retail" "syn1K" "syn10K" "syn100K"
do
    for i in {0..4}
    do
        grep -x -v -f ../${dataset}/rules/${dataset}_sarweak_0${i}.txt ../${dataset}/rules/${dataset}_arm.txt > ../${dataset}/rules/${dataset}_narweak_0${i}.txt
    done
done