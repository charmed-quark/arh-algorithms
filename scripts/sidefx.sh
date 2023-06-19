for dataset in chess mushrooms bms1 syn1K syn10K; do
    for algorithm in abc4arh samdp vidpso; do
        echo "Computing side effects for ${dataset}/${algorithm}..."
        python3 /home/charlie/thesis/scripts/computesidefx.py ${dataset} ${algorithm} > /home/charlie/thesis/results/${dataset}_${algorithm}_sidefx.txt
    done
done