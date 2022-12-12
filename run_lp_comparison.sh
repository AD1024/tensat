node_limit=50000
iter_limit=15
time_limit=10
ilp_time_sec=10

num_passes=10

models=(
    resnet50
)

cargo build --release
for iter_limit in $(seq 3 $(expr $iter_limit - 1)); do
    for pass in $(seq 0 $(expr $num_passes - 1)); do
            for model in "${models[@]}"; do
                echo "=================="
                echo "$model, CPLEX vs LP"
                echo "=================="
                ./target/release/tensat -r converted.txt -t converted_multi.txt -u -s none --n_iter $iter_limit --no_cycle --n_sec $time_limit --n_nodes $node_limit -e cplex -d $model -o target/"$model"_stats.txt # -x tmp/"$model"_optimized.model
            done
    done
done
