run_for_an_hour() {
    local start_time=$SECONDS
    local duration=100  # 3600 seconds = 1 hour

    while [ $((SECONDS - start_time)) -lt $duration ]; do
        ./ds_pretrain_gpt_1.3B_dense_base_script.sh
    done
}

# Main loop
while true; do
    echo "Starting 1 hour of work..."
    ./ds_pretrain_gpt_cycle.sh
    echo "Work phase complete, starting 2 hours rest..."
    sleep 6000  # 7200 seconds = 2 hours
done
