#!/bin/bash

### EXPERIMENT configurations
sweep_ids=('wandb agent ais2t/reconstruction_survey/myslsuke') # Sweep IDs
job_name="reconstruction" #"reconstruction" # Job name for tracking

### SLURM configurations
max_time="0-06" # Maximum time in days-hours format
partitions=("gcp1-gpu-h200")
node_subset="" # Node subset
exclude_nodes="htc-gpu[001-004,010-019,024-038]" #"gpu906"
n_gpus=1 # Number of GPUs per job
n_cpus=16 # Number of CPUs per job
submit_all_at_once=false # Submit all jobs at once
num_experiments=8 # Number of experiments per sweep, adjust as needed
max_concurrent_runs=8 # Maximum number of concurrent running jobs
sleep_between_runs=0
mem=200G # Memory per job

### CODE
partition_list=$(IFS=,; echo "${partitions[*]}")
codebase_path="/home/htc/$USER/Data/reconstruction_survey"

# Iterate over the array and remove the "wandb agent " prefix if present
for i in "${!sweep_ids[@]}"; do
    sweep_ids[$i]=${sweep_ids[$i]#wandb agent }
done

# Function to check current running jobs
check_running_jobs() {
    current_running=$(squeue -u $USER -t RUNNING,PENDING,CONFIGURING -p $partition_list -o "\"%j\"" | grep $job_name | wc -l)
    if [[ $current_running -lt $max_concurrent_runs ]]; then
        return 0 # Okay to submit
    else
        return 1 # Wait needed
    fi
}

# Submit jobs
for sweep_id in "${sweep_ids[@]}"; do
    echo "Starting sweep $sweep_id."
    for (( i=1; i<=$num_experiments; i++ )); do
        if [[ $submit_all_at_once == false ]]; then
            while ! check_running_jobs; do
                sleep 5
            done
        fi
                # Determine if --nodelist should be included
        if [[ -n $node_subset ]]; then
            sbatch -p $partition_list --gres=gpu:$n_gpus --mem=$mem --cpus-per-task=$n_cpus --comment="${sweep_id}"\
                   --time=$max_time --nodelist=$node_subset --job-name=$job_name --exclude=$exclude_nodes\
                   runnerb200.sh $codebase_path $sweep_id
        else
            sbatch -p $partition_list --gres=gpu:$n_gpus --mem=$mem --cpus-per-task=$n_cpus --comment="${sweep_id}"\
                   --time=$max_time --job-name=$job_name --exclude=$exclude_nodes\
                   runnerb200.sh $codebase_path $sweep_id
        fi
        sleep $sleep_between_runs
    done
done
