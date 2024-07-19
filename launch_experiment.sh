#!/bin/bash

# Give access to the intel powercap energy to monitor cpu consumption
# sudo-g5k chmod 444 '/sys/class/powercap/intel-rapl:0/energy_uj'
# sudo-g5k chmod 444 '/sys/class/powercap/intel-rapl:1/energy_uj'
# sudo-g5k chmod 444 '/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj'
# sudo-g5k chmod 444 '/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj'

source envs/graphique2/bin/activate
cd code/toolbox_energy

# Set the file path for the CSV file
output_path="output/$OAR_JOB_ID"
csv_file="$output_path/nvidia_tracker.csv"

mkdir -p $output_path

# Check if the CSV file exists, if not, create it with headers
if [ ! -e "$csv_file" ]; then
    echo "timestamp,name,power.draw [W],utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB]" > "$csv_file"
fi

# Run the nvidia-smi command and append the output to the CSV file with timestamp
gpu_tracker() {
    while true; do
        # Run the nvidia-smi command and append the output to the CSV file with timestamp
        nvidia-smi --format=csv --query-gpu=timestamp,name,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total | tail -n 1 >> "$csv_file"

        # Sleep for 5 seconds
        sleep 5
    done
}

gpu_tracker &
gpu_tracker_pid=$!


python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --num_layers 1 --num_frame 64 --hidden_size 512 --batch_size 8
# sleep 30 
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30 
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 4 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30 
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 4 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 4 --num_frame 64 --hidden_size 4096 --batch_size 8
# sleep 30 
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 6 --num_frame 64 --hidden_size 4096 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 10 --num_frame 64 --hidden_size 4096 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 16 --num_frame 64 --hidden_size 4096 --batch_size 8
# sleep 30 
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'mlp' --no_time_limit --num_layers 32 --num_frame 64 --hidden_size 4096 --batch_size 8
# sleep 30

# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 128 --batch_size 8 
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 512 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 128 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 384 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 6 --num_frame 64 --hidden_size 384 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 512 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 768 --batch_size 8
# sleep30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 6 --num_frame 64 --hidden_size 768 --batch_size 8
# sleep30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'cnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 1024 --batch_size 8 
# sleep 30

# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 128 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 512 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 4 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 6 --num_frame 64 --hidden_size 1024 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 1 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 2 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 4 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 6 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 10 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'rnn' --no_time_limit --num_layers 14 --num_frame 64 --hidden_size 2048 --batch_size 8
# sleep 30

# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 1 --num_frame 64 --hidden_size 64 64 --batch_size 8 
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 2 1 --num_frame 64 --hidden_size 64 64 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 2 --num_frame 64 --hidden_size 64 64 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 1 --num_frame 64 --hidden_size 256 64 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 1 --num_frame 64 --hidden_size 512 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 2 --num_frame 64 --hidden_size 512 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 1 2 --num_frame 64 --hidden_size 1024 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 2 1 --num_frame 64 --hidden_size 512 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 2 2 --num_frame 64 --hidden_size 728 256 --batch_size 8
# sleep 30
# python3 main.py --output "$output_path" --epochs 10 --train --gpu 0 --dataset 'desed' --model 'crnn' --no_time_limit --num_layers 2 2 --num_frame 64 --hidden_size 1024 256 --batch_size 8


# Terminate the GPU tracker using its PID
kill "$gpu_tracker_pid"


