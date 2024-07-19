import os
import matplotlib.pyplot as plt

import subprocess
import pandas as pd
from datetime import datetime, timedelta
import re
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
from carbontracker import parser, constants



def run_command(command):
    result = subprocess.run(command, capture_output=True, shell=True, text=True)
    output = result.stdout.strip()
    return output

def tracker_bmc(node, start_date, stop_date, metric, output_path):
    header = "timestamp,device,metric_id,value,labels"
    if start_date == stop_date:
        #Record at least 1min 
        stop_date = stop_date[:-1] + str(int(stop_date[-2:-1]+1))
    url = f"https://api.grid5000.fr/stable/sites/nancy/metrics?nodes={node}&metrics={metric}&start_time={start_date}&end_time={stop_date}"
    print(url)
    command = f'{{ echo "{header}"; curl "{url}"| jq -r \'.[] | [.timestamp, .device_id, .metric_id, .value, .labels|tostring] | @csv\'; }} > {output_path}/{metric}.csv'

    subprocess.run(command, shell=True)

def tracker_carbontracker(path, start_times, stop_dates) :
    df_ct = pd.DataFrame()
    df = pd.DataFrame()
    counter = 0
    for root, directories, files in os.walk(path):
        directories.sort()
        if 'carbontracker' in directories and counter < len(start_times):
            logs = parser.parse_all_logs(log_dir=f'{root}/carbontracker')
            first_log = logs[0]
            are_gpu = ('gpu' in first_log['components'])
            are_cpu = ('cpu' in first_log['components'])
            if are_gpu:
                power_gpus = first_log['components']['gpu']['avg_power_usages (W)']
                df_gpu = pd.DataFrame(power_gpus, columns=[f'GPU {i}' for i in range(len(power_gpus[0]))])
                df_gpu['GPU TOT'] = df_gpu.sum(axis=1)
                epoch_durations = first_log['components']['gpu']['epoch_durations (s)']
            if are_cpu:
                power_cpus = first_log['components']['cpu']['avg_power_usages (W)']
                df_cpu = pd.DataFrame(power_cpus, columns=[f'CPU {i}' for i in range(len(power_cpus[0]))])
                df_cpu['CPU TOT'] = df_cpu.sum(axis=1)
                epoch_durations = first_log['components']['cpu']['epoch_durations (s)']
            if are_gpu and are_cpu:
                df = pd.concat([df_gpu, df_cpu], axis=1)
                df['TOT'] = df_gpu['GPU TOT'] + df_cpu['CPU TOT'] 
            elif are_gpu and not are_cpu:
                df = df_gpu
                df['TOT'] = df_gpu['GPU TOT']
            elif not are_gpu and are_cpu:
                df = df_cpu
                df['TOT'] = df_cpu['CPU TOT']
            else:
                raise Exception('No GPU or CPU recorded')

            time_array = []
            current_datetime = start_times[counter]
            for duration in epoch_durations:
                time_array.append(datetime.fromtimestamp(current_datetime).strftime('%Y-%m-%dT%H:%M:%S'))
                current_datetime += duration
            df.insert(0, 'Time', time_array)

            df_ct = pd.concat([df_ct, df], ignore_index=True)
            counter += 1
    df_ct.to_csv(f'{path}/carbontracker_power_watt.csv')

def tracker_loss(path, start_time):
    df_loss = pd.DataFrame()
    for root, directories, files in os.walk(path):
        directories.sort()
        if 'loss.csv' in files:
            df_loss = pd.concat([df_loss, pd.read_csv(f'{root}/loss.csv')], axis = 0)
    df_loss['Time'] = pd.to_datetime(df_loss['Time'], unit='s') + timedelta(hours=1)
    df_loss['Time'] = df_loss['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_loss.to_csv(f'{path}/loss.csv')

def tracker_codecarbon(file_path, start_time, stop_time, output_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'codecarbon INFO' in line and 'Energy consumed' in line : 
                time_match = re.search(r'@ (\d{2}:\d{2}:\d{2})', line)
                time = time_match.group(1) if time_match else None

                gpu_power_match = re.search(r'Total GPU Power : ([\d.]+) W', line)
                gpu_power = float(gpu_power_match.group(1)) if gpu_power_match else None

                ram_power_match = re.search(r'RAM Power : ([\d.]+) W', line)
                ram_power = float(ram_power_match.group(1)) if ram_power_match else None

                cpu_power_match = re.search(r'Total CPU Power : ([\d.]+) W', line)
                cpu_power = float(cpu_power_match.group(1)) if cpu_power_match else None
                data.append({'Time': time, 'GPU Power': gpu_power, 'RAM Power': ram_power, 'CPU Power': cpu_power})
    
    df_cc = pd.DataFrame(data)
    df_cc['Time'] = pd.to_datetime(df_cc['Time'], format='%H:%M:%S')
    
    year = int(datetime.fromtimestamp(start_time).strftime('%Y'))
    month = int(datetime.fromtimestamp(start_time).strftime('%m'))
    day = int(datetime.fromtimestamp(start_time).strftime('%d'))
    df_cc['Time'] = df_cc['Time'].apply(lambda x: x.replace(year, month, day))

    df_cc = df_cc.groupby('Time').agg({'GPU Power': 'sum', 'RAM Power': 'sum', 'CPU Power': 'sum'}).reset_index()
    

    df_cc['Overall Power'] = df_cc['GPU Power'] + df_cc['RAM Power'] + df_cc['CPU Power']
    df_cc.to_csv(f'{output_path}/codecarbon_power_watt.csv')



def compute_energy_consumption(csv_file, start_date, stop_date):
    df_bmc = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
    # df_filtered = df_bmc[(df_bmc.index >= start_date) & (df_bmc.index <= stop_date)]
    # bmc_power_rows = df_bmc[df_bmc['metric_id'] == 'bmc_node_power_watt']
    # Filter rows based on the specified date range
    df_filtered = df_bmc[(df_bmc.index >= start_date) & (df_bmc.index <= stop_date)]

    bmc_power_rows = df_filtered[df_filtered['metric_id'] == 'bmc_node_power_watt']

    power_watt = bmc_power_rows['value'].astype(int)
    power_watt = bmc_power_rows['value'].astype(int)

    time_interval_s = 5  # Each row corresponds to 5s
    power_wh = power_watt * (time_interval_s / 3600)  # Convert to Wh
    total_power_wh = power_wh.sum()

    # Convert total power consumption to kWh
    total_power_kwh = total_power_wh / 1000

    return total_power_kwh

def check_experiment(output_path):
    max_counter = 0
    for directory in os.listdir(output_path):
        if 'run_' in directory :
            experiment_counter = int(directory.split('_')[1])
            if experiment_counter >= max_counter  :
                max_counter = experiment_counter +1
    return max_counter


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def initialize_tracker(args):
    log_dir = os.path.join(args.output_path, 'carbontracker')
    create_directory_if_not_exists(log_dir) 
    tracker_CT = CarbonTracker(epochs = args.epochs, log_dir = log_dir, monitor_epochs=-1, verbose=2, update_interval = args.timer, devices_by_pid = (args.mode == 'process'))

    output_dir = os.path.join(args.output_path, 'codecarbon')
    create_directory_if_not_exists(output_dir)
    # tracker_CC = EmissionsTracker(output_dir = output_dir, measure_power_secs = args.timer, tracking_mode = args.mode)
    tracker_CC = EmissionsTracker(output_dir = output_dir, measure_power_secs = args.timer, tracking_mode = args.mode, gpu_ids=[0])

    return tracker_CT, tracker_CC

def plot_power_time(df_bmc, df_cc, output_path):
    """
    Plot BMC Node Power over time.

    Args:
    - args: Command-line arguments or configuration parameters.
    - csv_file (str): Path to the CSV file containing BMC node power data.
    """
    # df_bmc = pd.read_csv(f'{output_path}/bmcnodepower/bmc_node_power_watt.csv')
    # df_bmc['timestamp'] = pd.to_datetime(df_bmc['timestamp'])
    # df_bmc['timestamp'] = df_bmc['timestamp'].dt.tz_localize(None)
    # df_cc = pd.read_csv(f'{output_path}/codecarbon/codecarbon_power_watt.csv')
    # df_cc['Time'] = pd.to_datetime(df_cc['Time'])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_bmc['timestamp'], df_bmc['value'], label = 'BMC Power')
    plt.plot(df_cc['Time'], df_cc['Overall Power'], label='Overall Power Consumption')
    plt.plot(df_cc['Time'], df_cc['GPU Power'], label='GPU Power')
    plt.plot(df_cc['Time'], df_cc['RAM Power'], label='RAM Power')
    plt.plot(df_cc['Time'], df_cc['CPU Power'], label='CPU Power')
    plt.title('BMC Power')
    plt.xlabel('Time')
    plt.legend()
    plt.ylabel('Power (Watt)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.xticks(rotation=45)
    # Save the plot
    plt.savefig(f'{output_path}/energy_all.pdf', transparent=True)

    # plt.figure(figsize=(10, 6))

    # plt.title('CodeCarbon Power')
    # plt.xlabel('Time')
    # plt.ylabel('Power (Watt)')
    # plt.xticks(rotation=45)
    # plt.ylim(bottom=0)
    # plt.legend()
    # plt.savefig(f'{output_path}/energy_codecarbon.pdf', transparent=True)



def get_energy_from_carbontracker(output_path):
    logs = parser.parse_all_logs(log_dir=f'{output_path}/carbontracker')
    first_log = logs[0]

    energy_CT = first_log['actual']['energy (kWh)'] / constants.PUE_2022 

    return {'TOT' : energy_CT}

def get_energy_from_codecarbon(output_path):
    df_cc = pd.read_csv(f'{output_path}/codecarbon/emissions.csv')
    energy_cc_gpu = df_cc['gpu_energy'].iloc[-1]
    energy_cc_mem = df_cc['ram_energy'].iloc[-1]
    energy_cc_cpu = df_cc['cpu_energy'].iloc[-1]
    energy_cc = df_cc['energy_consumed'].iloc[-1]

    return {
        'GPU': energy_cc_gpu,
        'CPU': energy_cc_cpu,
        'MEM': energy_cc_mem,
        'TOT': energy_cc
    }

def get_duration_from_codecarbon(output_path):
    df_cc = pd.read_csv(f'{output_path}/codecarbon/emissions.csv')
    return df_cc['duration'].iloc[-1]

def get_energy_from_pyjoules(output_path):
    df_pj = pd.read_csv(f'{output_path}/pyjoules/result.csv', sep=';')
    df_pj['total_energy_nvidia'] = df_pj.filter(like='nvidia_gpu').sum(axis=1)
    df_pj['total_energy_package'] = df_pj.filter(like='package').sum(axis=1)
    df_pj['total_energy_dram'] = df_pj.filter(like='dram').sum(axis=1)

    energy_pj_gpu = df_pj['total_energy_nvidia'].iloc[-1] / 3.6e+9
    energy_pj_cpu = df_pj['total_energy_package'].iloc[-1] / 3.6e+12
    energy_pj_mem = df_pj['total_energy_dram'].iloc[-1] / 3.6e+12

    return {
        'GPU': energy_pj_gpu,
        'CPU': energy_pj_cpu,
        'MEM': energy_pj_mem,
        'TOT': energy_pj_gpu + energy_pj_cpu + energy_pj_mem
    }

def get_energy_from_bmcnodepower(output_path, start_date, stop_date):
    energy_BMC = compute_energy_consumption(f'{output_path}/bmc_node_power_watt.csv', start_date, stop_date)

    return {'TOT' : energy_BMC}

def get_mean_from_df(df, start_date, stop_date, column1, column2):
    df_filtered = df[(df[column1] >= start_date) & (df[column1] <= stop_date)]
    # print(df_filtered)
    mean = df_filtered[column2].mean()
    return mean 

def get_power_mean_from_codecarbon(output_path):
    df_cc = pd.read_csv(f'{output_path}/codecarbon/emissions.csv')
    energy_cc = df_cc['energy_consumed'].iloc[-1]
    duration = df_cc['duration'].iloc[-1]
    power_mean_W = energy_cc / (duration / 3600) * 1000
    return power_mean_W
# def plot_histogram(output_path, trackers):
#     fig, ax = plt.subplots()
#     bottom = np.zeros(len(trackers))
#     color_dict = {'GPU': '',
#                   'CPU': '',
#                   'MEM' : '',
#                   'TOT' : ''}
    
#     tracker_keys = list(trackers.keys())
    
#     for i, key in enumerate(tracker_keys):
#         tracker = trackers[key][0]  # Access the first (and only) element of the list
#         for component, energy in tracker.items():
#             if component != 'TOT' and energy != 0:
#                 hist = ax.bar(i, energy, bottom=bottom[i], color=color_dict[component], edgecolor='black', linewidth=0.5)
#                 bottom[i] += energy
#                 ax.text(i, bottom[i] - energy/2, component, ha='center', va='center', color='black', fontsize=8)            
#             elif component == 'TOT':
#                 hist = ax.bar(i, energy, color=color_dict['TOT'], edgecolor='black', linewidth=1)

#     ax.set_xticks(range(len(tracker_keys)))
#     ax.set_xticklabels(tracker_keys)
#     ax.set_title('Energy consumption comparisons')
#     plt.savefig(f'{output_path}/energy_comparisons.pdf', transparent=True)


# def plot_energy_from_csv(output_path):
#     energy_CT = get_energy_from_carbontracker(output_path) 
#     energy_CC = get_energy_from_codecarbon(output_path)
#     energy_PJ = get_energy_from_pyjoules(output_path)
#     energy_BMC = get_energy_from_bmcnodepower(output_path)
    
#     trackers = {'Codecarbon' : [energy_CC], 
#                 'PyJoules' : [energy_PJ],
#                 'CarbonTracker' : [energy_CT],
#                 'BMC' : [energy_BMC]}
    
#     plot_histogram(output_path, trackers)
#     plot_power_time(output_path)

#     return energy_CT, energy_CC, energy_PJ, energy_BMC