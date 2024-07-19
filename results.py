import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import matplotlib.dates as md
from matplotlib.gridspec import GridSpec
from openpyxl import Workbook

from src.utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--output',             type=str,                   default='output')

args=parser.parse_args()



def get_metadata(path):
    with open(f'{path}/metadata.txt', 'r') as file:
        lines = file.readlines()
    runs = []
    start_times = []
    stop_times = []
    models =  []
    num_layers = []
    batch_sizes = []
    hidden_sizes = []
    flops = []
    macs = []
    params = []
    num_frames = []

    for line in lines:
        node, jobId, run, start_time, stop_time, model, flop, mac, param,num_frame, num_layer, hidden_size, batch_size = line.strip().split(';')
        start_times.append(float(start_time))
        stop_times.append(float(stop_time))
        runs.append(run)
        models.append(model)
        flops.append(flop)
        macs.append(mac)
        params.append(param)
        num_layers.append(num_layer)
        hidden_sizes.append(hidden_size)
        batch_sizes.append(batch_size)
        num_frames.append(num_frame)
        
    return node, jobId, runs, start_times, stop_times, models, flops, macs, params, num_frames, num_layers, hidden_sizes, batch_sizes

def plot_times(ax, start_dates, stop_dates, models, batch_sizes):
    # Convert start_dates and stop_dates to datetime objects
    start_dates = [datetime.fromisoformat(date) for date in start_dates]
    stop_dates = [datetime.fromisoformat(date) for date in stop_dates]

    for i, model in enumerate(models):
        # print(start_dates[i])
        ax.axvline(start_dates[i], color='green', linestyle='--', alpha = 0.5)
        ax.axvline(stop_dates[i], color='red', linestyle='--', alpha = 0.5)

        # Add text at the top of the lines
        # ax.text(start_dates[i], ax.get_ylim()[1], f'{model} (Batch {batch_sizes[i]})', va='bottom')

    # Format x-axis as dates
    # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":

    color_dict = {'CC': {'GPU' : 'firebrick',
                                'CPU' : 'orange',
                                'MEM' : 'indigo',
                                'TOT' : 'green'},
                'PJ' : {'GPU' : 'blue',
                                'CPU' : 'red',
                                'MEM' : 'purple',
                                'TOT' : 'green'},
                'CT' : {'GPU' : 'blue',
                                'CPU' : 'red',
                                'TOT' : 'darkblue'},
                'BMC' : {'TOT' : 'black'}}


    constant_mode_dict = {  'graffiti' : {'name':'Intel Xeon Silver 4110',
                                        'TDP' : 85,
                                        'RAM': 128 },
                            'grele' : {'name': 'Intel Xeon E5-2650 v2,95',
                                'TDP' : 105,
                                'RAM' : 128},
                            'grue': {'name':'AMD EPYC 7351',
                                    'TDP': 170,
                                    'RAM':  128},
                            'gruss': {'name':'AMD EPYC 7352',
                                    'TDP': 155,
                                    'RAM': 256}}


    wb = Workbook()
    ws = wb.active

    # Add headers
    ws.append(['Node', 'Model', 'Layers', 'Hidden', 'FLOPs', 'MACs', 'Param', 'Duration Nvidia', 'Duration CC', 'Power Nvidia','Power CodeCarbon (GPU)', 'Energy Nvidia', 'Energy BMC', 'Energy CodeCarbon (GPU)', 'Energy CodeCarbon (TOT)',  'Energy CodeCarbon (CPU)', 'Energy CodeCarbon (RAM)', 'Mean GPU Use', 'Mean GPU Mem'])

    runçdict = {}
    for root, directories, files in os.walk(args.output):
        if 'metadata.txt' in files :
            directories.sort()
            output_path = root
            print(root)
            node, jobId, runs, start_times, stop_times, models, flops, macs, params, num_frames, num_layers, hidden_sizes,batch_sizes = get_metadata(output_path)
            
            metric = 'bmc_node_power_watt'
            node_name = node.split('-')[0]
            
            start_dates = [datetime.fromtimestamp(start_times[i]).strftime('%Y-%m-%dT%H:%M:%S') for i in range(len(start_times))]
            stop_dates = [datetime.fromtimestamp(stop_times[i]).strftime('%Y-%m-%dT%H:%M:%S') for i in range(len(stop_times))]
            
            if not os.path.exists(f'{output_path}/bmc_node_power_watt.csv'):
                tracker_bmc(node, start_dates[0], stop_dates[-1], metric, output_path = f'{output_path}/')
            if not os.path.exists(f'{output_path}/codecarbon_power_watt.csv'):
                tracker_codecarbon(f'/home/cdouwes/OAR.{jobId}.stderr',start_times[0], stop_times[-1], f'{output_path}/')
            if not os.path.exists(f'{output_path}/carbontracker_power_watt.csv'):
                tracker_carbontracker(output_path, start_times, stop_times)
            if not os.path.exists(f'{output_path}/loss.csv'):
                tracker_loss(f'{output_path}', start_times[0])


            df_nvidia_tracker = pd.read_csv(f'{output_path}/nvidia_tracker.csv')
            df_nvidia_tracker = df_nvidia_tracker.iloc[::4]
            df_nvidia_tracker['timestamp'] = pd.to_datetime(df_nvidia_tracker['timestamp'])
            df_nvidia_tracker['utilization.gpu [%]'] = df_nvidia_tracker['utilization.gpu [%]'].str.extract('(\d+)').astype(float)
            df_nvidia_tracker['utilization.memory [%]'] = df_nvidia_tracker['utilization.memory [%]'].str.extract('(\d+)').astype(float)
            df_nvidia_tracker['power.draw [W]'] = df_nvidia_tracker['power.draw [W]'].str.extract('(\d+.\d+)').astype(float)
            df_nvidia_tracker['memory.used [MiB]'] = df_nvidia_tracker['memory.used [MiB]'].str.extract('(\d+)').astype(float)
            df_nvidia_tracker['memory.total [MiB]'] = df_nvidia_tracker['memory.total [MiB]'].str.extract('(\d+)').astype(float)


            df_bmc = pd.read_csv(f'{output_path}/bmc_node_power_watt.csv')
            df_bmc['timestamp'] = pd.to_datetime(df_bmc['timestamp'])
            df_bmc['timestamp'] = df_bmc['timestamp'].dt.tz_localize(None)

            df_cc = pd.read_csv(f'{output_path}/codecarbon_power_watt.csv')
            df_cc['Time'] = pd.to_datetime(df_cc['Time'])
            

            # df_ct = pd.read_csv(f'{output_path}/carbontracker_power_watt.csv')
            # df_ct['Time'] = pd.to_datetime(df_ct['Time'])

            df_loss = pd.read_csv(f'{output_path}/loss.csv')
            df_loss['Time'] = pd.to_datetime(df_loss['Time'])
            df_loss.sort_values(by='Time')

            df_estim = pd.DataFrame()
            tdp = constant_mode_dict[node_name]['TDP']
            ram = constant_mode_dict[node_name]['RAM']
            df_estim['Time'] = df_nvidia_tracker['timestamp']
            df_estim['CPU'] = tdp*0.5  # Assign the TDP value to the 'CPU ESTIM' column
            df_estim['RAM'] = ram/8 *3
            df_estim['TOT'] = df_estim['CPU'] + df_estim['RAM'] + df_nvidia_tracker['power.draw [W]']

            
            # fig = plt.figure(figsize=(15, 15))
            fig = plt.figure(figsize=(20, 10))
            gs = GridSpec(nrows=3, ncols=len(start_times))
            fig.suptitle(f'Energy consumption comparisons on {node}')

            counter_exp = 0
            max_power = 0
            energy_dict = {}
                    
            for counter_exp in range(len(start_times)):
                # print(counter_exp, runs[counter_exp])
                root_sub = os.path.join(root,runs[counter_exp])

                # print(node, models[counter_exp], flops[counter_exp])
                
                # energy_CT = get_energy_from_carbontracker(root)
                energy_CC = get_energy_from_codecarbon(root_sub) 
                duration_CC = get_duration_from_codecarbon(root_sub)

                energy_BMC = get_energy_from_bmcnodepower(output_path, start_dates[counter_exp], stop_dates[counter_exp]) 
                duration_BMC = stop_times[counter_exp] - start_times[counter_exp]
                
                power_mean_CC_GPU = energy_CC['GPU'] / (duration_CC/3600)*1000
                power_mean_CC = energy_CC['TOT'] / (duration_CC/3600)*1000
                power_mean_BMC = energy_BMC['TOT'] / (duration_BMC/3600)*1000
                # get_mean_from_df(df_cc, start_dates[counter_exp], stop_dates[counter_exp], 'Time', 'Overall Power')
                # mean_BMC = get_mean_from_df(df_bmc, start_dates[counter_exp], stop_dates[counter_exp], 'timestamp', 'value')
                power_mean_NvidiaGPU = get_mean_from_df(df_nvidia_tracker, start_dates[counter_exp], stop_dates[counter_exp], 'timestamp', 'power.draw [W]')
                energy_NvidiaGPU = power_mean_NvidiaGPU * (duration_BMC/3600)/1000
                # print(power_mean_NvidiaGPU)
                gpu_use_mean = get_mean_from_df(df_nvidia_tracker, start_dates[counter_exp], stop_dates[counter_exp], 'timestamp', 'utilization.gpu [%]')
                mem_use_mean = get_mean_from_df(df_nvidia_tracker, start_dates[counter_exp], stop_dates[counter_exp], 'timestamp', 'utilization.memory [%]')
                
                ws.append([node, models[counter_exp], num_layers[counter_exp], hidden_sizes[counter_exp], int(flops[counter_exp]), int(macs[counter_exp]), int(params[counter_exp]), duration_BMC, duration_CC, power_mean_NvidiaGPU, power_mean_CC_GPU, energy_NvidiaGPU, energy_BMC['TOT'], energy_CC['GPU'], energy_CC['TOT'], energy_CC['CPU'],energy_CC['MEM'], gpu_use_mean, mem_use_mean])

                # energy_dict[f'Run_{counter_exp}'] = {'CC': energy_CC,'CT': energy_CT,'BMC': energy_BMC}
                energy_dict[f'Run_{counter_exp}'] = {'CC': energy_CC,'BMC': energy_BMC}
                for key, energy in energy_dict[f'Run_{counter_exp}'].items():
                    if max_power < energy['TOT']:
                        max_power = energy['TOT']
 


            ax0 = fig.add_subplot(gs[0, :])
            ax0.plot(df_bmc['timestamp'], df_bmc['value'], label = 'BMC', color = color_dict['BMC']['TOT'], linewidth=2)
            # ax0.plot(df_bmc['timestamp'], (df_bmc['value'] - df_estim['CPU'] - df_estim['RAM'])/4, label = '(BMC-CPU-RAM)/4', color = color_dict['BMC']['TOT'], linewidth=2, linestyle='-')
            ax0.plot(df_cc['Time'], df_cc['Overall Power'], label='CodeCarbon', color = color_dict['CC']['TOT'], linewidth=2)
            ax0.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['power.draw [W]'], label='Nvidia GPU', linewidth=2)
            # ax0.plot(df_ct['Time'], df_ct['TOT'], label='CarbonTracker', linewidth=2, color = color_dict['CT']['TOT'])
            # ax0.plot(df_nvidia['Time'],df_nvidia['GPU Power (W)'], label = 'Nvidia (GPU 0)', linewidth=2)
            # ax0.plot(df_estim['Time'], df_estim['TOT'] , label = 'Total Estim', linewidth=2 )
            # ax0.plot(df_estim['Time'], df_estim['CPU'], label= 'CPU Estim', alpha = 0.5)
            # ax0.plot(df_estim['Time'], df_estim['RAM'], label= 'RAM Estim', alpha = 0.5)
            #plot some estimations
            # plt.axhline(y=tdp, color='r', linestyle='-', linewidth=2)
            # ax0.plot(df_ct['Time'], df_ct['TOT'], label='CarbonTracker', color = color_dict['CT']['TOT'], linewidth=2)
            # plot_times(ax0,start_dates,stop_dates,models,batch_sizes)
            # ax0.set_title(f'Combined Power Consumption on {node}')
            ax0.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            ax0.legend()
            ax0.set(xlabel='Time', ylabel='Power (Watt)')
            ax0.grid(True)
            ax0.set_ylim(bottom=0)

            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['utilization.gpu [%]'], label = 'GPU Use',linewidth=2, color = 'orange', alpha = 0.8)
            ax3.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['utilization.memory [%]'], label = 'GPU Memory',linewidth=2, color = 'purple', alpha = 0.8)
            # ax3.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['memory.used [MiB]']/df_nvidia_tracker['memory.total [MiB]']*100, label = 'Assigned Memory %', color = 'purple' ,alpha = 0.5, linewidth=2)
            ax3.grid(True)
            ax3.set_title(f'Nvidia-smi queries')
            ax3.set_ylabel('%')
            ax3.set_ylim([0,100])
            ax3.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            # plot_times(ax3,start_dates,stop_dates,models,batch_sizes)
            ax3.legend()

            # ax2 = fig.add_subplot(gs[3, :])
            # ax2.plot(df_cc['Time'], df_cc['Overall Power'], label='TOT', color = color_dict['CC']['TOT'], linewidth=2)
            # ax2.plot(df_cc['Time'], df_cc['GPU Power'], label='GPU', color = color_dict['CC']['GPU'], linewidth=2)
            # ax2.plot(df_cc['Time'], df_cc['RAM Power'], label='RAM', color = color_dict['CC']['MEM'], linewidth=2)
            # ax2.plot(df_cc['Time'], df_cc['CPU Power'], label='CPU', color = color_dict['CC']['CPU'], linewidth=2)
            # plot_times(ax2,start_dates,stop_dates,models,batch_sizes)
            # ax2.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            # ax2.legend()
            # ax2.set_ylabel('Power')
            # ax2.set_ylim(bottom=0)
            # ax2.set_title(f'CodeCarbon detailed')

            # ax3b = fig.add_subplot(gs[3, :])
            # ax3b.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['memory.used [MiB]'], label = 'Memory used',linewidth=2, color = 'red')
            # ax3b.plot(df_nvidia_tracker['timestamp'],df_nvidia_tracker['memory.total [MiB]'], label = 'Memory total',color = 'purple', linewidth=2)
            # ax3b.set_ylabel('MiB')
            # ax3b.set_ylim(bottom=0)
            # ax3.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            # plot_times(ax3b,start_dates,stop_dates,models,batch_sizes)
            # ax3b.legend()
            # ax3.plot(df_nvidia['Time'],df_nvidia[df_nvidia['GPU Utilization'] != 0], label = 'GPU Use', linewidth=2)
            # ax3.plot(df_nvidia['Time'],df_nvidia['GPU Memory'], label = 'GPU Memory', linewidth=2)
            
            

            
            # ax5 = fig.add_subplot(gs[5, :])
            # ax5.set_ylabel('GPU RAM Used (MB)')
            # ax5.plot(df_nvidia['Time'],df_nvidia['GPU RAM Used (MB)'], label = 'RAM Used', linewidth=2)
            # ax5.plot(df_nvidia['Time'],df_nvidia['GPU RAM Used (MB)'], label = 'RAM Used', linewidth=2)
            # ax5.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

            # ax3_2.plot(df_nvidia['Time'],df_nvidia['GPU Memory'], label = 'GPU Memory', color = color)
            # ax3_2.set_ylim([0,100])
            # ax4.tick_params(axis='y', labelcolor=color)


            # ax5 = fig.add_subplot(gs[4, :])
            # columns_to_plot = df_ct.columns[2:]  # Exclude the 'Time' column
            # for column in columns_to_plot:
            #     if column == 'TOT':
            #         ax5.plot(df_ct['Time'], df_ct[column], label=column, linewidth=2, color = color_dict['CT']['TOT'])
            #     else : 
            #         ax5.plot(df_ct['Time'], df_ct[column], label=column, linewidth=2)
            # plot_times(ax5,start_dates,stop_dates,models,batch_sizes)
            # ax5.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            # ax5.legend()
            # ax5.set_title('Carbontracker detailed')

            ax4 = fig.add_subplot(gs[2, :])  # instantiate a second axes that shares the same x-axis
            ax4.set_ylabel('Loss')
            # ax4.set_ylabel('GPU Loss', color=color)  # we already handled the x-label with ax1
            ax4.plot(df_loss['Time'],df_loss['Loss'], label = 'Loss', linewidth=2)
            plot_times(ax4,start_dates,stop_dates,models,batch_sizes)
            ax4.set_title("Loss")
            ax4.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))


                
            # ax4 = fig.add_subplot(gs[4, :])
            # ax4.plot(df_cc['Time'], df_cc['GPU Power'], label='GPU CodeCarbon', color = color_dict['CC']['GPU'], linewidth=2)
            # ax4.plot(df_ct['Time'], df_ct['GPU TOT'], label='GPU CarbonTracker', linewidt
            # h=2)
            # if 'CPU TOT' in df_ct.keys():
            #     ax4.plot(df_cc['Time'], df_cc['CPU Power'], label='CPU CodeCarbon', color = color_dict['CC']['CPU'], linewidth=2)
            #     ax4.plot(df_ct['Time'], df_ct['CPU TOT'], label='CPU CarbonTracker', linewidth=2)
            # if node_name in constant_mode_dict.keys():
            
            #     ax4.plot(df_estim_cpu['Time'], df_estim_cpu['CPU ESTIM'], linewidth=2, alpha=0.5, label='CPU Estimate from TDP')
            # ax4.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
            # ax4.legend()
            # ax4.set_title(f'CarbonTracker and Codecarbon comparisons on {node}')
            plt.tight_layout()
            plt.savefig(f'{output_path}/energy_all.pdf', transparent=True)
            # plt.show()
    wb.save("energy_comparisons_desed10ep_same.xlsx")