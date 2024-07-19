# External imports
import os
import subprocess
import time
import argparse
import torch

# Internal imports
from src.utils import *
from src.dataloader import get_dataloader
from src.model import *

from thop import profile, clever_format
from deepspeed.profiling.flops_profiler import get_model_profile

parser = argparse.ArgumentParser()

parser.add_argument('--train',              action="store_true",        default=False)
parser.add_argument('--test',               action="store_true",        default=False)
parser.add_argument('--run',                action="store_true",        default=False)
parser.add_argument('--no_time_limit',      action="store_true",        default=False)
parser.add_argument('--output',             type=str,                   default='output')
parser.add_argument('--epochs',             type=int,                   default=100)
parser.add_argument('--batch_size',         type=int,                   default=8)
parser.add_argument('--gpu',                type=int,                   default=0)
parser.add_argument('--timer',              type=int,                   default=5)
parser.add_argument('--mode',               type=str,                   default='machine', help='machine or process')
parser.add_argument('--model',              type=str,                   default='mlp')
parser.add_argument('--num_frame',          type=int,                   default=1723)
parser.add_argument('--num_layers',         type=int,nargs="+",         default=1)
parser.add_argument('--hidden_size',        type=int,nargs="+",         default=500)
parser.add_argument('--dataset',            type=str,                   default='desed')

args=parser.parse_args()

args.node = subprocess.check_output("uniq $OAR_NODEFILE", shell=True, text=True).split('.')[0]
args.jobId = subprocess.check_output("echo $OAR_JOB_ID", shell=True, text=True).strip()
args.num_experiment = check_experiment(args.output)
args.output_path = f'{args.output}/run_{args.num_experiment}'
os.makedirs(args.output_path)

# # Specify device
args.device = torch.device(f'cuda:{args.gpu}')

df_loss = pd.DataFrame(columns= ['Epoch', 'Time', 'Loss'])


dataloader = get_dataloader(args)
print('DATALOADER',len(dataloader))

# dummy_melspec, dummy_label = next(iter(dataloader))
# size_melspec = dummy_melspec.size()

# dummy_melspec_np = dummy_melspec[0].squeeze().detach().cpu().numpy()

# # Plot the mel spectrogram using librosa.display.specshow
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(dummy_melspec_np, ref=np.max), y_axis='mel', x_axis='time')
# # plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.savefig(f'{args.output_path}/lejolimelspec.jpg')

if len(args.num_layers) == 1 and len(args.hidden_size) == 1 :
    args.num_layers = args.num_layers[0]
    args.hidden_size = args.hidden_size[0]


if args.model == 'mlp':
    model = DynamicMLP(input_size= 128*args.num_frame, hidden_size=args.hidden_size, output_size=10, num_layers=args.num_layers)

elif args.model == 'cnn':
    model = DynamicCNN(input_channels = 1, output_classes= 10, hidden_size=args.hidden_size, num_layers=args.num_layers, num_frame=args.num_frame)

elif args.model == 'rnn':
    model = DynamicRNN(input_size = 128*args.num_frame, output_size=10, hidden_size=args.hidden_size, num_layers=args.num_layers)

elif args.model == 'crnn':
    model = DynamicCRNN(input_channels = 1, output_classes=10, hidden_size=args.hidden_size, num_layers=args.num_layers)



print(model)

model.to(args.device)
# criterion = torch.nn.CrossEntropyLoss() #for single class 
criterion = nn.BCELoss() #for multiclass
# criterion = torch.nn.MSELoss(reduction='mean')
# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


dummy_melspec, dummy_label = next(iter(dataloader))

shape = (1, 1, 128, args.num_frame)
print(shape)
args.flops, args.macs, args.params = get_model_profile(model=model, input_shape=(shape), output_file= f'{args.output_path}/flops_params.txt', module_depth=-1, as_string=False)
print(f'FLOPs:{args.flops} MACs: {args.macs} PARAMs:{args.params}')

tracker_CT, tracker_CC = initialize_tracker(args)

time_limit = 20 * 60 #20 minutes

def train(args, start_time):
    iter = 0
    for epoch in range(args.epochs):
        tracker_CT.epoch_start() 
        if args.train:
            model.train()
            running_loss = 0.0
            for (inputs, labels) in dataloader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if iter % 100 == 0:
                    df_loss.loc[len(df_loss)] = [iter, time.time(), running_loss/len(dataloader)]
                iter+=1
                # Check if the time limit has been reached
                if args.no_time_limit == False and time.time() - start_time > time_limit:
                    print("Time limit reached. Training terminated.")
                    return
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(dataloader)}')
                        
        else :
            time.sleep(10)

        tracker_CT.epoch_end()

def test(args, start_time):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during testing
        tracker_CT.epoch_start() 
        for (inputs, labels) in dataloader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            _, digit_label = torch.max(labels, 1)

            # print(labels, digit_label)

            total += labels.size(0)
            correct += (predicted == digit_label).sum().item()
            if time.time() - start_time > time_limit:
                print("Time limit reached. Testing terminated.")
                return
        tracker_CT.epoch_end()
    accuracy = correct / total
    average_loss = test_loss / len(dataloader)

    print(f'Test Loss: {average_loss}, Accuracy: {accuracy}')

def run(args):
    inputs = torch.rand(1, 1, 128, args.num_frame)
    model.eval()
    with torch.no_grad():
        tracker_CT.epoch_start() 
        for i in range (60):
            outputs = model(inputs)
        tracker_CT.epoch_end()

#Start time experiments
print('Start experiments')
start_time = time.time()
tracker_CC.start()

if args.train : 
    train(args, start_time)
if args.test : 
    test(args, start_time)
if args.run:
    run(args)

#Stop time expériments
tracker_CT.stop()
tracker_CC.stop()
stop_time = time.time()
print('Stop experiments')

#Save metadata
with open(f'{args.output}/metadata.txt', 'a') as f:
    f.write(f'{args.node};{args.jobId};run_{args.num_experiment};{start_time};{stop_time};{args.model};{args.flops};{args.macs};{args.params};{args.num_frame};{args.num_layers};{args.hidden_size};{args.batch_size}\n')
    print('Metadata saved')



#Save Loss
csv_file_path = f'{args.output_path}/loss.csv'
df_loss.to_csv(csv_file_path, index=False)
print('Loss csv saved')
