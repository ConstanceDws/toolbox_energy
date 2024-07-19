import os
from collections import OrderedDict
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader


class WeakDataset(Dataset):
    def __init__(self, data_folder, metadata, num_frame):
        self.data_folder = data_folder
        print(self.data_folder)
        self.metadata = metadata
        self.file_list = os.listdir(data_folder)
        self.transform = MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=256,
            f_min=0,
            f_max=22050,
            n_mels=128,
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        )
        self.target_len = 10 * 44100 #10 seconds
        self.num_frame = num_frame
        # sample_rate=44100,
        # self.target_len = target_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        # print(filename)
        file_path = os.path.join(self.data_folder, filename)
        waveform, _ = torchaudio.load(file_path)
        # only one channel
        waveform = self.to_mono(waveform)

        # pad audio
        waveform = self.pad_audio(waveform)

        # if self.transform:
        #     waveform = self.transform(waveform)
        # mel_spec = self.transform(waveform)
        mel_spec = self.transform(waveform)[:, :, :self.num_frame]

        event_label = self.metadata[filename]
        event_label = torch.tensor(event_label, dtype=torch.float32)


        return mel_spec, event_label
    
    # def pad_audio(self, waveform):
    #     if waveform.shape[-1] < self.target_len:
    #         waveform = torch.nn.functional.pad(
    #             waveform, (0, self.target_len - waveform.shape[-1]), mode="constant")

    #     elif len(waveform) > self.target_len:
    #         rand_onset = random.randint(0, len(waveform) - self.target_len)
    #         waveform = waveform[rand_onset:rand_onset + self.target_len]

    #     return waveform
    def pad_audio(self, waveform):
        if waveform.size(-1) < self.target_len:
            # If the waveform is shorter than the target length, pad it
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_len - waveform.size(-1)), mode="constant")
        elif waveform.size(-1) > self.target_len:
            # If the waveform is longer than the target length, randomly crop it
            rand_onset = torch.randint(0, waveform.size(-1) - self.target_len + 1, (1,))
            waveform = waveform[..., rand_onset:rand_onset + self.target_len]
        return waveform

    def to_mono(self, waveform):
        if waveform.shape[0] > 1:
            indx = np.random.randint(0, waveform.shape[0] - 1)
            waveform = waveform[indx]
            waveform = waveform.unsqueeze(0)
        return waveform



def get_dataloader(args):
    # Create a dataset with the specified data folder

    # dataset_path = '/home/cdouwes/dataset/DCASE23_Tutorial/'
    # data_folder = os.path.join(dataset_path, 'DESED_public_eval_sample')
    # metadata_file = os.path.join(dataset_path, 'public_sample.tsv')
    if args.dataset == 'desed':
        dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/corpus/environmental_audio/DESED_real'
        data_folder = os.path.join(dataset_path, 'audio/train/weak')
        metadata_file = os.path.join(dataset_path, 'metadata/train/weak.tsv')
        classes_labels = OrderedDict(
        {
            "Alarm_bell_ringing": 0,
            "Blender": 1,
            "Cat": 2,
            "Dishes": 3,
            "Dog": 4,
            "Electric_shaver_toothbrush": 5,
            "Frying": 6,
            "Running_water": 7,
            "Speech": 8,
            "Vacuum_cleaner": 9,
        }
    )
    elif args.dataset == 'sc09':
        dataset_path = '/srv/storage/talc3@storage4.nancy.grid5000.fr/multispeech/corpus/speech_synthesis/sc09'
        data_folder = os.path.join(dataset_path, 'audio')
        metadata_file = os.path.join(dataset_path, 'metadata/sc09.tsv')
        classes_labels =  OrderedDict(
        {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
        }
        )
    # print(f"data folder: {data_folder}")
    # print(f"data folder: {metadata_file}")

    # Define the label-to-number mapping dictionary

    # Create a data loader
    # fs = 44100
    # target_len = 10 * fs

    # Initialize an empty metadata dictionary
    metadata = {}

    # Read the metadata file (weak.tsv) while skipping the first line (headers)
    with open(metadata_file, 'r') as file:
        next(file)  # Skip the first line (headers)
        for line in file:
            parts = line.strip().split()
            # print(parts)
            filename = parts[0]

            # Extract all event labels from the line
            event_labels = parts[1].split(',')
            # print(event_labels)
            # Map each event label to a number and store in a list
            encoded_event_labels = [classes_labels.get(label, -1) for label in event_labels]
            # print(encoded_event_labels)
            encoded_event_labels = [1 if i in encoded_event_labels else 0 for i in range(10)]
            # print(encoded_event_labels)
            # Add the filename and encoded event labels to the metadata dictionary
            metadata[filename] = encoded_event_labels
    dataset = WeakDataset(data_folder, metadata, args.num_frame)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return data_loader 


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import librosa

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',            type=str,                   default='desed')
    parser.add_argument('--num_frame',          type=int,                   default=64)
    parser.add_argument('--batch_size',         type=int,                   default=8)

    args=parser.parse_args()
    dataloader = get_dataloader(args)
    print('DATALOADER',len(dataloader))

    dummy_melspec, dummy_label = next(iter(dataloader))
    size_melspec = dummy_melspec.size()
    print(dummy_label)
    dummy_melspec_np = dummy_melspec[0].squeeze().detach().cpu().numpy()
    print(dummy_label[0])
    # Plot the mel spectrogram using librosa.display.specshow
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(dummy_melspec_np, ref=np.max), y_axis='mel', x_axis='time')
    
    plt.title('Mel Spectrogram')
    plt.savefig(f'lejolimelspec.jpg')