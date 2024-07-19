# toolbox_energy

This toolbox has been designed for the evaluation of the energy consumption for different architecture types : MLP, CNN, RNN and CRNN for training or testing those models on an audio tagging tasks using the real part of the desed dataset.

Use this command to run the code for training: 
```
python3 main.py --output output_path --epochs 10 --batch_size 8 --train --gpu 0 --dataset 'sc09' --model 'mlp' --num_layers 1 --num_frame 64 --hidden_size 512 --batch_size 8
```
The energy consumption is monitored using trackers from CodeCarbon and Carbontracker.

Summary of all the configurations tested in our experiment. For each number of layers, we tested different hidden sizes. For CRNN, the configurations first indicate the convolutional layers and then the recurrent layers.


| **Model** | **Num Layers**        | **Hidden Sizes**                            |
|-----------|-----------------------|---------------------------------------------|
| **MLP**   | 1                     | 512, 1024, 2048                             |
|           | 4                     | 1024, 2048, 4096                            |
|           | 6, 10, 16, 32         | 4096                                        |
| **CNN**   | 1                     | 128, 256, 512, 1024                         |
|           | 2                     | 128, 256, 384, 512, 768, 1024               |
|           | 6                     | 384, 768                                    |
| **RNN**   | 1                     | 128, 512, 1024, 2048                        |
|           | 4, 6                  | 1024, 2048                                  |
|           | 2, 10, 14             | 2048                                        |
| **CRNN**  | [1,1], [2,1], [1,2]   | [64,64], [256,64], [512, 256]               |
|           | [2,2]                 | [728, 256]                                  |
|           | [1,2], [2,2]          | [1024, 256]                                 |

