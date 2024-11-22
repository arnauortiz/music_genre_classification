import torch.nn as nn
import torch.nn.functional as F
import torch

# Conventional and convolutional neural network

    
class CNNGH1D(nn.Module):
        """
        Convolutional Neural Network of 1 dimension, on the time dimension. It has 3 CNN layers with LeakyReLU activations and kernel of 4, and with
        Max Pooling between them of 4,4 and 2 respectively. After the convolutions it has 1 FC layer.
        """
        def __init__(self):
            super(CNNGH1D, self).__init__()
            self.name="CNNGH1D"
            self.layer1 = nn.Sequential(
                nn.Conv1d(in_channels=128,out_channels=128, kernel_size=4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer2 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(4)
            )
            self.layer3 = nn.Sequential(
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.25),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2)
            )

            self.fc1 = nn.Linear(in_features=4864 ,out_features=8)
            self.dropout = nn.Dropout(0.3)
            
                
        def forward(self, x):
            #x = x.reshape(100,128,1291)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.dropout(out)
            return out, F.softmax(out, dim=1)
        

class ResBlock2d(nn.Module):
    """
    Residual Convolutional Neural Networks, two CNN layers that are added during the activation function (ReLU).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels)
        )
    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.layer1(x) + self.layer2(x))


class ConvBlock2d(nn.Module):
  """
  Block of 4 Residual CNN, with channel (input,output) of (1,16), (16,32), (32,64) and (64,128) respectively, with Max Pooling between
  them with kernel (8,2), (4,2) and (4,2), used when training with Spectrograms
  """
  def __init__(self):
    super().__init__()
    self.block = nn.Sequential(
      ResBlock2d(1, 16),
      nn.MaxPool2d(kernel_size=[8, 2]),
      ResBlock2d(16, 32),
      nn.MaxPool2d(kernel_size=[4, 2]),
      ResBlock2d(32, 64),
      nn.MaxPool2d(kernel_size=[4, 2]),
      ResBlock2d(64, 128)
    )

  def forward(self, x):
    x = self.block(x)
    return x

class ConvBlock2dMFCCs(nn.Module): #per mfccs, funciona pel chroma tmb
    """
    Block of 4 Residual CNN, with channels (input, output) of (1, 16), (16, 32), (32, 64), and (64, 128) respectively,
    with Max Pooling between them with kernel sizes and strides adjusted to maintain dimensions, used when training with Chroma Frequencies or MFCC
    """
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock2d(1, 16),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(16, 32),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(32, 64),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(2, 2), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(64, 128)
        )

    def forward(self, x):
        #print("Before conv block:", x.shape)
        x = self.block(x)
        #print("After conv block:", x.shape)
        return x
    
class ConvBlock2dWhisper(nn.Module): #per mfccs, funciona pel chroma tmb
    """
    Block of 4 Residual CNN, with channels (input, output) of (1, 16), (16, 32), (32, 64), and (64, 128) respectively,
    with Max Pooling between them with kernel sizes and strides adjusted to maintain dimensions, used withh Whisper Spectrograms
    """
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ResBlock2d(1, 16),
            nn.MaxPool2d(kernel_size=(4, 2)),#, stride=(2, 1), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(16, 32),
            nn.MaxPool2d(kernel_size=(4, 2)),#, stride=(2, 1), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(32, 64),
            nn.MaxPool2d(kernel_size=(4, 2)),#, stride=(2, 1), padding=(1, 0)),  # Adjust kernel size, stride, and padding
            ResBlock2d(64, 128)
        )

    def forward(self, x):
        #print("Before conv block:", x.shape)
        x = self.block(x)
        #print("After conv block:", x.shape)
        return x

class RNN(nn.Module):
    """
    Combination of Residual Convolutional Neural Network and Recurrent Neural Network, using 4 blocks of Residual Convolutions (ConvBlock2d), 1 LSTM layer and
    1 FC layer.
    """
    def __init__(self,data_type):
        super().__init__()
        self.name="RNN"
        self.data_type = data_type
        
        if data_type=='spectrogram':
            self.conv_block = ConvBlock2d()
        elif data_type=='whisper':
            self.conv_block = ConvBlock2dWhisper()
        else:
            self.conv_block = ConvBlock2dMFCCs()

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128,8)
        

    def forward(self,x):
        #print("Input shape:", x.shape)  # Add this line to print input shape
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        #print(x.shape)
        x = x.reshape((x.shape[0], x.shape[3], x.shape[1]))
        out, _ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        output = self.fc1(out)
        return output, F.softmax(output, dim=1), _
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)      
                    nn.init.zeros_(m.bias)

class CNN64(nn.Module):
    """
    2D Convolutional Neural Network model with 2 layers of convolutions and 3 FC layers, 5 layers depth.
    """
    def __init__(self):
        super(CNN64, self).__init__()
        self.name="CNN64"

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(148016, 256)  # Reduce the number of neurons
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 64)  # Reduce the number of neurons
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor

        out = self.fc1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.fc3(out)
        return out,F.softmax(out, dim=1)

class WaveNetBlock(nn.Module):
    '''
    1D Convolution Block which padding scales each iteration
    '''
    def __init__(self, in_channels, out_channels, dilation, kernel_size=2):
        super(WaveNetBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.view(128, 1, -1)
        out = self.conv(x)
        out = self.relu(out)
        return out

class WaveNet(nn.Module):
    '''
    WaveNet Convolutional Neural Network, simple convolutions wich padding scales and creates new 'images',
    whic are then pooled and classified
    '''
    def __init__(self, input_shape, num_classes, num_blocks=3, num_layers_per_block=5, num_filters=64, kernel_size=2):
        super(WaveNet, self).__init__()
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.initial_conv = nn.Conv1d(input_shape[0], num_filters, kernel_size=1)
        self.blocks = nn.ModuleList()

        for b in range(num_blocks):
            for i in range(num_layers_per_block):
                dilation = 2 ** i
                block = WaveNetBlock(num_filters, num_filters, dilation, kernel_size)
                self.blocks.append(block)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        out = self.initial_conv(x)

        for block in self.blocks:
            out = block(out)

        out = self.global_pooling(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class EmbeddingRNN(nn.Module):
    """
    Combination of Residual Convolutional Neural Network and Recurrent Neural Network, using 4 blocks of Residual Convolutions (ConvBlock2d), 1 LSTM layer and
    1 FC layer. It also has a method for returning the embeddings, before being passed trough the dense layer 
    """
    def __init__(self,data_type):
        super().__init__()
        self.name="RNN"
        self.data_type = data_type
        
        if data_type=='spectrogram':
            self.conv_block = ConvBlock2d()
        elif data_type=='whisper':
            self.conv_block = ConvBlock2dWhisper()
        else:
            self.conv_block = ConvBlock2dMFCCs()

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128,8)
        

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.reshape((x.shape[0], x.shape[3], x.shape[1]))
        out, _ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        output = self.fc1(out)
        return output, F.softmax(output, dim=1), out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)      
                    nn.init.zeros_(m.bias)
    
    def get_embedding(self,x):
        return self.forward(x)[2]
    

class NeuralNetwork(nn.Module):
    '''
    Simple Neural Network, with 2 FC layers, that classifies vectors of 512 features into 8 possible classes. It is used with the combination of the embeddings.
    '''
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=0)