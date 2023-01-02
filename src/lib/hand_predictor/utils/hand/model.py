import torch, pdb
import torch.nn as nn, numpy as np


class SampleCNNGA(nn.Module):
    def __init__(self, n_classes=2, in_channels=42):
        super(SampleCNNGA, self).__init__()

        self.in_channels = in_channels
        # 300 x 42
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128), nn.ReLU())
        # 100 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3, stride=3))
        # 33 x 128
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3, stride=3))
        # 11 x 128
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(3, stride=3))
        # 3 x 256
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(3, stride=3))
        # 1 x 256
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.5))

        # 1 x 256
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, n_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # input x : 23 x 59049 x 1
        # expected conv1d input : minibatch_size x num_channel x width
        x = x.view(x.shape[0], self.in_channels, -1)
        # x : 23 x 1 x 59049
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #out = self.conv6(out)

        out = self.avgpooling(out)

        out = out.view(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)

        #logit = self.activation(logit)

        return logit


class PoseConvBlock(nn.Module):
    def __init__(self,
                 input_channels=32,
                 middle_channels=32,
                 dilation_list=[1, 2, 4]):
        super(PoseConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels,
                               middle_channels,
                               3,
                               stride=1,
                               padding=dilation_list[0],
                               dilation=dilation_list[0],
                               groups=1,
                               bias=True)
        self.batchnorm1 = nn.BatchNorm1d(middle_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(middle_channels,
                               middle_channels,
                               3,
                               stride=1,
                               padding=dilation_list[1],
                               dilation=dilation_list[1],
                               groups=1,
                               bias=True)
        self.batchnorm2 = nn.BatchNorm1d(middle_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(middle_channels,
                               middle_channels,
                               3,
                               stride=1,
                               padding=dilation_list[2],
                               dilation=dilation_list[2],
                               groups=1,
                               bias=True)
        self.batchnorm3 = nn.BatchNorm1d(middle_channels)
        self.relu3 = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        #residual = x
        x = self.conv1(x)  # (batch, channels, d) -> (batch, 12, d)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        x = self.conv2(x)  # (batch, 12, d) -> (batch, 12, d)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        x = self.conv3(x)  # (batch, 12, d) -> (batch, 12, d/2)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        #x = x + residual
        x = self.pool(x)  # (batch, 12, d/2) -> (batch, 12, d/4)
        return x


class HandConvNet(nn.Module):
    def __init__(self,
                 input_channels,
                 output_class,
                 crop_len,
                 middle_channels=32,
                 device='cuda:0'):
        super(HandConvNet, self).__init__()
        self.middle_channels = middle_channels
        self.crop_len = crop_len
        self.device = device

        # CNN setting
        self.extcnn = nn.Conv1d(input_channels,
                                middle_channels,
                                3,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                bias=False)
        self.cnnblock1 = PoseConvBlock(input_channels=middle_channels,
                                       middle_channels=middle_channels * 2,
                                       dilation_list=[1, 2, 4])
        self.cnnblock2 = PoseConvBlock(input_channels=middle_channels * 2,
                                       middle_channels=middle_channels * 4,
                                       dilation_list=[8, 16, 32])
        self.cnnblock3 = PoseConvBlock(input_channels=middle_channels * 4,
                                       middle_channels=middle_channels * 8,
                                       dilation_list=[64, 128, 256])
        
        self.dropout = nn.Dropout(0.5)
        self.linear_class = nn.Sequential(
            nn.Linear(self.middle_channels * 8, self.middle_channels * 4), nn.ReLU(),
            nn.Linear(self.middle_channels * 4, self.middle_channels * 2), nn.ReLU(),
            nn.Linear(self.middle_channels * 2, output_class))
        self.adtavgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.extcnn(x)
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        x = self.cnnblock3(x)
        x = self.dropout(x)
        output = self.adtavgpool(x)
        output = output.view(-1, self.middle_channels * 8)
        output = self.linear_class(output)
        return output

    def latentspace(self, x, type="FC"):
        '''
        type: 
            'FC': for fully connected layer: shape=(batch, hidden layers)
            'CNN-raw': for CNN raw output: shape=(batch, hidden layers, length)
        '''
        x = self.extcnn(x)
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        output = self.cnnblock3(x)
        
        if type == "FC":
            output = self.dropout(output)
            output = self.adtavgpool(output)
            output = output.view(-1, self.middle_channels * 4)
        elif type == "CNN-raw":
            pass
        else:
            raise NotImplementedError
        
        return output

class HandConvNet_o(nn.Module):
    def __init__(self,
                 input_channels,
                 output_class,
                 crop_len,
                 middle_channels=32,
                 device='cuda:0'):
        super(HandConvNet_o, self).__init__()
        self.middle_channels = middle_channels
        self.crop_len = crop_len
        self.device = device

        # CNN setting
        self.extcnn = nn.Conv1d(input_channels,
                                middle_channels,
                                32,
                                stride=8,
                                padding=0,
                                dilation=1,
                                groups=1,
                                bias=False)
        self.cnnblock1 = PoseConvBlock(input_channels=middle_channels,
                                       middle_channels=middle_channels * 2,
                                       dilation_list=[1, 2, 4])
        self.cnnblock2 = PoseConvBlock(input_channels=middle_channels * 2,
                                       middle_channels=middle_channels * 4,
                                       dilation_list=[8, 16, 32])
        
        self.dropout = nn.Dropout(0.4)
        self.linear_class = nn.Sequential(
            nn.Linear(self.middle_channels * 4, self.middle_channels * 2), 
            nn.Linear(self.middle_channels * 2, output_class))
        self.adtavgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.extcnn(x)
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        x = self.dropout(x)
        output = self.adtavgpool(x)
        output = output.view(-1, self.middle_channels * 4)
        output = self.linear_class(output)
        return output

    def latentspace(self, x, type="FC"):
        '''
        type: 
            'FC': for fully connected layer: shape=(batch, hidden layers)
            'CNN-raw': for CNN raw output: shape=(batch, hidden layers, length)
        '''
        x = self.extcnn(x)
        x = self.cnnblock1(x)
        output = self.cnnblock2(x)
        
        if type == "FC":
            output = self.dropout(output)
            output = self.adtavgpool(output)
            output = output.view(-1, self.middle_channels * 4)
        elif type == "CNN-raw":
            pass
        else:
            raise NotImplementedError
        
        return output

class HandRNNConvNet(nn.Module):
    def __init__(self,
                 input_channels,
                 output_class,
                 crop_len,
                 middle_channels=32,
                 hiddens=64,
                 device='cuda:0'):
        super(HandRNNConvNet, self).__init__()
        self.device = device
        self.crop_len = crop_len
        self.middle_channels = middle_channels
        self.num_rnn_layers = 2

        # CNN setting
        self.extcnn = nn.Conv1d(input_channels,
                                middle_channels,
                                3,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1,
                                bias=False)
        self.cnnblock1 = PoseConvBlock(input_channels=middle_channels,
                                       middle_channels=middle_channels * 2,
                                       dilation_list=[1, 2, 4])
        self.cnnblock2 = PoseConvBlock(input_channels=middle_channels * 2,
                                       middle_channels=middle_channels * 4,
                                       dilation_list=[8, 16, 32])
        self.cnnblock3 = PoseConvBlock(input_channels=middle_channels * 4,
                                       middle_channels=middle_channels * 8,
                                       dilation_list=[64, 128, 256])
        self.cnnblocktemp = PoseConvBlock(input_channels=middle_channels,
                                       middle_channels=middle_channels * 4,
                                       dilation_list=[1, 4, 32])
        
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1d = nn.BatchNorm1d(middle_channels * 8)
        self.last_leakyrelu = nn.ReLU()
        self.last_dropout = nn.Dropout(0.5)
        self.linear_class = nn.Sequential(
            nn.Linear(self.middle_channels * 8, self.middle_channels * 4), nn.ReLU(),
            nn.Linear(self.middle_channels * 4, self.middle_channels * 2), nn.ReLU(),
            nn.Linear(self.middle_channels * 2, output_class))
        self.adtavgpool = nn.AdaptiveAvgPool1d(1)

        # GRU setting
        self.rnnblock = nn.GRU(batch_first=True,
                                input_size=input_channels,
                                hidden_size=middle_channels,
                                num_layers=self.num_rnn_layers,
                                bias=True, dropout=0.5)
        
        self.gru_linear = nn.Linear(self.middle_channels, self.middle_channels*8)


    def forward(self, x):
        # print(x.shape)
        cnn_output = self.extcnn(x)
        # print("1. ", cnn_output.shape)
        cnn_output = self.cnnblock1(cnn_output)
        # print("2. ", cnn_output.shape)
        cnn_output = self.cnnblock2(cnn_output)
        # print("3. ", cnn_output.shape)
        cnn_output = self.cnnblock3(cnn_output)
        # print("4. ", cnn_output.shape)
        cnn_output = self.dropout(cnn_output)
        # print("5. ", cnn_output.shape)
        cnn_output = self.adtavgpool(cnn_output)
        # print("6. ", cnn_output.shape)
        cnn_output = cnn_output.view(-1, self.middle_channels * 8)
        # print("7. ", cnn_output.shape)
        cnn_output = self.batchnorm1d(cnn_output)

        # rnn forward
        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size)
        gru_output = x.transpose(1, 2)
        gru_output, self.hidden = self.rnnblock(gru_output, self.hidden)
        self.hidden = self.hidden.detach().cuda(self.device)
        gru_output = torch.squeeze(self.gru_linear(gru_output))
        # print("1.", gru_output.shape)
        gru_output = gru_output.transpose(1, 2)
        # print("2.", gru_output.shape)
        gru_output = self.adtavgpool(gru_output)
        # print("3.", gru_output.shape)
        gru_output = gru_output.view(-1, self.middle_channels * 8)
        # print("4.", gru_output.shape)
        # print(cnn_output.shape)
        # raise
        gru_output = self.batchnorm1d(gru_output)
        
        output = gru_output.add(cnn_output)
        output = self.linear_class(output)
        return output

    def latentspace(self, x):
        cnn_output = self.extcnn(x)
        cnn_output = self.cnnblock1(cnn_output)
        cnn_output = self.cnnblock2(cnn_output)
        cnn_output = self.cnnblock3(cnn_output)
        cnn_output = self.dropout(cnn_output)
        cnn_output = self.adtavgpool(cnn_output)
        cnn_output = cnn_output.view(-1, self.middle_channels * 8)
        cnn_output = self.batchnorm1d(cnn_output)

        batch_size = x.shape[0]
        self.hidden = self.init_hidden(batch_size)
        gru_output = x.transpose(1, 2)
        gru_output, self.hidden = self.rnnblock(gru_output, self.hidden)
        self.hidden = self.hidden.detach().cuda(self.device)
        gru_output = torch.squeeze(self.gru_linear(gru_output))
        gru_output = gru_output.transpose(1, 2)
        gru_output = self.adtavgpool(gru_output)
        gru_output = gru_output.view(-1, self.middle_channels * 8)
        gru_output = self.batchnorm1d(gru_output)
        
        output = gru_output.add(cnn_output)

        return output
    
    
    '''Initializes hidden state'''
    def init_hidden(self, batch_size):

        # Creates initial hidden state for GRU of zeroes
        hidden = torch.ones(self.num_rnn_layers, 
                            batch_size, 
                            self.middle_channels).cuda(self.device)

        return hidden

# Multichannel CNN-GRU model.
# A Multichannel CNN-GRU Model for Human Activity Recognition
# doi: 10.1109/ACCESS.2022.3185112

class CNNChannel(nn.Module):
    def __init__(self,
                 input_channels=32,
                 middle_channels=64,
                 filter_size = 3):
        super(CNNChannel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels,
                               middle_channels,
                               kernel_size=filter_size,
                               stride=1,
                               padding='same',
                               bias=True)
        self.batchnorm1 = nn.BatchNorm1d(middle_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(middle_channels,
                               middle_channels*2,
                               kernel_size=filter_size,
                               stride=1,
                               padding='same',
                               bias=True)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        #residual = x
        x = self.conv1(x)  # (batch, channels, d) -> (batch, 12, d)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)  # (batch, 12, d) -> (batch, 12, d)
        x = self.relu2(x)

        #x = x + residual
        x = self.pool(x)  # (batch, 12, d) -> (batch, 12, d/2)
        return x


class HandMultichannelCNNGRU(nn.Module):
    def __init__(self,
                 input_channels,
                 output_class,
                 crop_len,
                 middle_channels=64,
                 device='cuda:0'):
        super(HandMultichannelCNNGRU, self).__init__()
        self.device = device
        self.crop_len = crop_len
        self.middle_channels = middle_channels
        self.num_rnn_layers = 3

        # CNN setting
        self.cnn_channel1 = CNNChannel(input_channels=input_channels,
                                       middle_channels=middle_channels,
                                       filter_size=3)
        self.cnn_channel2 = CNNChannel(input_channels=input_channels,
                                       middle_channels=middle_channels,
                                       filter_size=5)
        self.cnn_channel3 = CNNChannel(input_channels=input_channels,
                                       middle_channels=middle_channels,
                                       filter_size=7)
        self.linear_class = nn.Linear(middle_channels, output_class)
        #self.linear_class = nn.Linear(self.crop_len//2, output_class)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.batchnorm1d = nn.BatchNorm1d(middle_channels)
        #self.batchnorm1d = nn.BatchNorm1d(self.crop_len//2)

        # GRU setting
        self.rnnblock1 = nn.GRU(batch_first=True,
                                input_size=middle_channels*2*3,
                                hidden_size=128,
                                num_layers=self.num_rnn_layers,
                                bias=True, dropout=0)
        
        self.rnnblock2 = nn.GRU(batch_first=True,
                                input_size=128,
                                hidden_size=64,
                                num_layers=self.num_rnn_layers,
                                bias=True, dropout=0)

    def forward(self, x):
        
        # 3 channels of CNN
        cnn_output1 = self.cnn_channel1(x)
        cnn_output2 = self.cnn_channel2(x)
        cnn_output3 = self.cnn_channel3(x)

        cnn_output = torch.cat([cnn_output1, cnn_output2, cnn_output3], axis=1)
        # print(cnn_output1.shape, cnn_output2.shape, cnn_output3.shape)

        # rnn forward
        batch_size = x.shape[0]
        self.hidden1 = self.init_hidden(batch_size, middle_channels=128)
        self.hidden2 = self.init_hidden(batch_size, middle_channels=64)

        gru_output = cnn_output.transpose(1, 2)
        gru_output, self.hidden1 = self.rnnblock1(gru_output, self.hidden1)
        self.hidden1 = self.hidden1.detach().cuda(self.device)
        # print("1.", gru_output.shape)

        gru_output, self.hidden2 = self.rnnblock2(gru_output, self.hidden2)
        self.hidden2 = self.hidden2.detach().cuda(self.device)
        # print("2.", gru_output.shape)
        
        gru_output = gru_output.transpose(1, 2)
        gru_output = self.globalavgpool(gru_output)
        # print("3.", gru_output.shape)
        
        gru_output = gru_output.view(-1, self.middle_channels)
        #gru_output = gru_output.view(-1, self.crop_len//2)
        # print("4.", gru_output.shape)
        
        gru_output = self.batchnorm1d(gru_output)
        
        output = self.linear_class(gru_output)
        return output

    def latentspace(self, x):
        cnn_output1 = self.cnn_channel1(x)
        cnn_output2 = self.cnn_channel2(x)
        cnn_output3 = self.cnn_channel3(x)
        cnn_output = torch.cat([cnn_output1, cnn_output2, cnn_output3], axis=1)

        batch_size = x.shape[0]
        self.hidden1 = self.init_hidden(batch_size, middle_channels=128)
        self.hidden2 = self.init_hidden(batch_size, middle_channels=64)

        gru_output = cnn_output.transpose(1, 2)
        gru_output, self.hidden1 = self.rnnblock1(gru_output, self.hidden1)
        self.hidden1 = self.hidden1.detach().cuda(self.device)

        gru_output, self.hidden2 = self.rnnblock2(gru_output, self.hidden2)
        self.hidden2 = self.hidden2.detach().cuda(self.device)
        
        gru_output = gru_output.transpose(1, 2)
        gru_output = self.globalavgpool(gru_output)
        
        gru_output = gru_output.view(-1, self.middle_channels)
        gru_output = self.batchnorm1d(gru_output)
        return gru_output
    
    
    '''Initializes hidden state'''
    def init_hidden(self, batch_size, middle_channels):

        # Creates initial hidden state for GRU of zeroes
        hidden = torch.ones(self.num_rnn_layers, 
                            batch_size, 
                            middle_channels).cuda(self.device)

        return hidden

# Thought from Multichannel CNN-GRU model.
# A Multichannel CNN-GRU Model for Human Activity Recognition
# doi: 10.1109/ACCESS.2022.3185112

class BothHandLRchannelGRU(nn.Module):
    # latent space: (batch size, L+R channels, length) 
    def __init__(self, 
                in_channels=256,
                middle_channels=64,
                output_class=2,
                num_rnn_layers = 3,
                device='cuda:0'):
        super(BothHandLRchannelGRU, self).__init__()

        self.device = device
        self.middle_channels = middle_channels
        self.num_rnn_layers = num_rnn_layers

        self.linear_class = nn.Linear(middle_channels, output_class)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.batchnorm1d = nn.BatchNorm1d(middle_channels)

        # GRU setting
        self.rnnblock1 = nn.GRU(batch_first=True,
                                input_size=in_channels,
                                hidden_size=middle_channels*2,
                                num_layers=self.num_rnn_layers,
                                bias=True, dropout=0.3)
        
        self.rnnblock2 = nn.GRU(batch_first=True,
                                input_size=middle_channels*2,
                                hidden_size=middle_channels,
                                num_layers=self.num_rnn_layers,
                                bias=True, dropout=0.3)

    def forward(self, x):
        # rnn forward
        batch_size = x.shape[0]
        self.hidden1 = self.init_hidden(batch_size, middle_channels=128)
        self.hidden2 = self.init_hidden(batch_size, middle_channels=64)

        # x: (batch, L+R channels, length)
        gru_output = x.transpose(1, 2) # equal to 'x.permute(0,2,1)'
        # gru_output: (batch, length, L+R channels)
        gru_output, self.hidden1 = self.rnnblock1(gru_output, self.hidden1)
        self.hidden1 = self.hidden1.detach().cuda(self.device)
        # gru_output: (batch, length, hidden_layers=128) #print("1.", gru_output.shape)

        gru_output, self.hidden2 = self.rnnblock2(gru_output, self.hidden2)
        self.hidden2 = self.hidden2.detach().cuda(self.device)
        # gru_output: (batch, length, hidden_layers=64) #print("2.", gru_output.shape)
        
        gru_output = gru_output.transpose(1, 2) # equal to 'x.permute(0,2,1)'
        # gru_output: (batch, hidden_layers=64, length)
        gru_output = self.globalavgpool(gru_output)
        # gru_output: (batch, hidden_layers=64, 1) #print("3.", gru_output.shape)
        
        gru_output = gru_output.view(-1, self.middle_channels)
        # gru_output: (batch, hidden_layers=64) #print("4.", gru_output.shape)

        gru_output = self.batchnorm1d(gru_output)
        output = self.linear_class(gru_output)
        # output: (batch, class_nums)
        return output

    '''Initializes hidden state'''
    def init_hidden(self, batch_size, middle_channels):

        # Creates initial hidden state for GRU of zeroes
        hidden = torch.ones(self.num_rnn_layers, 
                            batch_size, 
                            middle_channels).cuda(self.device)

        return hidden

class BothHandFullConnected(nn.Module):
    # only for (batch size, L+R channels) 
    def __init__(self, in_channels=256, output_class=2) -> None:
        super(BothHandFullConnected, self).__init__()
    
        self.in_channels = in_channels

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(), 
            nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_class), 
        ) 
    
    def forward(self, x):
        output = self.fc(x)
        return output

class BothHandTransformer(nn.Module):
    # only for (batch size, L+R channels) 
    def __init__(self, in_channels=256, d_model=8, output_class=2) -> None:
        super(BothHandTransformer, self).__init__()
    
        self.in_channels = in_channels
        self.each_hand_channels = in_channels//2
        self.prenet = nn.Linear(2, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=64, nhead=4
        )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), 
            #nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(64, output_class), 
        ) 
    
    def forward(self, x):
        # reshape: (batch size, L+R channels) -> (batch, each hand channels, 2)
        batch_size = np.shape(x)[0]
        output = torch.reshape(x, (batch_size, 2, self.each_hand_channels))
        output = output.permute(0, 2, 1)
        # prenet transform: (batch, each hand chs, 8)
        output = self.prenet(output)
        # output: (each hand chs, batch size, d_model)
        output = output.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (each hand chs, batch size, d_model).
        output = self.encoder_layer(output)
        # output: (batch size, each hand chs, d_model)
        output = output.transpose(0, 1)
		# mean pooling
        stats = output.mean(dim=1)

        # (batch, output_class)
        output = self.fc(stats)
        return output

class BothHandCNN(nn.Module):
    # only for (batch size, L+R channels) 
    def __init__(self, in_channels=256, middle_channels=8, output_class=2) -> None:
        super(BothHandCNN, self).__init__()

        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.prenet = nn.Conv1d(1, middle_channels, 3,
                                stride=1, padding=0, dilation=1, groups=1,
                                bias=False)
        self.cnnblock1 = PoseConvBlock(input_channels=middle_channels,
                                       middle_channels=middle_channels * 2,
                                       dilation_list=[1, 2, 4])
        
        self.fc = nn.Sequential(
            nn.Linear(middle_channels * 2, middle_channels), nn.ReLU(), 
            #nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(middle_channels, middle_channels), nn.ReLU(), 
            nn.Linear(middle_channels, output_class), 
        ) 
        self.dropout = nn.Dropout(0.5)
        self.adtavgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # reshape: (batch size, L+R channels) -> (batch, 1, L+R channels)
        batch_size = np.shape(x)[0]
        output = torch.reshape(x, (batch_size, 1, self.in_channels))
        # output: (batch, middle_channels, L+R channels)
        output = self.prenet(output)
        # output: (batch size, middle_channels*2, L+R channels).
        output = self.cnnblock1(output)
        output = self.dropout(output)
        # output: (batch size, middle_channels*2, 1).
        output = self.adtavgpool(output)
        # output: (batch size, middle_channels*2).
        output = output.view(-1, self.middle_channels * 2)
        
        # output: (batch, output_class)
        output = self.fc(output)

        
        return output