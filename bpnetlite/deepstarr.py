import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSTARR(torch.nn.Module):

    """
    Pytorch implementation of the DeepSTARR model.
    Written by Monika.
    """
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_length, input_channel, output_channel1, output_channel2, output_channel3, output_channel4,
                 kernel_size1, kernel_size2, kernel_size3, kernel_size4, stride, kernel_pool,
                 padding, nlayers_cnn, nlayers_fc, output_fc1, output_fc2, prob):
        super(DeepSTARR, self).__init__()
        self.input_length = torch.tensor([input_length])
        self.input_channel = 4
        self.output_channel1 = output_channel1
        self.output_channel2 = output_channel2
        self.output_channel3 = output_channel3
        self.output_channel4 = output_channel4
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.kernel_size4 = kernel_size4
        self.nlayers_cnn = nlayers_cnn
        self.nlayers_fc = nlayers_fc
        self.stride = 1
        self.padding = 'same'
        self.kernel_pool = 2 # also default stride
        self.prob = prob
                
        # Conv layer 1
        self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel1, 
            kernel_size=self.kernel_size1, stride=self.stride, padding=self.padding)

        x_len_out = self.input_length
            
        # Batch Normalization 1
        self.conv1_bn = nn.BatchNorm1d(self.output_channel1)
        self.relu1 = nn.ReLU()
        
        # Max pooling 1
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
        x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
        
        current_dim = self.output_channel1
        self.layers_cnn = nn.ModuleList()
        self.layers_fc = nn.ModuleList()
        hidden_dim = [output_channel1, output_channel2, output_channel3, output_channel4]
        kernel_sizes = [kernel_size1, kernel_size2, kernel_size3, kernel_size4]
        outputs_fc = [output_fc1, output_fc2]
        for i in range(1, nlayers_cnn):
            self.layers_cnn.append(torch.nn.Conv1d(current_dim, hidden_dim[i], 
                kernel_size=kernel_sizes[i], stride=self.stride, padding=self.padding))

            self.layers_cnn.append(nn.BatchNorm1d(hidden_dim[i]))
            self.layers_cnn.append(nn.ReLU())
            self.layers_cnn.append(nn.MaxPool1d(kernel_size=self.kernel_pool))
            
            x_len_out = outputLen_MaxPool(x_len_out, self.layers_cnn[-1].kernel_size, self.layers_cnn[-1].padding, self.layers_cnn[-1].stride, self.layers_cnn[-1].dilation)
            current_dim = hidden_dim[i]
            
        self.flatten = nn.Flatten()
        
        input_final_fc = current_dim*int(x_len_out.item())

        for i in range(0, nlayers_fc):
            self.layers_fc.append(nn.Linear(input_final_fc, outputs_fc[i]))
            self.layers_fc.append(nn.BatchNorm1d(outputs_fc[i]))
            self.layers_fc.append(nn.ReLU())
            self.layers_fc.append(nn.Dropout(self.prob))
            input_final_fc = outputs_fc[i]
        
        self.dense_output = nn.Linear(input_final_fc, 1)
        
    def forward(self, x):
        x = self.relu1(self.conv1_bn(self.conv1(x.float())))
        x = self.pool(x)

        for layer in self.layers_cnn:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.layers_fc:
            x = layer(x)

        return self.dense_output(x)
