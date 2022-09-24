import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class tTGCN(nn.Module):
# class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(tTGCN, self).__init__()
        # super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class senet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//16, False),
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        #b,c,h,w -> b,c,1,1
        avg = self.avg_pool(x).view([b, c])

        #b,c -> b,c // ratio -> b,c -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])

        return x*fc

class senet_1d(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(senet_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//16, False),
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self,x):
        b, n, flow = x.size()
        #b,c,h,w -> b,c,1,1
        avg = self.avg_pool(x).view([b, n])
        # print(avg.size())
        #b,c -> b,c // ratio -> b,c -> b,c,1,1
        fc = self.fc(avg).view([b, n, 1])
        # print(fc.size())

        return x*fc

class WeaLSTM(nn.Module):
    def __init__(self, num_nodes, hidden_dim, seq_len):
        super(WeaLSTM, self).__init__()
        self._num_nodes = num_nodes
        self._hidden_dim = hidden_dim
        self._seq_len = seq_len
        self.linear_1 = nn.Linear(self._num_nodes, 2*self._num_nodes)
        self.lstm_1 = nn.LSTM(input_size=2*self._num_nodes, hidden_size=self._num_nodes, num_layers=1, bias=False)
        # self.lstm_2 = nn.LSTM(input_size=2*self._num_nodes, hidden_size=self._num_nodes, num_layers=1, bias=False)
        self.linear_2 = nn.Linear(self._seq_len, self._hidden_dim)

    def forward(self, input):
        input = input.transpose(0, 1)
        output = self.linear_1(input)
        output, (b, c) = self.lstm_1(output)
        # output = self.lstm_2(output)
        output = output.transpose(0, 1).transpose(1, 2)
        output = self.linear_2(output)

        return output


class ResNet_1d(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResNet_1d, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        # 输入进来的参数是（batch，156，3*self._hidden_dim）
        self.layer = nn.Sequential(
            nn.Conv1d(self._input_dim, self._input_dim//2, 1),
            nn.BatchNorm1d(self._input_dim//2),
            nn.ReLU(),
            nn.Conv1d(self._input_dim//2, self._input_dim//2, 1),
            nn.BatchNorm1d(self._input_dim//2),
            nn.ReLU(),
        )
        self.conv_1d_1 = nn.Conv1d(self._input_dim, self._input_dim//2, 1)
        self.conv_1d_2 = nn.Conv1d(self._input_dim//2, self._input_dim, 1)

    def forward(self, X):
        batch, num_nodes, hidden_dim = X.shape
        output = self.layer(X)
        # print(output.size())
        output = output + self.conv_1d_1(X)
        output = self.conv_1d_2(output)
        # print(output.size())
        return output

class TGCN(nn.Module):
# class At_TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, **kwargs):
        super(TGCN, self).__init__()
        # super(At_TGCN).__init__()
        # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.t_tgcn = tTGCN(self.adj, self._hidden_dim)
        self.d_tgcn = tTGCN(self.adj, self._hidden_dim)
        self.w_lstm = WeaLSTM(156, self._hidden_dim, 15)
        self._attention = senet(channel=3)       # 156为城市节点数
        self._attention_1d = senet_1d(156)
        # self.ResNet = ResNet_1d(self._input_dim, 3*hidden_dim)
        self._gru = nn.GRU(3*self._hidden_dim, 3*self._hidden_dim, bidirectional=False)
        self.linear = nn.Linear(3*self._hidden_dim, self._hidden_dim)
        # self.w_tgcn = TGCN(self._adj, self._hidden_dim)
        # self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=1, bias=Ture, padding=0)  #kernel_size还不确定

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        t_inputs = inputs[:, :12, :]
        # print(t_inputs.size())
        d_inputs = inputs[:, 12:24, :]
        weather_inputs = inputs[:, 24:, :]
        # print(weather_inputs.size())
        # w_inputs = inputs[:, 2, :, :]
        t_output = self.t_tgcn(t_inputs)
        # t_output = self._attention_1d(t_output).reshape(-1, 1, self._input_dim, self._hidden_dim)
        d_output = self.d_tgcn(d_inputs)
        # d_output = self._attention_1d(d_output).reshape(-1, 1, self._input_dim, self._hidden_dim)
        # w_output = self.w_lstm(weather_inputs).reshape(-1, 1, self._input_dim, self._hidden_dim)
        w_output = self.w_lstm(weather_inputs)
        output = torch.cat((t_output, d_output, w_output), dim=2)
        # output = torch.cat((t_output, d_output, w_output), dim=1)   #(-1,3,156,64)
        # print('output:', output.size())
        # output: torch.Size([32, 156, 192])
        output = self._attention_1d(output)
        # output = self._attention(output).reshape(batch_size, self._input_dim, -1)  #(-1, 156, 192)
        # output = self.ResNet(output)
        output = self._gru(output)[0]
        output = self.linear(output)
        return output


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


