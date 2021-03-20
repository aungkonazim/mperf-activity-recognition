from import_lib import *

class CNN2GRU_whole(nn.Module):
    def __init__(self, num_labels, use_cuda=False, n_layers=1, input_channel_size=3, hidden_dim=128, is_probability=False, do_ret_features=False):
        super(CNN2GRU_whole, self).__init__()
        self.input_size = input_channel_size # three axis accel
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels # numbe of participants
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.is_probability = is_probability
        self.do_ret_features = do_ret_features

        self.c1 = nn.Conv1d(input_channel_size, hidden_dim, 2)
        self.p1 = nn.MaxPool1d(2)
        self.c2 = nn.Conv1d(hidden_dim, hidden_dim, 2)
        self.p2 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.5)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def initHidden(self, batch_size):
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim))

    def forward(self, inputs):
        # input is (n_size, 1, 3) accel
        
#         print(np.shape(inputs))

        batch_size = inputs.size(1)
        hidden = self.initHidden(batch_size)
        if self.use_cuda:
            hidden = hidden.cuda()

        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        c = F.relu(c)

        p = self.p1(c)
        c = self.c2(p)
        c = F.relu(c)
        #         p = c
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)

        p = F.tanh(p)
        p = self.dropout1(p)
        gru_output, hidden = self.gru(p, hidden)

        #         h = gru_output.mean(dim=0)
        hidden = F.tanh(hidden)
        h = hidden.view(batch_size, -1)

        #         out = F.relu(self.fc(h))
        out = self.fc(h)
        if self.is_probability and self.do_ret_features:
            return h, F.softmax(out, dim=1)
        elif self.is_probability:
            return [], F.softmax(out, dim=1)
        else:
            return [], F.log_softmax(out, dim=1)
