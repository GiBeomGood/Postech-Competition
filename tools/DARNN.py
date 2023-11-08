import torch
from torch import nn, einsum
from torch.nn import functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, n, m):  # m: dimension of Encoder hidden state
        super().__init__()
        self.lstm = nn.LSTM(input_size=n, hidden_size=m, batch_first=True)
        self.initial_state = None

    def forward(self, x_t):
        # (1, -1 x m), (1, -1, m)
        _, (h_t, s_t) = self.lstm(x_t, self.initial_state)
        self.initial_state = (h_t, s_t)
        return h_t, s_t

    def reset_state(self, h_0, s_0):
        self.initial_state = (h_0, s_0)


class InputAttention(nn.Module):
    def __init__(self, T, n, m):
        super().__init__()
        self.v_e = nn.Linear(T, 1, bias=False)
        self.W_e = nn.Linear(2*m, T, bias=False)
        self.U_e = nn.Linear(T, T, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.n = n

    def forward(self, h_t, s_t, data):  # (1 x -1 x m), (1 x -1 x m)
        query = torch.cat([h_t, s_t], dim=2).permute(1, 0, 2)  # (1 x -1 x 2m) -> (-1 x 1 x 2m)
        query = query.repeat(1, self.n, 1)  # (-1 x n x 2m)

        alpha_t = torch.tanh(
            self.W_e(query) + self.U_e(data.permute(0, 2, 1)))  # (-1 x n x T)
        alpha_t = self.v_e(alpha_t).permute(0, 2, 1)  # (-1 x 1 x n)
        alpha_t = self.softmax(alpha_t)  # (-1 x 1 x n)

        return alpha_t  # (-1 x 1 x n)


class DecoderLSTM(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=p, batch_first=True)
        self.initial_state = None

    def forward(self, y_tilde):
        _, (d_t, s_t) = self.lstm(y_tilde, self.initial_state)  # (-1 x 1 x p)
        self.initial_state = (d_t, s_t)
        return d_t, s_t

    def reset_state(self, d_0, s_0):
        self.initial_state = (d_0, s_0)


class TemporalAttention(nn.Module):
    def __init__(self, m, p):
        super().__init__()
        self.v_d = nn.Linear(m, 1, bias=False)
        self.W_d = nn.Linear(2*p, m, bias=False)
        self.U_d = nn.Linear(m, m, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, d_t, s_t, H):  # (1 x -1 x p)
        T = H.shape[1]
        query = torch.cat([d_t, s_t], dim=2).permute(1, 0, 2)  # (1 x -1 x 2p) -> (-1 x 1 x 2p)
        query = query.repeat(1, T, 1) # (-1 x T x 2p)

        beta_t = torch.tanh(self.W_d(query) + self.U_d(H))  # (-1 x T x m)
        beta_t = self.v_d(beta_t)  # (-1 x T x 1)
        beta_t = self.softmax(beta_t)  # (-1 x T x 1)

        return beta_t


class Encoder(nn.Module):
    def __init__(self, T, n, m):
        super().__init__()
        self.input_attention_score = InputAttention(T, n, m)
        self.lstm = EncoderLSTM(n, m)

    def forward(self, data, h_0, s_0, T):  # data: (-1 x T+P x n)
        self.lstm.reset_state(h_0, s_0)
        h_t, s_t = h_0, s_0  # (1 x -1 x m)
        self.alpha_t = []

        H = torch.zeros((*data.shape[:-1], h_0.shape[-1])).to(data.device)  # (-1 x T+P x m)
        for t in range(T):
            # finding x_t_tilde
            x_t = data[:, [t], :]  # (-1 x 1 x n)
            alpha_t = self.input_attention_score(h_t, s_t, data)  # (-1 x 1 x n)
            # h_(t-1), s_(t-1), [x(1) ... x(n)] -> alpha_(t)
            x_t_tilde = x_t.mul(alpha_t)  # (-1 x 1 x n)
            self.alpha_t.append(alpha_t)

            # update hidden state
            h_t, s_t = self.lstm(x_t_tilde)  # (1 x -1 x m)
            H[:, t, :] = h_t[0]

        return H  # (-1 x T+P x m)


class Decoder(nn.Module):
    def __init__(self, m, p):
        super().__init__()
        self.temporal_attention_score = TemporalAttention(m, p)

        self.lstm = DecoderLSTM(p)
        self.w_tilde = nn.Linear(m+1, 1, bias=True)
        self.v_d = nn.Linear(m, 1, bias=False)
        self.W_d = nn.Linear(2*p, m, bias=False)
        self.U_d = nn.Linear(m, m, bias=False)
        self.v_y = nn.Linear(p, 1, bias=True)
        self.W_y = nn.Linear(m+p, p, bias=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, H, dec_data, d_0, s_0, T):
        d_t, s_t = d_0, s_0
        self.lstm.reset_state(d_0, s_0)

        for t in range(T-1):
            beta_t_1 = self.temporal_attention_score(d_t, s_t, H)  # (-1 x T x 1)
            c_t_1 = beta_t_1.permute(0, 2, 1).matmul(H)  # (-1 x 1 x m)
            
            y_t_1_tilde = torch.cat([dec_data[:, [t], :], c_t_1], dim=2)  # (-1 x 1 x m+1)
            y_t_1_tilde = self.w_tilde(y_t_1_tilde)  # (-1 x 1 x 1)
            d_t, s_t = self.lstm(y_t_1_tilde)  # (1 x -1 x p)

        beta_t_1 = self.temporal_attention_score(d_t, s_t, H)  # (-1 x T x 1)
        c_t_1 = beta_t_1.permute(0, 2, 1).matmul(H)  # (-1 x 1 x m)
        d_t = d_t.permute(1, 0, 2)  # (-1 x 1 x p)

        return torch.cat([d_t, c_t_1], dim=2)  # (-1 x 1 x m+p)


class DARNN(nn.Module):
    def __init__(self, T, P, n, m, p):
        super().__init__()
        # T: Time step
        # m: dimension of Encoder hidden state
        # p: dimension of Deocder hidden state

        self.T = T
        self.P = P
        self.m = m
        self.p = p
        self.encoder = Encoder(T+P, n, m)
        self.decoder = Decoder(m, p)
        self.linear1 = nn.Linear(m+p, p)
        self.linear2 = nn.Linear(p, P)
        self.relu = nn.ReLU()

    def forward(self, enc_data, dec_data):
        # enc: (-1 x T+P x n)
        # dec: (-1 x T x 1)
        batch_size = dec_data.shape[0]
        h0 = torch.zeros(1, batch_size, self.m, device=dec_data.device)
        d0 = torch.zeros(1, batch_size, self.p, device=dec_data.device)

        H = self.encoder(enc_data, h0, h0, self.T+self.P)  # (-1 x T x m)
        dec_output = self.decoder(H, dec_data, d0, d0, self.T)  # (-1 x 1 x m+p)
        dec_output = dec_output.squeeze(dim=1)  # (-1 x m+p)

        output = self.relu(self.linear1(dec_output))  # (-1 x p)
        output = self.linear2(output)  # (-1 x P)

        return output


    def get_input_attention_score(self):
        if type(self.encoder.alpha_t) == list:
            self.encoder.alpha_t = torch.cat(self.encoder.alpha_t, dim=1)  # (-1 x T+P x m)
        return self.encoder.alpha_t


class QARNN(nn.Module):
    def __init__(self, T, P, n, m, p):
        super().__init__()
        # T: Time step
        # m: dimension of Encoder hidden state
        # p: dimension of Deocder hidden state

        self.T = T
        self.P = P
        self.m = m
        self.p = p
        self.encoder1 = Encoder(T+P, n-5, m)
        self.encoder2 = Encoder(T+P, 5, m)
        self.decoder1 = Decoder(m, p)
        self.decoder2 = Decoder(m, p)
        self.linear1 = nn.Linear(2*(m+p), 2*p)
        self.linear2 = nn.Linear(2*p, P)
        self.relu = nn.ReLU()

    def forward(self, enc_data, dec_data):
        # enc: (-1 x T+P x n)
        # dec: (-1 x T x 1)
        batch_size = dec_data.shape[0]
        h0 = torch.zeros(1, batch_size, self.m, device=dec_data.device)
        d0 = torch.zeros(1, batch_size, self.p, device=dec_data.device)

        H1 = self.encoder1(enc_data[:, :, :-5], h0, h0, self.T+self.P)  # (-1 x T+P x m)
        H2 = self.encoder2(enc_data[:, :, -5:], h0, h0, self.T+self.P)  # (-1 x T+P x m)

        dec_output1 = self.decoder1(H1, dec_data, d0, d0, self.T)  # (-1 x 1 x m+p)
        dec_output2 = self.decoder2(H2, dec_data, d0, d0, self.T)
        dec_output1 = dec_output1.squeeze(dim=1)  # (-1 x m+p)
        dec_output2 = dec_output2.squeeze(dim=1)  # (-1 x m+p)

        output = torch.cat([dec_output1, dec_output2], dim=1)  # (-1 x 2(m+p))
        output = self.relu(self.linear1(output))  # (-1 x 2p)
        output = self.linear2(output)  # (-1 x P)

        return output
    

    def get_input_attention_score(self):
        if type(self.encoder2.alpha_t) == list:
            self.encoder2.alpha_t = torch.cat(self.encoder2.alpha_t, dim=1)  # (-1 x T+P x m)
        return self.encoder2.alpha_t