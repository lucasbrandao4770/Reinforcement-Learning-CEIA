import torch
import torch.nn as nn
import numpy as np


class MyModel(nn.Module):
    """Rede neural"""
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.sig = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = self.fc3(x)
        x = self.sig(x)

        out = self.softmax(x)
        return out


def expert(input):
    v = np.asarray(input).mean()
    if v >= 2/3:
        a = 0
    elif 2/3 > v > 1/3:
        a = 1
    elif 1/3 >= v:
        a = 2
    return a


def collect_dataset(n):
    a = []
    inputs = []
    for i in range(n):
        input = np.random.rand(4)
        inputs.append(input)
        a.append(expert(input))
    a = nn.functional.one_hot(torch.Tensor(a).to(torch.int64), 3)
    return torch.Tensor(np.asarray(inputs)), a


def lossfunc(model_out, true_out):
    r"""Função de perda
        L = -log(P(a*|s))
    """
    a_prob = model_out*true_out
    return -torch.log(a_prob.sum(1)).mean()


def acc(model_out, true_out):
    """Acurácia, para medir o quão bom está o modelo.
    O quanto ele consegue acertar as respostas.
    """
    model_a = model_out.argmax(1)
    true_a = true_out.argmax(1)
    return torch.sum(model_a == true_a)


def main():
    model = MyModel(4, 3)  # 4 coisas entram, 3 coisas saem
    learning_rate = 0.0045
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.train()
    for epoch in range(50):
        x, y = collect_dataset(100000)
        m = model(x)
        loss = lossfunc(m, y)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_acc = acc(m, y) / 100000

        train_loss = loss.detach().item()
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.4f' \
              % (epoch, train_loss, train_acc))
    pass


if __name__ == '__main__':
    main()
