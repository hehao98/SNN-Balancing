import os
import json
import logging
import torch
import torchvision
from torch import nn
from spikingjelly.clock_driven import neuron, functional


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR, MOMENTNUM = 100, 100, 0.1, 0.5
TIME, V_THR, V_RESET = 100, 1.0, 0.0


def get_mnist():
    mnist_train = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True),
        BATCH_SIZE, shuffle=True)
    mnist_test = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True),
        BATCH_SIZE, shuffle=False)
    mnist_train = [(X.to(DEVICE), y.to(DEVICE)) for X, y in mnist_train]
    mnist_test = [(X.to(DEVICE), y.to(DEVICE)) for X, y in mnist_test]
    return mnist_train, mnist_test


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.1, 0.1)


def get_model_mlp():
    model_mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1200, bias=False), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1200, 1200, bias=False), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1200, 10))
    model_mlp.apply(init_weights)
    return model_mlp.to(DEVICE)


def accuracy(y_hat, y): 
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    num_corr, num = 0, 0
    for X, y in data_iter:
        num_corr += accuracy(net(X), y)
        num += y.numel()
    return num_corr /num


def train_epoch(net, train_iter, loss, updater): 
    loss_sum, train_acc, cnt = 0, 0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()
        loss_sum += float(l) * len(y)
        train_acc += accuracy(y_hat, y)
        cnt += y.numel()
    return loss_sum / cnt, train_acc / cnt
  

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    ret = ([], [], [])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        ret[0].append(train_loss), ret[1].append(train_acc), ret[2].append(test_acc)
        logging.info(f"Epoch {epoch}: loss = {train_loss}, train_acc = {train_acc}, test_acc = {test_acc}")
    return ret


def get_trained_model_mlp(train_data, test_data):
    if os.path.exists("model_mlp.pt"):
        model_mlp = torch.load("model_mlp.pt")
        with open("model_mlp_perf.json", "r") as f:
            perf = json.load(f)
        return model_mlp, perf
    model_mlp = get_model_mlp()
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(model_mlp.parameters(), lr=LR, momentum=MOMENTNUM)
    loss, train_acc, test_acc = train(model_mlp, train_data, test_data, loss, EPOCHS, trainer)
    perf = { "loss": loss, "train_acc": train_acc, "test_acc": test_acc }
    torch.save(model_mlp, "model_mlp.pt")
    with open("model_mlp_perf.json", "w") as f:
        json.dump(perf, f, indent=2)
    return model_mlp, perf


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (Process %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO)

    mnist_train, mnist_test = get_mnist()
    logging.info(f"train batches = {len(mnist_train)}, test batches = {len(mnist_test)}")

    model_mlp, perf = get_trained_model_mlp(mnist_train, mnist_test)
    logging.info(f"Using MLP with train_acc = {perf['train_acc'][-1]:.4f}, test_acc = {perf['test_acc'][-1]:.4f}")

    model_snn = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1200, bias=False),
        neuron.IFNode(v_threshold=V_THR, v_reset=V_RESET),
        nn.Linear(1200, 1200, bias=False),
        neuron.IFNode(v_threshold=V_THR, v_reset=V_RESET),
        nn.Linear(1200, 10)).to(DEVICE)

    with torch.no_grad():
        model_snn[1].weight = nn.Parameter(torch.clone(model_mlp[1].weight))
        model_snn[3].weight = nn.Parameter(torch.clone(model_mlp[4].weight))
        model_snn[5].weight = nn.Parameter(torch.clone(model_mlp[7].weight))

        test_sum = 0
        correct_sum = 0
        for img, label in mnist_test:
            for t in range(TIME):
                if t == 0:
                    out_spikes_counter = model_snn(img)
                else:
                    out_spikes_counter += model_snn(img)
            correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
            test_sum += label.numel()
            functional.reset_net(model_snn)
        logging.info(f"SNN test_acc = {correct_sum / test_sum:.4f}")
    