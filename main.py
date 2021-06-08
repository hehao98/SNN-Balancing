import os
import json
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from spikingjelly.clock_driven import neuron, functional


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH_SIZE, LR, MOMENTNUM = 100, 100, 0.1, 0.5
TIME, V_THR, V_RESET = 10000, 1.0, 0.0


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
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.uniform_(m.weight, -0.1, 0.1)


def get_model_mlp():
    model_mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1200, bias=False), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1200, 1200, bias=False), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1200, 10, bias=False))
    model_mlp.apply(init_weights)
    return model_mlp.to(DEVICE)


def get_model_cnn():
    model_cnn = nn.Sequential(
        nn.Conv2d(1, 12, kernel_size=5, bias=False), nn.AvgPool2d(2), nn.ReLU(), nn.Dropout(0.1), 
        nn.Conv2d(12, 64, kernel_size=5, bias=False), nn.AvgPool2d(2), nn.ReLU(), nn.Dropout(0.1),
        nn.Flatten(),
        nn.Linear(1024, 10, bias=False))
    model_cnn.apply(init_weights)
    return model_cnn.to(DEVICE)


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


def get_trained_model(train_data, test_data, model_type):
    if os.path.exists(f"model_{model_type}.pt"):
        model = torch.load(f"model_{model_type}.pt")
        with open(f"model_{model_type}_perf.json", "r") as f:
            perf = json.load(f)
        return model, perf
    if model_type == "mlp":
        model = get_model_mlp()
    elif model_type == "cnn":
        model = get_model_cnn()
    else:
        raise ValueError("model_type should be mlp or cnn")
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTNUM)
    loss, train_acc, test_acc = train(model, train_data, test_data, loss, EPOCHS, trainer)
    perf = { "loss": loss, "train_acc": train_acc, "test_acc": test_acc }
    torch.save(model, f"model_{model_type}.pt")
    with open(f"model_{model_type}_perf.json", "w") as f:
        json.dump(perf, f, indent=2)
    return model, perf


def get_snn_mlp_model(v_thr=V_THR, v_reset=V_RESET):
    model_snn = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1200, bias=False),
        neuron.IFNode(v_threshold=v_thr, v_reset=v_reset),
        nn.Linear(1200, 1200, bias=False),
        neuron.IFNode(v_threshold=v_thr, v_reset=v_reset),
        nn.Linear(1200, 10, bias=False))
    return model_snn.to(DEVICE)


def get_snn_cnn_model(v_thr=V_THR, v_reset=V_RESET):
    model_snn = nn.Sequential(
        nn.Conv2d(1, 12, kernel_size=5, bias=False), nn.AvgPool2d(2), 
        neuron.IFNode(v_threshold=v_thr, v_reset=v_reset),
        nn.Conv2d(12, 64, kernel_size=5, bias=False), nn.AvgPool2d(2), 
        neuron.IFNode(v_threshold=v_thr, v_reset=v_reset),
        nn.Flatten(),
        nn.Linear(1024, 10, bias=False))
    return model_snn.to(DEVICE)


def baseline_copy(layer: nn.Linear or nn.Conv2d):
    return nn.Parameter(torch.clone(layer.weight))


def model_norm(layer: nn.Linear or nn.Conv2d):
    weight = torch.clone(layer.weight)
    max_pos_input = 0
    for neuron in weight:
        input_sum = torch.maximum(neuron, torch.zeros_like(neuron)).sum()
        #for input_wt in neuron:
        #    input_sum += max(0, float(input_wt))
        max_pos_input = max(max_pos_input, input_sum)
    weight = weight / max_pos_input
    logging.info(f"Scaled layer {layer} by factor {1/max_pos_input:.4f}")
    return nn.Parameter(weight)


def data_norm(layer: nn.Linear, factor: float = 1.0):
    """TODO: Implement"""
    return nn.Parameter(torch.clone(layer.weight))


def eval_snn(model_snn, test_data):
    test_sum = [0] * TIME
    correct_sum = [0] * TIME
    for img, label in tqdm(test_data):
        for t in range(TIME):
            if t == 0:
                out_spikes_counter = model_snn(img)
            else:
                out_spikes_counter += model_snn(img)
            correct_sum[t] += (out_spikes_counter.max(1)[1] == label).float().sum().item()
            test_sum[t] += label.numel()
        functional.reset_net(model_snn)
    for i in range(0, TIME, TIME // 10):
        logging.info(f"SNN t = {i}:　test_acc = {correct_sum[i] / test_sum[i]:.4f}")
    return [x / y for x, y in zip(correct_sum, test_sum)]


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (Process %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO)

    mnist_train, mnist_test = get_mnist()
    logging.info(f"train batches = {len(mnist_train)}, test batches = {len(mnist_test)}")

    model_mlp, perf = get_trained_model(mnist_train, mnist_test, "mlp")
    logging.info(f"Using MLP with train_acc = {perf['train_acc'][-1]:.4f}, test_acc = {perf['test_acc'][-1]:.4f}")

    model_cnn, perf = get_trained_model(mnist_train, mnist_test, "cnn")
    logging.info(f"Using CNN with train_acc = {perf['train_acc'][-1]:.4f}, test_acc = {perf['test_acc'][-1]:.4f}")

    with torch.no_grad():
        logging.info("MLP copy weight directly:")
        model_snn = get_snn_mlp_model(v_thr=4.0)
        model_snn[1].weight = baseline_copy(model_mlp[1])
        model_snn[3].weight = baseline_copy(model_mlp[4])
        model_snn[5].weight = baseline_copy(model_mlp[7])
        acc_baseline = eval_snn(model_snn, mnist_test)

        logging.info("MLP copy weight with model based normalization:")
        model_snn = get_snn_mlp_model()
        model_snn[1].weight = model_norm(model_mlp[1])
        model_snn[3].weight = model_norm(model_mlp[4])
        model_snn[5].weight = model_norm(model_mlp[7])
        acc_model_norm = eval_snn(model_snn, mnist_test)

        logging.info("MLP copy weight with data based normalization:")
        model_snn = get_snn_mlp_model()
        model_snn[1].weight = data_norm(model_mlp[1])
        model_snn[3].weight = data_norm(model_mlp[4])
        model_snn[5].weight = data_norm(model_mlp[7])
        acc_data_norm = eval_snn(model_snn, mnist_test)
        
        logging.info("CNN copy weight directly:")
        model_snn = get_snn_cnn_model(v_thr=20.0)
        model_snn[0].weight = baseline_copy(model_cnn[0])
        model_snn[3].weight = baseline_copy(model_cnn[4])
        model_snn[7].weight = baseline_copy(model_cnn[9])
        acc_baseline2 = eval_snn(model_snn, mnist_test)

        logging.info("CNN copy weight with model based normalization:")
        model_snn = get_snn_cnn_model()
        model_snn[0].weight = model_norm(model_cnn[0])
        model_snn[3].weight = model_norm(model_cnn[4])
        model_snn[7].weight = model_norm(model_cnn[9])
        acc_model_norm2 = eval_snn(model_snn, mnist_test)

        logging.info("CNN copy weight with data based normalization:")
        model_snn = get_snn_cnn_model()
        model_snn[0].weight = data_norm(model_cnn[0])
        model_snn[3].weight = data_norm(model_cnn[4])
        model_snn[7].weight = data_norm(model_cnn[9])
        acc_data_norm2 = eval_snn(model_snn, mnist_test)

    with open("result.json", "w") as f:
        json.dump({
            "mlp": { "acc_baseline": acc_baseline, "acc_model_norm": acc_model_norm, "acc_data_norm": acc_data_norm },
            "cnn": { "acc_baseline": acc_baseline2, "acc_model_norm": acc_model_norm2, "acc_data_norm": acc_data_norm2 },
        }, f, indent=2)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot([100 * (1 - x) for x in acc_baseline], label="MLP Baseline (Direct Copy, v_thr = 4.0)", 
            linestyle=":", linewidth=2, color="tab:blue")
    ax.plot([100 * (1 - x) for x in acc_model_norm], label="MLP Model-Based Normalization",
            linestyle="--", linewidth=2, color="tab:green")
    ax.plot([100 * (1 - x) for x in acc_data_norm], label="MLP Data-Based Normalization", 
            linestyle="-.", linewidth=2, color="tab:cyan")
    ax.plot([100 * (1 - x) for x in acc_baseline2], label="CNN Baseline (Direct Copy, v_thr = 20.0)", 
            linestyle=":", linewidth=2, color="tab:red")
    ax.plot([100 * (1 - x) for x in acc_model_norm2], label="CNN Model-Based Normalization", 
            linestyle="--", linewidth=2, color="tab:orange")
    ax.plot([100 * (1 - x) for x in acc_data_norm2], label="CNN Data-Based Normalization", 
            linestyle="-.", linewidth=2, color="tab:pink")
    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_yscale("log")
    ax.set_ylabel("Test Error (%)")
    ax.set_title(f"Performance Overview\n"
              f"MLP/CNN Training: epochs = {EPOCHS}, batch = {BATCH_SIZE}, lr = {LR}, momentum = {MOMENTNUM}\n"
              f"SNN Setup: timesteps = {TIME}, v_thr = {V_THR}, v_reset = {V_RESET}")
    ax.legend()
    fig.savefig("result.png", bbox_inches="tight", dpi=300)

    logging.info("Finish!")
