from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_image_classifier(model, train_data, hparams):
    batch_size = hparams['batch_size_training']
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=hparams['pretrained'])

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hparams["lr"],
                                momentum=hparams["momentum"],
                                weight_decay=hparams["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    best_acc = -np.inf
    best_epoch = 0
    best_state = {}

    for epoch in tqdm(range(hparams["max_epochs"])):
        model.train()
        loss_train = 0
        correct = 0
        total = 0
        for (input, y) in train_loader:
            x, y = input.to(hparams["device"]), y.to(hparams["device"])
            optimizer.zero_grad()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum()
            total += len(y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            loss_train += loss
            optimizer.step()

        loss_train /= len(train_loader)
        acc_train = correct/total

        scheduler.step()

        if acc_train > best_acc:
            best_acc = acc_train
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

            print(f'Epoch {epoch:4}: '
                  f'loss_train: {loss_train.item():.5f}, '
                  f'acc_train: {acc_train.item():.5f} ')

        if epoch - best_epoch > early_stopping:
            print(f"early stopping at epoch {epoch}")
            break

    print('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()
