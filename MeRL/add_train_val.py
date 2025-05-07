import time
from tqdm import tqdm
import torch
import torch.nn as nn
from add_utils import overall_performance
import numpy as np
from transformers import AutoModel
import random
import pandas as pd #
import torch.nn.functional as F 

def train(epoch, model, loss_fn, optimizer, train_loader, scheduler=None):
    epoch_start_time = time.time()
    model.train()
    tr_loss = 0  # training loss in current epoch

    # ! training
    for step, batch in enumerate(tqdm(train_loader, desc='Iteration')):
        # unpack batch data
        batch = tuple(t.cuda() for t in batch) 
        inputs = batch[:-1]
        labels = batch[-1]

        out = model(inputs)
        loss = loss_fn(out, labels)
        tr_loss += loss.item()

        # back propagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # adjusting learning rate
        optimizer.zero_grad()

    timing = time.time() - epoch_start_time
    cur_lr = optimizer.param_groups[0]["lr"] 
    print(f"Timing: {timing}, Epoch: {epoch + 1}, training loss: {tr_loss}, current learning rate {cur_lr}")


def val(model, val_loader):
    # make sure to open the eval mode.
    model.eval()

    # prepare loss function
    loss_fn = nn.CrossEntropyLoss()

    val_loss = 0
    val_preds = []
    val_labels = []
    p_probs = []
    n_probs = []
    for batch in val_loader:
        # unpack batch data
        batch = tuple(t.cuda() for t in batch) 
        inputs = batch[:-1]
        labels = batch[-1]

        with torch.no_grad():
            # compute logits
            out = model(inputs)
            prob = torch.sigmoid(out) 
            p_prob = prob[:,1].cpu().detach().numpy() # 正例，类别为1
            n_prob = prob[:,0].cpu().detach().numpy() # 负例，类别为0
            # get the prediction labels
            preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()  # prediction labels [1, batch_size]
            # compute loss
            loss = loss_fn(out, labels)
            val_loss += loss.item()

            labels = labels.cpu().numpy().tolist()  # ground truth labels [1, batch_size]
            val_labels.extend(labels)
            val_preds.extend(preds)
            p_probs.extend(p_prob)
            n_probs.extend(n_prob)
    print(f"val loss: {val_loss}")
    df = pd.DataFrame({
        'val_labels': val_labels, 
        'val_preds':val_preds,
        'positive_prob': p_probs,
        'negative_prob': n_probs
    })
    df.to_csv('output.csv')
    
    # get overall performance 
    val_acc, val_prec, val_recall, val_f1 = overall_performance(val_labels, val_preds) 
    return val_acc, val_prec, val_recall, val_f1


def set_random_seeds(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_plm(args):
    # loading Pretrained Model
    plm = AutoModel.from_pretrained(args.DATA.plm) # 加载预训练的Roberta-base
    if args.DATA.use_context:
        config = plm.config 
        # 修改超参数
        config.type_vocab_size = 4 
        # 修改预训练模型
        plm.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        plm._init_weights(plm.embeddings.token_type_embeddings) 

    return plm