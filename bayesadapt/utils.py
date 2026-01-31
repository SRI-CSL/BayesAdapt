import math
import torch
import torch.nn.functional as F

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

def move_to_device(batch, device):
    new_batch = []
    for item in batch:
        try:
            new_batch.append(item.to(device))
        except:
            new_batch.append(item)
    return tuple(new_batch)

#probs is B x C
#labels is B
def brier_score(probs, labels):
    num_classes = probs.shape[1]
    target = F.one_hot(labels.long(), num_classes=num_classes).float()
    squared_diff = (probs - target) ** 2
    return torch.mean(torch.sum(squared_diff, dim=1))

def average_log_probs(sample_logits):
    B, n_samples, C = sample_logits.shape
    sample_log_probs = torch.log_softmax(sample_logits, dim=-1)
    avg_log_probs = torch.logsumexp(sample_log_probs, dim=1) - math.log(n_samples)
    return avg_log_probs

#batch is a list of length N (which is NOT the batch size)
#each list item is either a tensor is shape B x ... or a dict of such tensors, where B is the batch size
#the output should be a list of length num_chunks with each item being a list of length N
def split_batch(inputs, labels, num_chunks=1):
    chunked_inputs = {}
    for key, value in inputs.items():
        chunked_inputs[key] = torch.chunk(value, num_chunks)
    chunked_labels = torch.chunk(labels, num_chunks)
    
    split_batches = []
    for i in range(num_chunks):
        split_batch = ({key: chunked_inputs[key][i] for key in chunked_inputs}, chunked_labels[i])
        split_batches.append(split_batch)
    
    return split_batches
