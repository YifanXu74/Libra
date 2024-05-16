from functools import partial
import torch
import torch.utils.checkpoint

def reset_gradient_checkponinting_without_reentrant():
    notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    torch.utils.checkpoint.checkpoint = notfailing_checkpoint