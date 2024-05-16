def initialize_system_settings_for_training():
    '''
    must be called at the very beginning of the training script
    '''
    import os
    #############################################################
    # DEBUG: if set TOKENIZERS_PARALLELISM to true (by default), an error is encountered on processing thread with FastTokenizers.
    # Error message: pyo3_runtime.PanicException: The global thread pool has not been initialized.
    # Solution: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    #############################################################
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #############################################################
    # Enable FlashAttention
    # NOTE: FlashAttention is not supported in Libra v1
    # Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
    # Modified to suit transformers==4.30.2
    # Modified from transformers.modelling_llama.LlamaFlashAttention2 (transformers==4.34.1)
    # Need to call this before importing transformers.
    # LLaMA under transformers==4.30.2 costs less memory than the one under transformers==4.34.1?
    #############################################################
    # from utils.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    # replace_llama_attn_with_flash_attn()

    #############################################################
    # Replace callbacks for recording
    # No necessary modification required now
    #############################################################
    # from utils.reset_callbacks import replace_callbacks_on_log
    # replace_callbacks_on_log()

    #############################################################
    # Reset gradient checkpointing with use_reentrant=False
    #############################################################
    from utils.reset_gradient_checkpointing import reset_gradient_checkponinting_without_reentrant
    reset_gradient_checkponinting_without_reentrant()


import torch
import logging

def setup_logger():
    import sys
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    )

class DebugModel(torch.nn.Module):
    '''
    An empty model typically used for debugging the data loading process.
    '''
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(100, 200)

    def forward(self, samples,  **kwargs):
        x = torch.randn(2, 100, device = self.model.weight.device, dtype=self.model.weight.dtype)
        x = self.model(x)
        loss = 10 - x.sum()
        return {"loss": loss}