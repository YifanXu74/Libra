# Libra: Building Decoupled Vision System on Large Language Models

This repository provides a simple implementation of Libra in PyTorch, including pretraining, finetuning, and inference.

Please refer to the ICML 2024 paper:

[**Libra: Building Decoupled Vision System on Large Language Models**](https://arxiv.org/abs/2405.10140)

Yifan Xu,  Xiaoshan Yang, Yaguang Song, Changsheng Xu

## Preparation

**Environment.** Install the required dependencies:

```
pip install -r requirements.txt
```

**DATA.** The code supports data in the webdatasets, coco,  [LLaVA-instruction](https://github.com/haotian-liu/LLaVA) formats, specifically as: 
```
DATASETS/
├── laion/
│   ├── 00000.tar
│   ├── 00001.tar
│   ├── ...
│   └── 07776.tar
├── instruction/
│   ├── llava_v1_5_mix665k.json
│   ├── data/
│   |   ├── coco/
│   |   ├── gqa/
│   |   ├── ...
│   └── └── vg
└── coco/
    ├── annotations/
    │   ├── coco_karpathy_train.json
    |   └── ...
    ├── train2017/
    ├── val2017/
    ├── train2014/
    └── ...
```
      

**CHECKPOINTS.** If you want to train Libra from scratch, several praparations are needed.

1. Prepare the huggingface version of the ``llama-2-7b-chat-hf`` model. Please refer to [here](https://huggingface.co/docs/transformers/main/model_doc/llama2). Then rename the folder name to ``llama-2-7b-chat-hf-libra``.
2. Merge the vision tokenizer weight into the pretrained llama path. The pretrained vision tokenizer weight can be found [here]().
3. Download the pretrained CLIP model in huggingface and merge it into the pretrained model paths. The CLIP model can be downloaded [here](https://huggingface.co/openai/clip-vit-large-patch14-336).

If you want to run the official Libra models, you need to download [``libra-11b-chat``](https://huggingface.co/YifanXu/libra-11b-chat) or [``libra-11b-base``](https://huggingface.co/YifanXu/libra-11b-base).

The final checkpoint path should be like:
```
CHECKPOINTS/
├── libra-11b-base/
│   ├── ...
│   └── openai-clip-vit-large-patch14-336/
│       └── ...    
├── libra-11b-chat/
│   ├── ...
│   └── openai-clip-vit-large-patch14-336/
│       └── ...    
└── llama-2-7b-chat-hf-libra/
    |
    │   # original llama files
    |
    ├── config.json
    ├── pytorch_model-00001-of-00002.bin
    ├── ...
    ├── tokenizer.model
    │   
    │   # newly added vision tokenizer
    │   
    ├── vision_tokenizer_config.yaml
    ├── vqgan.ckpt
    │
    │   # CLIP model
    │
    └── openai-clip-vit-large-patch14-336/
        └── ...    

```

## Inference

We provide a simple jupyter demo [here](demo/libra_demo.ipynb).

## Pretraining
We use the LAION dataset for pretraining.
Please refer to the [config file](libra/configs/libra_pretrain.yaml) for detailed usage. The training command is:

```
torchrun --nnodes=5 --nproc_per_node=8 train.py --cfg-path libra/configs/libra_pretrain.yaml
```

## Instruction Tuning
The code supports finetuning data in the LLaVA instruction format. Please refer to [LLaVA](https://github.com/haotian-liu/LLaVA) to organize the data.
Or you can use customized data, as long as its annonation is similar to ``llava_v1_5_mix665k.json``.

```
torchrun --nnodes=1 --nproc_per_node=8 train.py --cfg-path libra/configs/libra_instruction.yaml
```

## Model Weights
We provide the pretrained base model (Libra-Base) and the model after instruction tuning (Libra-Chat).
| Model | Url | 
| ---   | --- |
| Libra-Base | [HuggingFace](https://huggingface.co/YifanXu/libra-11b-base) |
| Libra-Chat  | [HuggingFace](https://huggingface.co/YifanXu/libra-11b-chat) |


<!-- ## Citation

If you find our work useful in your research, please consider citing:

```

``` -->




