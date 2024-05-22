# DinoV2-SigLIP-Phi3(LoRA) VLM

This repo provides the scripts and instructions to build a custom VLM using the [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms) repository. The model details are as follows,

* **Vision Encoder** - DinoV2 + SigLIP @384px resolution. [Why 2 vision encoders?](https://arxiv.org/abs/2401.06209)
* **Connector** - MLP (Dino and SigLIP features are concatenated and then projected to Phi3 representation space)
* **Language Model** - Phi3 + LoRA
* **Training Dataset** - LLAVA + LRV-Instruct

---

## Installation

Clone this repo and follow the installation instructions [here](https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file#installation). Additionally run the following.

```bash
pip uninstall transformers

pip install git+https://github.com/huggingface/transformers

pip install git+https://github.com/huggingface/peft
```


---

## Download pre-training Datasets

```bash
python scripts/preprocess.py --dataset_id "llava-laion-cc-sbu-558k" --root_dir training_data/

python scripts/preprocess.py --dataset_id "llava-v1.5-instruct" --root_dir training_data/
```
Instructions and scripts for downloading LRV-Instruct datasets can be found in [`scripts/additional-datasets`](scripts/additional-datasets).


---

## Steps to add a new LLM (Phi3 + LoRA)
1. **LLM and LoRA Config:** The [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model from HuggingFace is added in [`prismatic/models/backbones/llm/phi3.py`](prismatic/models/backbones/llm/phi3.py). The LoRA configuration is also specified here.
2. **Instruction Template:** Phi3 is intruction tuned and follows a specific prompt template [`prismatic/models/backbones/llm/prompting/phi3_chat_prompter.py`](prismatic/models/backbones/llm/prompting/phi3_chat_prompter.py).
3. **LoRA:** From the LoRA configuration in 1, the LoRA layers are added to the base LLM (phi-3) using the [HuggingFace PEFT](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods) library in [`prismatic/models/backbones/llm/base_llm.py`](prismatic/models/backbones/llm/base_llm.py)
4. **Freeze LLM Params:** The get_peft_model() function freezes the LLM layers and finetunes only LoRA params. Make sure to comment line-153 in [`prismatic/models/vlms/prismatic.py`](prismatic/models/vlms/prismatic.py), which finetunes the entire LLM.
5. **Update Entries:** Update [`prismatic/models/backbones/llm/__init__.py`](prismatic/models/backbones/llm/__init__.py) with the new LLM.
6. **Update Entries:** Update LLM_BACKBONES registry in [`prismatic/models/materialize.py`](prismatic/models/materialize.py)
7. **Update Entries:** Finally add a new entry for your entire VLM in [`prismatic/conf/models.py`](prismatic/conf/models.py). This is also where you specify the Vision Backbone, Connector type (linear or MLP), and the image resizing strategy.

---

## Training

The entry point for training models is [`scripts/pretrain.py`](scripts/pretrain.py). Specify the desired model config, dataset config, stage (align or fine-tune) etc.

**Note:** set enable_peft = False in [`prismatic/models/backbones/llm/phi3.py`](prismatic/models/backbones/llm/phi3.py) (line 63), for "align" stage training.

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py
```

<img src="https://github.com/NMS05/DinoV2-SigLIP-Phi3-LoRA-VLM/blob/main/assets/loss_curve.png" width="500" height="250">

### Model Weights

- The weights for the "align" stage (trains MLP connector) and the "finetune" stage (requires MLP weights and trains MLP+LoRA) is available in [HuggingFace](https://huggingface.co/nms05/Dinov2-SigLIP-Phi3-LoRA/tree/main).
- Download them to runs/
- For training hyperparameters, refer to the config files.

---

## Inference

Run [`scripts/generate.py`](scripts/generate.py) to chat with the model via terminal.

<img src="https://github.com/NMS05/DinoV2-SigLIP-Phi3-LoRA-VLM/blob/main/assets/test_image.jpg" width="400" height="400">

### Model Output

```
Instruction: "Provide a detailed description of the given image."
Response:
The image features a dining table with a white plate containing a breakfast meal. The plate is filled with various food items, including eggs, toast, and orange slices.
There are also a couple of sandwiches on the plate. In addition to the plate, there are several cups and a bottle placed on the table. A knife and a fork can be seen near the plate, ready for use.
The table is surrounded by multiple chairs, with some people sitting on them, enjoying their meal. The scene appears to be a casual dining setting, with people gathered around the table to share a meal together.

```


---
## Citation 

```bibtex
@article{karamcheti2024prismatic,
  title = {Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models},
  author = {Siddharth Karamcheti and Suraj Nair and Ashwin Balakrishna and Percy Liang and Thomas Kollar and Dorsa Sadigh},
  journal = {arXiv preprint arXiv:2402.07865},
  year = {2024},
}
```
