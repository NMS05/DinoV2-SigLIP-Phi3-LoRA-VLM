# Prismatic SigLIP-Phi2(LoRA) VLM

This repo provides the scripts and instructions to build a custom VLM using the [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms) repository. The model details are as follows,

* **Vision Encoder** - SigLIP@384px
* **Connector** - MLP
* **Language Model** - Phi2 with LoRA
* **Train Dataset** - LLAVA + LVIS + LRV-Instruct

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
Instructions and scripts for downloading LVIS and LRV-Instruct datasets can be found in [`scripts/additional-datasets`](scripts/additional-datasets).


---

## Steps to add a new LLM (Phi2 + LoRA)
1. The [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) model from HuggingFace is added in [`prismatic/models/backbones/llm/phi2.py`](prismatic/models/backbones/llm/phi2.py). Since Phi2 is not instruction tuned, there are no constraints for a specific prompt template.
2. Next LoRA layers are added to the base LLM (phi-2) in [`prismatic/models/backbones/llm/base_llm.py`](prismatic/models/backbones/llm/base_llm.py)
3. The get_peft_model() function freezes the LLM layers and finetunes only LoRA params. Make sure to comment line-153 in [prismatic/models/vlms/prismatic.py](https://github.com/NMS05/Prismatic-SigLIP-Phi2-LoRA-VLM/blob/3a317483d1ec888395fa36158f7ff54f96b7b639/prismatic/models/vlms/prismatic.py#L153), which finetunes the entire LLM.
4. Update [`prismatic/models/backbones/llm/__init__.py`](prismatic/models/backbones/llm/__init__.py) with the new LLM.
5. Update LLM_BACKBONES registry in [`prismatic/models/materialize.py`](prismatic/models/materialize.py)
6. Finally add a new entry for your entire VLM in [prismatic/conf/models.py](https://github.com/NMS05/Prismatic-SigLIP-Phi2-LoRA-VLM/blob/3a317483d1ec888395fa36158f7ff54f96b7b639/prismatic/conf/models.py#L442). This is also where you specify the Vision Backbone, Connector type (linear or MLP), and the image resizing strategy.


Use the notebook [`scripts/understand_model.ipynb`](scripts/understand_model.ipynb) to check if everything is in order.

---

## Training

The entry point for training models is [`scripts/pretrain.py`](scripts/pretrain.py). Specify the desired model config, dataset config, stage (align or fine-tune) etc.

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py
```

<img src="https://github.com/NMS05/DinoV2-SigLIP-Phi3-LoRA-VLM/blob/main/assets/finetune_loss.png" width="500" height="250">

### Model Weights

- The weights for the "align" stage (trains MLP connector) and the "finetune" stage (requires MLP weights and trains MLP+LoRA) is available in [HuggingFace](https://huggingface.co/nms05/SigLIP_Phi2_LoRA_VLM).
- Download them to runs/
- For training hyperparameters, refer to the config files.

---

## Inference

Use the notebook [`scripts/inference.ipynb`](scripts/inference.ipynb) to chat with the model.

<img src="https://github.com/NMS05/DinoV2-SigLIP-Phi3-LoRA-VLM/blob/main/assets/test_image.jpg" width="400" height="400">

### Model Output

```
Instruction: "Provide a detailed description of the given image."

Response:

  The image features a dining table with a white plate containing a breakfast meal. The plate is filled with various food items, including eggs, toast, and orange slices. There are also a couple of sandwiches on the plate.
  In addition to the plate, there are several cups and a bottle placed on the table. A knife and a fork can be seen near the plate, ready for use. The table is surrounded by multiple chairs, with some people sitting on them, enjoying their meal. The scene appears to be a casual dining setting, with people gathered around the table to share a meal together.

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
