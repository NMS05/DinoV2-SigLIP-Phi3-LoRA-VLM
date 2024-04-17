# Prismatic SigLIP-Phi2(LoRA) VLM

This repo provides the scripts and instructions to build a custom VLM using the [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms) repository. The model details are as follows,

* Vision Encoder - SigLIP@384px
* Connector - MLP
* Language Model - Phi2 with LoRA
* Train Dataset - LLAVA + LVIS + LRV-Instruct

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
1. The [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) model from HuggingFace is added in prismatic/models/backbones/llm/phi2.py. Since Phi2 is not instruction tuned, there are no constraints for a specific prompt template.
2. Next LoRA layers are added to the base LLM (phi-2) in prismatic/models/backbones/llm/base_llm.py
3. The get_peft_model() function freezes the LLM layers and finetunes only LoRA params. Make sure to comment line-153 in prismatic/models/vlms/prismatic.py, which finetunes the entire LLM.
4. Update prismatic/models/backbones/llm/init.py with the new LLM.
5. Update LLM registry in prismaric/models/materialize.py
6. Finally add a new entry for your entire VLM in prismatic/conf/models.py. This is also where you specify the Vision Backbone, Connector type (linear or MLP), and the image resizing strategy.


Use the notebook [`scripts/understand_model.ipynb`](scripts/understand_model.ipynb) to check if everything is in order.

---

## Training

The entry point for training models is [`scripts/pretrain.py`](scripts/pretrain.py). Specify the desired model config, dataset config, stage (align or fine-tune) etc.

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py
```

<img src="https://github.com/NMS05/Prismatic-SigLIP-Phi2-LoRA-VLM/blob/main/assets/finetune_loss.png" width="500" height="250">

### Model Weights

- The weights for the "align" stage (trains MLP connector) and the "finetune" stage (requires MLP weights and trains MLP+LoRA) is available in [HuggingFace](https://huggingface.co/nms05/SigLIP_Phi2_LoRA_VLM).
- Download them to runs/

---

## Inference

Use the notebook [`scripts/inference.ipynb`](scripts/inference.ipynb) to chat with the model.

<img src="https://github.com/NMS05/Prismatic-SigLIP-Phi2-LoRA-VLM/blob/main/assets/test_image.png" width="400" height="400">

### Model Output

```
Q: "What is written on the boat?"
A: The boat has a blue and white flag with the words "Blue Art" written on it.

Q: "How many people do you see in this picture?"
A: I see two people in this picture.

Q: "Is any of them wearing green colored dress?"
A: Yes, one of the people in the picture is wearing a green colored dress. (** Hallucination!!)
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
