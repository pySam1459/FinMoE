# FinMoE

This repo contains the code for "FinMoE: A Mixture of Experts Finance Language Model" paper by Samuel Barnett, supervised by Dr JingJing Deng, submitted as part of the degree of MEng Computer Science to the Board of Examiners in the Department of Computer Sciences, Durham University.

Ensure you have installed the necessary packages found in [`requirements.txt`](./requirements.txt), along with PyTorch and CUDA.

To train and run FinMoE:
1. Download the base Llama 3.2-1B model, [`llama-3.2-download.ipynb`](./llama-3.2-download.ipynb).
2. Download the datasets, [`datasets.ipynb`](./datasets.ipynb). Datasets include: `AdaptLLM/FPB`, `AdaptLLM/Headline`, `sujet-ai/Sujet-Finance-Instruct-177k`
3. Fine-tune 3 experts using the [`expert-finetune.ipynb`](./expert-finetune.ipynb) notebook
4. Train FinMoE with [`FinMoE-train.ipynb`](./FinMoE-train.ipynb) notebook, change any commented variables with custom names used
5. The experts and FinMoE can be evatuated at the bottom the the notebooks used to train them.
6. FinMoE's gating network can be evaluated using the [`gating-eval.ipynb`](./gating-eval.ipynb) notebook.
7. General models and external models can be trained and evaluated using [`general-finetune.ipynb`](./general-finetune.ipynb) and [`external-model-eval.ipynb`](./external-model-eval.ipynb)