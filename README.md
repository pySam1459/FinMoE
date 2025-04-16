# FinMoE

This repo contains the code for "FinMoE: A mixture of experts finance language model" paper by Samuel Barnett, supervised by Dr JingJing Deng, submitted as part of the degree of MEng Computer Science to the Board of Examiners in the Department of Computer Sciences, Durham University.

To train and run FinMoE:
1. Download the base Llama 3.2-1B model, `llama-3.2-download.ipynb`
2. Download the datasets, `datasets.ipynb`. Datasets include: `AdaptLLM/FPB`, `AdaptLLM/Headline`, `sujet-ai/Sujet-Finance-Instruct-177k`
3. Fine-tune 3 experts using the `expert-finetune.ipynb` notebook
4. Train FinMoE with `general-finetune.ipynb` notebook, running only the cells labeled for FinMoE training as this notebook also supports training general models (a model trained on all financial tasks).
5. The experts and FinMoE can be evatuated at the bottom the the notebooks used to train them.
6. FinMoE's gating network can be evaluated using the `gating-eval.ipynb` notebook.