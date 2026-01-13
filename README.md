# MiLaSol  
**Modeling Protein Solubility by Mixing Up Multiple Protein Language Models**

MiLaSol is a deep learning framework for protein solubility prediction that integrates multiple protein language models (PLMs) to learn complementary sequence representations. The model eliminates reliance on handcrafted protein engineering features and achieves strong, balanced performance on both soluble and insoluble proteins.

---

## 🚀 Key Features

- **Multi-PLM integration**: Combines representations from ESM, ProtT5, and RayGun  
- **End-to-end learning**: No handcrafted biophysical features required  
- **Advanced supervision**:
  - Binary classification loss  
  - Contrastive learning  
  - Reconstruction and regularization losses (entropy, prototype, center loss)  
- **Robust performance**: High accuracy and MCC with balanced sensitivity  
- **Extensible design**: Easy to plug in new embeddings or loss components  

---

## 🧠 Model Overview

MiLaSol learns protein representations by fusing multiple PLM embeddings and optimizing a composite objective that balances classification accuracy with representation structure.

The training objective integrates:
- Supervised classification loss  
- Contrastive loss for feature discrimination  
- Reconstruction loss for representation preservation  
- Regularization via entropy, prototype, and center losses  

Learnable weights dynamically balance each component during training.


## ⚙️ Installation

```bash
git clone git@github.com:weiweiloutufts/milasol.git
cd milasol
conda create -n milasol python=3.10
conda activate milasol
pip install -r src/requirements.txt
```
Note: A GPU with CUDA support is strongly recommended.

## Training
Example prediction command:

```bash
python -m milasol.models.predict_new \
  --modelname checkpoints/daps_train0.3_0.1_0.1_0.0_64_2.5_1.0.pth \
  --out_dir outputs/ \
  --cache_dir /modelcache/ \
  --label_file /datasets/deepsol_data/test_tgt.txt \
  --sequence_file /datasets/deepsol_data/test_src.txt \
  --esm_file /datasets/deepsol_data/test_src_esm_embeddings.csv \
  --prot_file /datasets/deepsol_data/test_src_prot_embeddings.csv \
  --ray_file /datasets/deepsol_data/test_src_raygun_embeddings_v2.csv \
```
Example training command:
```bash
python -m milasol.models.train \
--batch_size $batch_size \
--kernel_size $kernel_size \
--num_filters $num_filters \
--lstm_hidden_dim $lstm_hidden_dim \
--num_lstm_layers $num_lstm_layers \
--learning_rate $learning_rate \
--latent_dim $latent_dim \
--contrastive_weight $contrastive_weight \
--rec_loss_weight $rec_loss_weight \
--entropy_weight $entropy_weight \
--triplet_weight $triplet_weight \
--proto_weight $proto_weight \
--pos_rate $pos_rate \
```
Acknowledgements
This work was developed within the Cowen Lab at Tufts University.