#!/bin/bash
#SBATCH -J dapspredict   # Job name
#SBATCH --time=00-12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
# --constraint="a100-40G"
#SBATCH --mem=64G
#SBATCH --output=logs/%j.predict.%A_%a.out
#SBATCH --error=logs/%j.predict.%A_%a.err   # Error file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weiwei.lou@tufts.edu


# Load required modules



module purge  
module load cuda/11.0
module load anaconda/2024.10
module load ngc/1.0


module load pytorch/2.5.1-cuda12.1-cudnn9 



set -euxo pipefail
export PYTHONUNBUFFERED=
/cluster/tufts/cowenlab/wlou01/condaenv/biosolu/bin/python -m milasol.models.predict_pre \
  --modelname /cluster/tufts/cowenlab/wlou01/collect/best_model_v2.pth \
  --source_dir /cluster/tufts/cowenlab/wlou01/datasets/deepsol_data/ \
  --out_dir outputs/ \
  --cache_dir /cluster/tufts/cowenlab/wlou01/modelcache/ \
  --device cuda:0
  
