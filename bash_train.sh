sbatch <<EOF
#!/bin/bash
#SBATCH -J VisionMonoSemanticity
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100"
#SBATCH -o /mnt/home/dheurtel/VisionMonoSemanticity/log/inceptionv1_mixed4a_monosemantic/version_0.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
echo "Starting training" >> /mnt/home/dheurtel/VisionMonoSemanticity/log/inceptionv1_mixed4a_monosemantic/version_0.log
python train.py --dataset imagenet --model_to_hook inceptionv1 --layer_name mixed4a --batch_size 512 --channels 508 --hidden_size 5080 --l1_coeff 1e-2 --lr 1e-3 --seed 0 --max_epochs 30 --gpus 1 --optimizer Adam --num_workers 8 --logger wandb
EOF