for l1_coeff in 1e-5 ;do
    for hidden_size in 5080 ;do
sbatch <<EOF
#!/bin/bash
#SBATCH -J inceptionv1_mixed4a_${l1_coeff}_${hidden_size}
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="h100|a100-80gb"
#SBATCH -o /mnt/home/dheurtel/VisionMonoSemanticity/log/inceptionv1_mixed4a_${l1_coeff}_${hidden_size}.log
source ~/.bashrc
source /mnt/home/dheurtel/venv/genv_DL/bin/activate
python train.py --dataset imagenet --model_to_hook inceptionv1 --layer_name mixed4a --batch_size 1324 --channels 508 --hidden_size ${hidden_size} --l1_coeff ${l1_coeff} --lr 1e-3 --seed 0 --max_epochs 140 --gpus 1 --optimizer Adam --num_workers 8 --logger wandb --model_name inceptionv1_mixed4a_${l1_coeff}_${hidden_size}
EOF
    done
done

echo "1e-4 3e-5"