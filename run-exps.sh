datasets=(
    "MNIST 3v5"
    "MNIST OvE"
    "CIFAR CarTruck"
    "CIFAR MachineAnimal"
    "STL MachineAnimal"
    "CDC-Diabetes"
    "Synthetic SCAR"
    "MNIST 3v5 SCAR"
    "MNIST OvE SCAR"
    "CIFAR CarTruck SCAR"
    "CIFAR MachineAnimal SCAR"
    "STL MachineAnimal SCAR"
    "CDC-Diabetes SCAR"
)
training_modes=(
    "VAE-PU"
)
label_frequencies=(
    0.02
    0.1
    0.3
    0.5
    0.7
    0.9
)
start_idxs=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
)

for dataset in "${datasets[@]}"; do
    for training_mode in "${training_modes[@]}"; do
        for label_frequency in "${label_frequencies[@]}"; do
            for start_idx in "${start_idxs[@]}"; do
                # echo $dataset $training_mode $label_frequency $start_idx
                
                # srun -A partial-obs --gpus=1 -p short,long,experimental conda init; conda activate vae-pu-env; python ./main.py --dataset "$dataset" --training_mode "$training_mode" --c $label_frequency --start_idx $start_idx --num_experiments 1  > "log/$dataset/$training_mode/$label_frequency/$start_idx/log.txt"

                sbatch --export=dataset="$dataset",training_mode="$training_mode",label_frequency=$label_frequency,start_idx=$start_idx experiment.sh
            done
        done
    done
done