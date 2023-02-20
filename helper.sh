#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0 cuda/11.6.2-gcc/9.5.0 cudnn/8.1.0.77-11.2-gcc/9.5.0
source activate parallel
cd "$2" || exit
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint="$1":3000  "$3" -e 5 -p single_node_snapshot.pth
echo "$HOSTNAME" finished tasks