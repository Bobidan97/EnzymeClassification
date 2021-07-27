#!/bin/sh
#$ -cwd
#$ -N GRUclassifer
#$ -l h_rt=48:00:00
#$ -pe gpu 1
#$ -l h_vmem=8G
#$ -o enzymeclassifier_GRU.o
#$ -e enzymeclassifier_GRU.e

/etc/profile.d/modules.sh
module load cuda
module load anaconda
source activate deep_learning

python3 main.py --positive_set methyltransferaseEC_2.1.1.fasta --negative_set EC_2.3andEC_2.1.4.fasta --max_seq_length 1000 --early_stopping



