#!/bin/bash
#SBATCH --job-name=ore_sorting     
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err     
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=16            
#SBATCH --time=00:30:00               
#SBATCH --partition=debug         
#SBATCH --mem=8G                  

module load apps/anaconda3/2024.06

source activate py3127

cd /home/yaojian/codes/jt_ore_sorting/server_scripts

python run.py