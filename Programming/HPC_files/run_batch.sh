#!/usr/bin/env bash
#SBATCH --job-name=test                         # name of job
#SBATCH --time=0-20:20:00                               # maximum time until job is cancelled
#SBATCH --ntasks=1                                      # number of tasks
#SBATCH --cpus-per-task=10                             # number of cpus requested
#SBATCH --mem-per-cpu=7G                               # memory per cpu requested
#SBATCH --mail-type=begin                               # send mail when job begins
#SBATCH --mail-type=end                                 # send mail when job ends
#SBATCH --mail-type=fail                                # send mail if job fails
#SBATCH --mail-user=agriesel@uos.de              # email of user
#SBATCH --output=/home/student/a/agriesel/logs/job-%x-%j.out    # output file of stdout messages
#SBATCH --error=/home/student/a/agriesel/logs/job-%x-%j.err     # output file of stderr messages
#SBATCH --nice=1                                        # higher number for lower priority 1-10000

# usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 1;}
# [ $# -eq 0 ] && usage

# while getopts ":c:s:b:" o; do
#     case "${o}" in
#         c) # case-study
#             CASE=${OPTARG}
#             ;;
#         s) # case-study
#             SCENARIO=${OPTARG}
#             ;;
#         b) # password
#             BACKEND=${OPTARG}
#             ;;
#         h | *) # display help
#             usage
#             exit 1
#             ;;
#     esac
# done


echo "Requested $SLURM_CPUS_PER_TASK cores."
echo "Starting script..."

# activate conda environment
source activate eai

# this launches inference
# in python script access "SLURM_CPUS_PER_TASK" environmental 
cd /home/student/a/agriesel/Python/
echo $PWD
srun python main.py --dataset "vkitti" --complexity "2D"
#srun python main.py --dataset "kitti" --complexity "2D"


echo "Finished cool alina script."
