import subprocess
from time import sleep, strftime

def run_sbatch(train_command, train_name, hours, killable, gpu_memory, cpu_memory=64, task_name="sbatch"):
    sbatch_text = f"#!/bin/csh\n#SBATCH --time={hours}:0:0" \
                   f"\n#SBATCH --mem={cpu_memory}g" \
                   f"\n#SBATCH -c 5" \
                   f"\n#SBATCH --mail-type=END" \
                   f"\n#SBATCH --mail-user=ariel1" \
                   f"\n#SBATCH --output=/cs/labs/yweiss/ariel1/cluster_runs/{train_name}.out"
    if gpu_memory > 0:
        sbatch_text += f"\n#SBATCH --gres=gpu:1,vmem:{gpu_memory}g"
    if killable:
        sbatch_text += "\n#SBATCH --killable"
    sbatch_text += f"\nsource /cs/labs/yweiss/ariel1/venv/bin/activate.csh" \
                   f"\ncd /cs/labs/yweiss/ariel1/repos/In_defence_of_wasserstein" \
                   f"\n{train_command} "
    if train_name is not None:
        sbatch_text += f" --train_name {train_name}"  # _{strftime('%m-%d_T-%H:%M:%S')}"

    print("writing")
    f = open(f"{task_name}.csh", "w")
    f.write(sbatch_text)
    f.close()
    subprocess.Popen(["cat", f"{task_name}.csh"])
    subprocess.Popen(["sbatch", f"{task_name}.csh"])
    sleep(2)
    # subprocess.Popen(["rm", "-rf", f"{task_name}.csh"])

