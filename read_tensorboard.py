import os
import sys
from tensorboard.backend.event_processing import event_accumulator


def load_tb_from_collection(collection_path):
    event_files = []

    for root, _, files in os.walk(collection_path):
        for f in files:
            if "tfevents" in f:
                event_files.append(os.path.join(root, f))

    if not event_files:
        raise ValueError("No TensorBoard event file found!")

    event_files.sort(key=os.path.getmtime, reverse=True)
    latest_event = event_files[0]

    print(f"Using event file: {latest_event}")

    ea = event_accumulator.EventAccumulator(
        os.path.dirname(latest_event)
    )
    ea.Reload()

    return ea


def main():
    collection_path = sys.argv[1]

    ea = load_tb_from_collection(collection_path)

    print("Available scalar tags:")
    print(ea.Tags()["scalars"])

    loss_events = ea.Scalars("lm loss") 
    loss_values = [str(e.value) for e in loss_events]

    grad_events = ea.Scalars("grad-norm")
    grad_values = [str(e.value) for e in grad_events]

    job_id = os.getenv('SLURM_JOB_ID') or "local_dev" 
    logFP = f"./control_{job_id}/eval_results.txt"

    with open(logFP, "a") as file:
        file.write(" ".join(loss_values))
        file.write("\n")
        file.write(" ".join(grad_values))
        file.write("\n")

if __name__ == "__main__":
    main()