from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
def get_ea(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc
def extract_csv_from_tensorboard(log_dir, tag, output_csv):
    event_acc = get_ea(log_dir)
    # Load the TensorBoard events

    # Extract the scalar values for the specified tag
    scalar_events = event_acc.Scalars(tag)

    # Convert the events to a DataFrame
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    df = pd.DataFrame({'step': steps, 'value': values})

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

# Example usage
import os

log_dir = '/home/test/test01/sa/workspace/InternEvo/tensorboard/llama13b_32xgpu'

res = get_ea(log_dir)
from IPython import embed;embed()
# extract_csv_from_tensorboard(log_dir, tag, output_csv)

