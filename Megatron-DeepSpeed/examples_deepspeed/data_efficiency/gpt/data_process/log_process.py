import re
import ast
import pprint

def extract_snapshot_data(log_file):
    # Step 1: Read the log file
    with open(log_file, 'r') as file:
        log_data = file.read()

    # Step 2: Define regex patterns for both types of strings
    pattern_snapshot = re.compile(
        r'Iteration (?P<iteration>\d+) '
        r'dp_(?P<dp>\d+)_pp_(?P<pp>\d+)_tp_(?P<tp>\d+) '
        r'snapshot time: (?P<snapshot_time>[0-9.]+), '
        r'snapshot_size: (?P<snapshot_size>[0-9.]+) MB, '
        r'snapshot_speed: (?P<snapshot_speed>[0-9.]+) MB/s'
    )

    pattern_throughput = re.compile(
        r'dp_(?P<dp>\d+)_pp_(?P<pp>\d+)_tp_(?P<tp>\d+) '
        r'throughput_metrics: (?P<throughput_metrics>\{.*?\})'
    )

    # Step 3: Find all matches for both patterns
    matches_snapshot = list(pattern_snapshot.finditer(log_data))
    matches_throughput = list(pattern_throughput.finditer(log_data))

    # Step 4: Extract data and store in a dictionary
    snapshot_data = {}
    
    # Process snapshot matches
    for match in matches_snapshot:
        dp = match.group('dp')
        pp = match.group('pp')
        tp = match.group('tp')
        iteration = int(match.group('iteration'))
        key = f'dp_{dp}_pp_{pp}_tp_{tp}'

        if key not in snapshot_data:
            snapshot_data[key] = {}

        snapshot_data[key][iteration] = {
            'snapshot_time': float(match.group('snapshot_time')),
            'snapshot_size': float(match.group('snapshot_size')),
            'snapshot_speed': float(match.group('snapshot_speed'))
        }

    # Process throughput matches
    for match in matches_throughput:
        dp = match.group('dp')
        pp = match.group('pp')
        tp = match.group('tp')
        throughput_metrics_str = match.group('throughput_metrics')

        try:
            throughput_metrics = ast.literal_eval(throughput_metrics_str)
            iteration = throughput_metrics['throughput/iteration']
            key = f'dp_{dp}_pp_{pp}_tp_{tp}'

            if key in snapshot_data and iteration in snapshot_data[key]:
                snapshot_data[key][iteration].update({
                    'iteration_time': throughput_metrics['throughput/iteration-time'],
                    'samples_per_sec': througthput_metrics['throughput/samples_per_sec'],
                    'tflops': throughput_metrics['throughput/tflops'],
                })
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing throughput metrics: {e}")
            print(f"Throughput metrics string: {throughput_metrics_str}")

    return snapshot_data
# Example usage:
log_file_path = '/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/output/log/2024.05.21_23.58/2024.05.21_23.58_gpu1-12.log'
snapshot_data = extract_snapshot_data(log_file_path)
pprint.pprint(snapshot_data)