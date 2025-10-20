import subprocess
import re
import pandas as pd
from time import sleep
from tqdm import tqdm

def get_sacct_data(job_id):
    """
    Run seff command for a given job_id and return parsed output as a dictionary.
    """
    command = ["seff", str(job_id)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        return parse_seff_output(output)
    except subprocess.CalledProcessError as e:
        print(f"Error executing seff: {e}")
        print(f"Stderr: {e.stderr}")
        return None


def parse_seff_output(output):
    """
    Parse seff output into a structured dictionary.
    """
    parsed = {}
    # Split by lines
    for line in output.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
    # CPU Efficiency: 83.79% of 23:59:00 core-walltime
    # mantain only the percentage
    if 'CPU Efficiency' in parsed:
        parsed['CPU Efficiency']= re.search(r'(\d+\.\d+)%', parsed['CPU Efficiency']).group(1) 

    if 'Memory Efficiency' in parsed:
        parsed['Memory Efficiency']= re.search(r'(\d+\.\d+)%', parsed['Memory Efficiency']).group(1)

    if 'Memory Utilized' in parsed:
        parsed['Memory Utilized']  = parsed['Memory Utilized'].replace('GB', '').strip()

    return parsed


# Example usage:
if __name__ == "__main__":
    job_list = []

    for i in tqdm(range(70000, 73910)):
        #sleep(0.01)  # Sleep for 0.1 seconds between requests
        job_info = get_sacct_data(str(i))
        if job_info:
            job_list.append(job_info)

    df = pd.DataFrame(job_list)
    df.to_csv("slurm_jobs_info.csv", index=False)
