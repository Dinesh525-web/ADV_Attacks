import pandas as pd

def save_results(results, filename="attack_results.csv"):
    """Saves attack results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
