import os
from pandasai import Agent
import pandas as pd
import configparser
import argparse




def load_config():
    """Load configuration from a file."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['PANDASAI_API_KEY']



def dataset_query(query, filename='Intern-NLP-Dataset.pkl'):
    """
    Perform a query on the dataset.

    Args:
        query (str): The query to be performed.

    Returns:
        str: The response from the dataset.

    """
    df = pd.read_pickle(filename)
    api_key = load_config()
    os.environ["PANDASAI_API_KEY"] = f"{api_key}"
    agent = Agent(df)
    response = agent.chat(query)
    return response


def main():
    parser = argparse.ArgumentParser(description='Query a dataset using PandasAI.')
    parser.add_argument('query', type=str, help='The query to be performed on the dataset.')
    args = parser.parse_args()
    
    # Perform the query
    result = dataset_query(args.query)
    
    # Print the result
    print(result)


if __name__ == '__main__':
    main()