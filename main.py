import os
from pandasai import Agent
import pandas as pd
import configparser



def load_config():
    """Load configuration from a file."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['PANDASAI_API_KEY']


def dataset_query(query):
    """
    Perform a query on the dataset.

    Args:
        query (str): The query to be performed.

    Returns:
        str: The response from the dataset.

    """
    df = pd.read_excel('Intern-NLP-Dataset.xlsx')
    api_key = load_config()
    os.environ["PANDASAI_API_KEY"] = f"{api_key}"
    agent = Agent(df)
    response = agent.chat(query)
    return response


if __name__ == '__main__':
    query = input('Enter your query: ')
    print(dataset_query(query))i