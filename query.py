import os
import pandas as pd
from pandasai import Agent
import configparser
import argparse


def load_config():
    """Load configuration from a file."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['PANDASAI_API_KEY']


def intialize_agent(filename):
    df = pd.read_excel(filename)
    api_key = load_config()
    os.environ["PANDASAI_API_KEY"] = f"{api_key}"
    agent = Agent(df)
    return agent


def train(filename):
    """
    Train the model with the dataset.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        Agent: The trained agent.

    """
    agent = intialize_agent(filename)
    queries = [
        "Who is the most common visitor?",
        "What time did I get the most visits?",
        "How many woman visited me in my peak time?"
    ]
    responses = [
    """
    import pandas as pd

    df = dfs[0]

    # Define columns related to visitor groups
    visitor_columns = ['Is Male', 'Is Female', 'Is Hijab', 'Is Child', 'Is Niqab']

    # Sum each column to find the most common group
    visitor_sums = df[visitor_columns].sum()

    # Find the most common visitor group
    most_common_visitor = visitor_sums.idxmax().replace('Is ', '')

    result = { "type": "text", "value": most_common_visitor }
    """,
    """
    import pandas as pd

    df = dfs[0]

    # Convert the 'Time' column to datetime if not already
    df['Time'] = pd.to_datetime(df['Time'])

    # Group by the hour to count the number of visits per hour
    df['Hour'] = df['Time'].dt.hour
    visits_per_hour = df.groupby('Hour').size()

    # Identify the hour with the maximum number of visits
    most_visits_hour = visits_per_hour.idxmax()
    most_visits_count = visits_per_hour.max()

    result = { "type": "text", "value": f"The hour with the most visits is {most_visits_hour}:00 with {most_visits_count} visits." }
    """,
    """
    import pandas as pd

    df = dfs[0]

    df['Time'] = pd.to_datetime(df['Time'])

    # Group by each minute to count the number of visits per minute
    df['Minute'] = df['Time'].dt.floor('T')
    visits_per_minute = df.groupby('Minute').size()

    # Identify the exact minute with the maximum number of visits
    peak_minute = visits_per_minute.idxmax()

    peak_minute_data = df[df['Minute'] == peak_minute]

    female_visitors_in_peak = peak_minute_data['Is Female'].sum()

    result = { "type": "text", "value": f"{female_visitors_in_peak} women visited during the peak minute at {peak_minute}." }
    """
    ]
    for query, response in zip(queries, responses):
        agent.train(queries=[query], codes=[response])
    return agent


def dataset_query(query, agent):
    """
    Perform a query on the dataset.

    Args:
        query (str): The query to be performed.

    Returns:
        str: The response from the dataset.

    """
    response = agent.chat(query)
    return response


def main():
    parser = argparse.ArgumentParser(description='Query a dataset using PandasAI.')
    parser.add_argument('query', type=str, help='The query to be performed on the dataset.')
    parser.add_argument('filename', type=str, help='The path to the CSV file.')
    args = parser.parse_args()
    
    Agent = train(args.filename)
    # Perform the query
    result = dataset_query(args.query, Agent)
    
    # Print the result
    print(result)


if __name__ == '__main__':
    main()