import os
import pandas as pd
from pandasai import Agent
import configparser
import argparse
from pandasai.ee.vectorstores import ChromaDB


def time_with_most_visits(filename):
    df = pd.read_excel(filename)
    df['Time'] = pd.to_datetime(df['Time'])

    # Set 'timestamp' as the index
    df.set_index('Time', inplace=True)

    # Resample or aggregate data by time intervals (e.g., per minute)
    visitor_counts = df.resample('T').size()  # 'T' stands for minute

    # Find the time with the most visitors
    peak_time = visitor_counts.idxmax()
    peak_visitors = visitor_counts.max()

    print(f"The time with the most visitors is {peak_time} with {peak_visitors} visitors.")

def women_in_peak_time(filename):
    df = pd.read_excel(filename)
    # Convert the 'Time' column to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Ensure 'Is Female' is a boolean or integer
    df['Is Female'] = df['Is Female'].astype(int)

    # Set 'Time' as the index
    df.set_index('Time', inplace=True)

    # Resample by minute to count visitors per minute
    visitor_counts = df.resample('T').size()

    # Find the time with the most visitors
    peak_time = visitor_counts.idxmax()

    # Filter the data for the peak minute
    peak_time_data = df[df.index == peak_time]

    # Count the number of female visitors during the peak minute
    female_count = peak_time_data['Is Female'].sum()
    
    print(f"Number of female visitors during the peak minute ({peak_time}): {female_count}")

def most_common_visitor(filename):
    df= pd.read_excel(filename)
    # Count the number of visitors
    columns = ['Is Male', 'Is Female', 'Is Hijab', 'Is Child', 'Is Niqab']

    # Sum the values in the specified columns
    count = df[columns].sum()

    # Find the column with the maximum number of 1s
    max_column = count.idxmax()

    print(f"The most common visitor is '{max_column}'")

def load_config():
    """Load configuration from a file."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']['PANDASAI_API_KEY']


def intialize_agent(filename):
    api_key = load_config()
    os.environ["PANDASAI_API_KEY"] = f"{api_key}"
    vectorstore=ChromaDB()
    agent = Agent(filename, vectorstore=vectorstore)
    return agent


def train(agent):
    queries = ["What time did I get the most visits?",
               "How many woman visited me in my peak time?",
               "who is my most common visitor?"
         ]
    responses = [
        """
    import pandas as pd

    df = dfs[0]

    # Set 'Time' as the index
    df.set_index('Time', inplace=True)

    # Resample or aggregate data by time intervals (e.g., per minute)
    visitor_counts = df.resample('T').size()  # 'T' stands for minute

    # Find the time with the most visitors
    peak_time = visitor_counts.idxmax()

    # Debugging output
    print(f"Peak time before formatting: {peak_time}")

    # Return only the time portion in 'HH:MM:SS' format
    result = { "type": "string", "value": peak_time.strftime('%H:%M:%S') }
    """,
    """
    df['Time'] = pd.to_datetime(df['Time'])

    # Set 'Time' as the index
    df.set_index('Time', inplace=True)

    # Resample by minute to count visitors per minute
    visitor_counts = df.resample('T').size()

    # Find the time with the most visitors
    peak_time = visitor_counts.idxmax()

    # Filter the data for the peak minute
    peak_time_data = df[df.index == peak_time]

    # Count the number of female visitors during the peak minute
    female_count = peak_time_data['Is Female'].sum()
    result = { "type": "string", "value": female_count }
    """,
    """
    import pandas as pd
    df = dfs[0]
    columns = ['Is Male', 'Is Female', 'Is Hijab', 'Is Child', 'Is Niqab']

    # Sum the values in the specified columns
    count = df[columns].sum()

    # Find the column with the maximum number of 1s
    max_column = count.idxmax()

    result = { "type": "string", "value": f" My most common visitor is {max_column}" }
    """
    ]
    for query, response in zip(queries, responses):
        agent.train(queries=[query], codes=[response])
    return agent
    

def main():
    parser = argparse.ArgumentParser(description='Query a dataset using PandasAI.')
    parser.add_argument('query', type=str, help='The query to be performed on the dataset.')
    parser.add_argument('filename', type=str, help='The path to the CSV file.')
    args = parser.parse_args()

    # time_with_most_visits(args.filename)
    # women_in_peak_time(args.filename)
    # most_common_visitor(args.filename)
    agent = intialize_agent(args.filename)
    agent = train(agent)
    # Perform the query
    result = agent.chat(args.query)
    # Print the result
    print(result)



if __name__ == '__main__':
    main()