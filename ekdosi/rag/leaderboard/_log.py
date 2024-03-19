import json
import pandas as pd
from ekdosi.rag.leaderboard import LeaderboardEntrySchema
import pathlib
from typing import Union


def log_entry(entry: LeaderboardEntrySchema, file_path: str = "leaderboard_log.txt"):
    """
    Logs a LeaderboardEntrySchema instance to a specified text file

    Args:
        entry (LeaderboardEntrySchema): The leaderboard entry to log.
        file_path (str): The path to the log file. Defaults to "leaderboard_log.txt".

    """
    entry_dict = entry.model_dump_json()
    with open(file_path, "a") as file:
        file.write(json.dumps(entry_dict) + "\n")


def read_log_as_dataframe(
    file_path: Union[pathlib.Path, str], sort_by_total_score: bool = False
) -> pd.DataFrame:
    """
    Reads the log file and returns it as a pandas dataframe. Optionally sorts the dataframe by total_score in descending order.

    Args:
        file_path (str): The path to the log file.
        sort_by_total_score (bool): Whether to sort the dataframe by total_score in descending order. Defaults to False.

    Returns:
        pd.DataFrame: A pandas dataframe containing all the entries in the log file.
    """
    with open(file_path, "r") as file:
        log_entries = [
            LeaderboardEntrySchema(**json.loads((json.loads(line)))) for line in file
        ]
    df = pd.DataFrame([entry.model_dump() for entry in log_entries])
    if sort_by_total_score:
        df = df.sort_values(by="total_score", ascending=False)
    return df


def test_log_entries():
    """
    Tests logging records and retrieving the results, sorted by total_score.

    Returns:
        pd.DataFrame: A pandas dataframe containing the logged entries, sorted by total_score.
    """
    test_file_path = pathlib.Path("test_leaderboard_log.txt")
    entries = [
        LeaderboardEntrySchema(
            team_name=f"Team {i}", vectordb="FAISS", ragas_score=95.5 + i
        )
        for i in range(5)
    ]
    entries_w_description = [
        LeaderboardEntrySchema(
            team_name=f"Team {i}",
            vectordb="FAISS",
            ragas_score=95.5 + i,
            description=f"test {i}",
        )
        for i in range(5)
    ]
    entries.extend(entries_w_description)

    for entry in entries:
        log_entry(entry, test_file_path)

    df = read_log_as_dataframe(test_file_path, sort_by_total_score=True)

    # Clean up by deleting the test file
    test_file_path.unlink()

    return df
