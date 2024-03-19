from pydantic import BaseModel, Field, field_validator, computed_field
import uuid
from datetime import datetime

__all__ = ["LeaderboardEntrySchema"]


class LeaderboardEntrySchema(BaseModel):
    """
    A Pydantic model representing an entry in the leaderboard.

    Attributes:
        team_name (str): The name of the team.
        process_id (str): A unique identifier for the process, defaults to a random UUID as a string.
        vectordb (str): vector database used.
        ragas_score (float): The RAGAS score of the process.
        latency (float, optional): The latency of the process, defaults to 0.0.
        cost (float, optional): The cost associated with the process, defaults to 0.0.
        description (str, optional): A description of the process, defaults to None.
        created_at (datetime): The ISO 8601 datetime when the entry was created, defaults to the current datetime.
    """

    team_name: str = Field(frozen=True)
    process_id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), frozen=True)
    vectordb: str
    ragas_score: float
    latency: float = Field(default=0.0)
    cost: float = Field(default=0.0)
    description: str = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now().isoformat(), frozen=True
    )

    @field_validator("vectordb")
    @classmethod
    def validate_vectordb(cls, v: str) -> str:
        """
        Validates that the vectordb is in the set of fields and sets the name to lowercase.
        """
        available_vector_db = [
            "milvus",
            "qdrant",
            "faiss",
            "chroma",
            "pinecone",
            "weaviate",
            "opensearch",
            "elasticsearch",
            "redis",
        ]

        if v.lower() not in available_vector_db:
            raise ValueError("Database not found in available set")
        return v.lower()

    @computed_field
    def total_score(self) -> float:
        """
        Calculate the total score for the leaderboard
        """
        return self.ragas_score - self.cost - self.latency


def test_leaderboard_entry():
    """
    Tests the creation of a LeaderboardEntry instance.
    """
    entry_data = {
        "team_name": "Team Ragatat",
        "ragas_score": 95.5,
        "vectordb": "FAISS",
    }
    entry = LeaderboardEntrySchema(**entry_data)
    assert entry.team_name == "Team Ragatat"
    assert entry.vectordb == "faiss"
    assert "total_score" in entry.model_computed_fields
    assert isinstance(entry.created_at, str) and entry.created_at != ""

    print("Test passed: LeaderboardEntry creation with datetime.")
