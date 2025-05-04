from app.crew.agents.data_analyst import (
    sql_dev,
    manager,
    data_analyst,
    extract_data,
    write_report,
    reseacher,
    find_related_info,
)

from crewai import Crew, Process, LLM
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from dotenv import load_dotenv
import os

load_dotenv()

storage_path = os.getenv("CREWAI_STORAGE_DIR", "./memory")

bejo_crew: Crew = Crew(
    agents=[reseacher, sql_dev, data_analyst],
    tasks=[find_related_info, extract_data, write_report],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=True,
    memory=True,
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(db_path=f"{os.getenv('CREWAI_STORAGE_DIR')}/ltm.db")
    ),
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "ollama",
                "config": {"model": "nomic-embed-text:latest"},
            },
            type="short_term",
            path=os.getenv("CREWAI_STORAGE_DIR"),
        )
    ),
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "ollama",
                "config": {"model": "nomic-embed-text:latest"},
            },
            type="entity",
            path=os.getenv("CREWAI_STORAGE_DIR"),
        )
    ),
)
