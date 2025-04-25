from app.crew.agents.data_analyst import (
    sql_dev, data_analyst, 
    report_writer, extract_data, 
    analyze_data, write_report,
    data_viz_agent, data_viz_task
)

from crewai import Crew, Process
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from dotenv import load_dotenv
import os

load_dotenv()

storage_path = os.getenv("CREWAI_STORAGE_DIR", "./memory")

bejo_crew: Crew = Crew(
    agents=[sql_dev, data_analyst, data_viz_agent, report_writer],
    tasks=[extract_data, analyze_data, data_viz_task, write_report],
    process=Process.sequential,
    verbose=True,
    memory=True,
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(
            db_path=f"{storage_path}/memory.db"
        )
    ),
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "google",
                "config": {
                    "model": "text-embedding-004",
                    "api_key": os.getenv("GEMINI_API_KEY")
                }
            },
            type="short_term",
            path=storage_path
        )
    ),
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config={
                "provider": "google",
                "config": {
                    "model": "text-embedding-004",
                    "api_key": os.getenv("GEMINI_API_KEY")
                }
            },
            type="short_term",
            path=storage_path
        )
    ),

)