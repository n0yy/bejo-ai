from app.crew.agents.data_analyst import (
    sql_dev,
    data_analyst,
    extract_data,
    write_report,
)

from crewai import Crew, Process
from dotenv import load_dotenv
import os

load_dotenv()

storage_path = os.getenv("CREWAI_STORAGE_DIR", "./memory")

bejo_crew: Crew = Crew(
    agents=[sql_dev, data_analyst],
    tasks=[extract_data, write_report],
    process=Process.sequential,
    verbose=True,
)
