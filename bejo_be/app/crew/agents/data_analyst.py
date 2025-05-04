from crewai import Agent, LLM, Task
from urllib.parse import quote_plus
from app.crew.tools.database import SQLTool
from crewai_tools import DirectoryReadTool, FileReadTool, RagTool

import os

db_uri = f"mysql+pymysql://{os.getenv('DB_USER')}:{quote_plus(os.getenv('DB_PASSWORD'))}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
sql_tool = SQLTool(db_uri=db_uri)

config = {
    "llm": {
        "provider": "google",
        "config": {
            "model": "gemini-2.0-flash",
        },
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {"model": "nomic-embed-text:latest"},
    },
}
rag_tool = RagTool(config=config, summarize=True)
rag_tool.add(data_type="directory", source="storage")
## AGENT
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0,
    ),
    max_iter=1,
    allow_delegation=True,
)

reseacher = Agent(
    role="A Researcher to find related information",
    goal="Find related information from documents like pdf, docx, xlsx, etc.",
    backstory=(
        "You are a researcher who can find related information from documents like pdf, docx, xlsx, etc."
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0,
    ),
    allow_delegation=False,
    max_iter=1,
    verbose=True,
    tools=[rag_tool],
)

sql_dev = Agent(
    role="SQL Developer",
    goal="Construct and execute SQL queries based on user questions.",
    backstory=(
        """You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries
    
        NOTES:
        "If you don't understand of the database run '_get_context()'"
        """
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0,
    ),
    tools=[sql_tool],
    max_iter=1,
    allow_delegation=False,
    verbose=True,
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal=("Analyze the data from the SQL Developer and provide insights."),
    backstory=(
        "You are skilled at analyzing SQL query results in markdown tables, extracting meaningful patterns, and presenting insights in clear, simple language."
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0,
    ),
    allow_delegation=False,
    max_iter=1,
    verbose=True,
)


## TASKS
find_related_info = Task(
    description=(
        "You are a researcher to find related information from documents like pdf, docx, xlsx, etc. based on the user's question.\n\n"
    ),
    expected_output="Related information",
    agent=reseacher,
)


extract_data = Task(
    description=(
        "# Database Context\n"
        "Understand every sample data in context to make sure you are understand what will you do.\n"
        f"{sql_tool._get_context()}\n\n"
        "User's Questions:\n{query}\n\nWrite any needed SQL, run it, return the result as markdown table.\n"
    ),
    expected_output="Markdown Table",
    agent=sql_dev,
)

write_report = Task(
    description=(
        "You are Experienced Reporter. Write a concise report in simple Indonesian that's easily understood by non-experts.\n\n"
        "Use simple, everyday Indonesian 😊\n"
        "Maximum 3-5 short paragraphs\n"
        "Provide concrete insights, not generalizations 🌟"
    ),
    expected_output="A concise report in simple Indonesian that's easily understood by non-experts",
    agent=manager,
    context=[extract_data, find_related_info],
)
