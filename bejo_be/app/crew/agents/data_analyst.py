from crewai import Agent, LLM, Task
from urllib.parse import quote_plus
from app.crew.tools.database import SQLTool
import os

db_uri = f"mysql+pymysql://{os.getenv('DB_USER')}:{quote_plus(os.getenv('DB_PASSWORD'))}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
sql_tool = SQLTool(db_uri=db_uri)

## AGENT
sql_dev = Agent(
    role="SQL Developer",
    goal="Construct and execute SQL queries based on user questions.",
    backstory=(
        """You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries
    
        NOTES:
        "Use the 'list_tables' to find available tables."
        "Use the 'tables_schema' to understand the metadata for the 'list_tables'."
        "Use the 'execute_sql' to execute queries against the database."
        "Use the 'check_sql' to check your queries for correctness."
        "Use 'get_samples_data_from_each_table' to get data samples to understand the data"  
        """
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0,
    ),
    tools=[sql_tool],
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
    verbose=True,
)


## TASKS
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
        "You are a senior data analyst. Analyze the markdown table from SQL Developer to provide insights 📈.\n\n"
        "Use simple, everyday Indonesian 😊\n"
        "Maximum 3-5 short paragraphs\n"
        "Provide concrete insights, not generalizations 🌟"
    ),
    expected_output="A concise report in simple Indonesian that's easily understood by non-experts",
    agent=data_analyst,
    context=[extract_data],
)
