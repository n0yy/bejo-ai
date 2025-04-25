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
        temperature=0
    ),
    tools=[sql_tool],
    allow_delegation=False,
    verbose=True
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal=(
        "Analyze the data from the SQL Developer and provide insights."
    ),
    backstory=(
        "From the Data (Markdown Data), you will analyze"
        "You analyze using Python and produce clear, concise insights."
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0
    ),
    allow_delegation=False,
    verbose=True
)

data_viz_agent = Agent(
    role="Data Visualization Agent",
    goal=(
        "Generate a data and layout (JSON format: example: {'data': [...], 'layout': {...}}) to visualize data based on user queries."
        "Your output must be a valid JSON object."
    ),
    backstory=(
        "You are a data analyst expertise in Data Visualization."
        "And alse expertise in Data Scientist."
    ),
    tools=[],
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.0
    ),
    verbose=True
)

report_writer = Agent(
    role="Report Writer",
    goal="Summarize the analysis into a EASY TO UNDERSTAND REPORT.",
    backstory=(
        "You must create concise reports highlighting the most important findings."
    ),
    llm=LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0
    ),
    allow_delegation=False, 
    verbose=True
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
    agent=sql_dev
)

analyze_data = Task(
    description="Analyze the data from the SQL Developer. provide a detail explanation for {query}",
    expected_output="Detailed analysis text",
    agent=data_analyst,
    context=[extract_data]
)

data_viz_task = Task(
    description=(
        "Generate a data and layout (JSON format: example: {'data': [...], 'layout': {...}}) to visualize data based on user queries."
        "The color palette should be a warm color palette."
        "Your output must be a valid JSON object."
    ),
    expected_output="{'data': [...], 'layout': {...}}",
    agent=data_viz_agent,
    context=[extract_data, analyze_data]
)

write_report = Task(
    description="Write an executive summary of the report from analysis.",
    expected_output="Paragraph summarizing the analysis.",
    agent=report_writer,
    context=[analyze_data]
)