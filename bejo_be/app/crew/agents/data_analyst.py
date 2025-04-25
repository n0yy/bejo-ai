from crewai import Agent, LLM, Task
from crewai.tools import tool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
    QuerySQLCheckerTool,
)
from urllib.parse import quote_plus
from langchain_community.utilities.sql_database import SQLDatabase

db_uri = f"mysql+pymysql://root:{quote_plus('Script@Kiddies321')}@localhost:3306/expb7"
db = SQLDatabase.from_uri(db_uri)

@tool("list_tables")
def list_tables_tool() -> str:
    """
    List all tables in the database.

    This tool returns a JSON string containing all tables in the database.

    Returns:
        str: A JSON string containing all tables in the database.
    """
    return ListSQLDatabaseTool(db=db).invoke("")

@tool("tables_schema")
def tables_schema_tool(table_name: str) -> str:
    """
    Get the schema of a table.

    Given a table name, this tool returns the schema of the table as a JSON string.

    Args:
        table_name (str): The table name to get the schema for.

    Returns:
        str: The schema of the table as a JSON string.
    """
    return InfoSQLDatabaseTool(db=db).invoke(table_name)

@tool("execute_sql")
def execute_sql_tool(sql_query: str) -> str:
    """
    Execute a SQL query.

    Given a SQL query, this tool executes it in the database and returns
    the result of the query as a JSON string.

    Args:
        sql_query (str): The SQL query to execute.

    Returns:
        str: The result of the query as a JSON string.
    """
    
    return QuerySQLDataBaseTool(db=db).invoke(sql_query)

@tool("check_sql")
def check_sql_tool(sql_query: str) -> str:
    """
    Check a SQL query for errors.

    Given a SQL query, this tool checks whether the query is valid or not.
    If the query is valid, this tool returns a JSON string indicating that
    the query is valid. If the query is invalid, this tool returns a JSON
    string indicating the error message.

    Args:
        sql_query (str): The SQL query to check.

    Returns:
        str: A JSON string containing the result of the check.
    """
    try:
        llm_checker = LLM(model="gemini/gemini-2.0-flash", temperature=0.0)
        return QuerySQLCheckerTool(db=db, llm=llm_checker).invoke(sql_query)
    except Exception as e:
        return f"Error using QuerySQLCheckerTool: {str(e)}"
    
@tool("get_samples_data_from_each_table")
def get_samples_data() -> str:
    """
    Get a sample of data from each table in the database.

    Returns a JSON string containing a sample of data from each table.
    """
    return QuerySQLDataBaseTool(db=db).invoke("SELECT * FROM information_schema.tables LIMIT 3")

tools = [list_tables_tool, tables_schema_tool, execute_sql_tool, check_sql_tool, get_samples_data]

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
    tools=tools,
    allow_delegation=False,
    verbose=True
)

data_analyst = Agent(
    role="Senior Data Analyst",
    goal=(
        "Analyze the data from the SQL Developer and provide insights."
    ),
    backstory=(
        "You analyze datasets using Python and produce clear, concise insights."
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