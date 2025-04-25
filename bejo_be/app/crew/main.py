from app.crew.agents.data_analyst import (
    sql_dev, data_analyst, 
    report_writer, extract_data, 
    analyze_data, write_report,
    data_viz_agent, data_viz_task
)

from crewai import Crew, Process

bejo_crew: Crew = Crew(
    agents=[sql_dev, data_analyst, data_viz_agent, report_writer],
    tasks=[extract_data, analyze_data, data_viz_task, write_report],
    process=Process.sequential,
    verbose=True,
    output_log_file="crew.log"
)