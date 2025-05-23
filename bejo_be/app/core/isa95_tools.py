from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

from .retrieval import RetrievalService

logger = logging.getLogger(__name__)


class ISA95LevelAnalysisInput(BaseModel):
    """Input for ISA-95 level analysis tool"""

    query: str = Field(description="Query to analyze across ISA-95 levels")
    start_level: int = Field(default=1, description="Starting ISA-95 level", ge=1, le=4)
    end_level: int = Field(default=4, description="Ending ISA-95 level", ge=1, le=4)


class ISA95LevelAnalysisTool(BaseTool):
    """Tool for analyzing information across different ISA-95 levels"""

    name: str = "isa95_level_analysis"
    description: str = """
    Analyze how a topic or query relates across different ISA-95 levels.
    This tool helps understand the impact and relationships of concepts from 
    field level up to management level.
    
    Parameters:
    - query: The topic or question to analyze across levels
    - start_level: Starting ISA-95 level (1=Field, 2=Supervisory, 3=Planning, 4=Management)
    - end_level: Ending ISA-95 level (must be >= start_level)
    
    Use this when you need to understand:
    - How field issues impact higher levels
    - How management decisions affect lower levels
    - Cross-level integration and dependencies
    """
    args_schema: type[BaseModel] = ISA95LevelAnalysisInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service
        self.level_names = {
            1: "Field & Control System",
            2: "Supervisory",
            3: "Planning",
            4: "Management",
        }

    def _run(self, query: str, start_level: int = 1, end_level: int = 4) -> str:
        """Execute cross-level analysis"""
        try:
            if start_level > end_level:
                return "Error: start_level must be <= end_level"

            analysis_result = f"ISA-95 Cross-Level Analysis for: '{query}'\n"
            analysis_result += "=" * 60 + "\n\n"

            level_results = {}

            # Analyze each level
            for level in range(start_level, end_level + 1):
                level_name = self.level_names[level]
                analysis_result += f"Level {level} - {level_name}:\n"
                analysis_result += "-" * 40 + "\n"

                # Search for relevant information at this level
                documents = self.retrieval_service.retrieve_documents(
                    query=query,
                    level=level,
                    strategy="hierarchical",
                    search_kwargs={"k": 3},
                )

                if documents:
                    level_insights = []
                    for doc in documents:
                        content = doc.page_content[:300]
                        # Extract key insights (first meaningful sentence)
                        sentences = [
                            s.strip() for s in content.split(".") if len(s.strip()) > 20
                        ]
                        if sentences:
                            level_insights.append(sentences[0])

                    analysis_result += f"Key Insights:\n"
                    for i, insight in enumerate(level_insights[:3], 1):
                        analysis_result += f"  {i}. {insight}\n"

                    level_results[level] = level_insights
                else:
                    analysis_result += f"No specific information found at this level.\n"
                    level_results[level] = []

                analysis_result += "\n"

            # Cross-level impact analysis
            if len(level_results) > 1:
                analysis_result += "Cross-Level Impact Analysis:\n"
                analysis_result += "=" * 30 + "\n"

                # Bottom-up impact
                if start_level < end_level:
                    analysis_result += (
                        f"Bottom-up Impact (Level {start_level} → Level {end_level}):\n"
                    )
                    analysis_result += f"- Issues at {self.level_names[start_level]} level can affect:\n"
                    for level in range(start_level + 1, end_level + 1):
                        analysis_result += f"  • {self.level_names[level]}: Operational efficiency and decision-making\n"

                # Top-down influence
                if end_level > start_level:
                    analysis_result += f"\nTop-down Influence (Level {end_level} → Level {start_level}):\n"
                    analysis_result += f"- Decisions at {self.level_names[end_level]} level influence:\n"
                    for level in range(end_level - 1, start_level - 1, -1):
                        analysis_result += f"  • {self.level_names[level]}: Resource allocation and priorities\n"

            return analysis_result

        except Exception as e:
            logger.error(f"Error in ISA-95 level analysis: {e}")
            return f"Error performing level analysis: {str(e)}"


class IntegrationPointsInput(BaseModel):
    """Input for integration points analysis"""

    domain: str = Field(
        description="Domain or system to analyze integration points for"
    )
    source_level: int = Field(description="Source ISA-95 level", ge=1, le=4)
    target_level: int = Field(description="Target ISA-95 level", ge=1, le=4)


class IntegrationPointsTool(BaseTool):
    """Tool for analyzing integration points between ISA-95 levels"""

    name: str = "integration_points_analysis"
    description: str = """
    Analyze integration points and data flows between different ISA-95 levels.
    This tool helps identify how systems at different levels should integrate
    and what information needs to flow between them.
    
    Parameters:
    - domain: The domain or system context (e.g., "production", "quality", "maintenance")
    - source_level: The ISA-95 level that sends information
    - target_level: The ISA-95 level that receives information
    
    Use this for:
    - Understanding data flow requirements
    - Identifying integration challenges
    - Planning system interfaces
    - Troubleshooting communication issues
    """
    args_schema: type[BaseModel] = IntegrationPointsInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service
        self.level_names = {
            1: "Field & Control System",
            2: "Supervisory",
            3: "Planning",
            4: "Management",
        }

        # Common integration patterns
        self.integration_patterns = {
            (1, 2): {
                "data_flow": "Real-time process data, alarms, status information",
                "protocols": "OPC-UA, Modbus, Ethernet/IP, HART",
                "challenges": "Real-time requirements, data volume, reliability",
            },
            (2, 3): {
                "data_flow": "Production reports, batch records, performance metrics",
                "protocols": "MES interfaces, databases, web services",
                "challenges": "Data aggregation, timing synchronization, context preservation",
            },
            (3, 4): {
                "data_flow": "KPIs, production schedules, resource utilization",
                "protocols": "ERP interfaces, BI systems, REST APIs",
                "challenges": "Data summarization, business context, decision support",
            },
            (1, 3): {
                "data_flow": "Direct process optimization data, quality parameters",
                "protocols": "Historian interfaces, advanced process control",
                "challenges": "Bypassing supervisory layer, data validation",
            },
            (2, 4): {
                "data_flow": "Operational dashboards, performance summaries",
                "protocols": "Business intelligence tools, reporting systems",
                "challenges": "Executive summary level, trend analysis",
            },
            (1, 4): {
                "data_flow": "Critical alarms, safety events, compliance data",
                "protocols": "Emergency notification systems, audit trails",
                "challenges": "Priority management, escalation procedures",
            },
        }

    def _run(self, domain: str, source_level: int, target_level: int) -> str:
        """Analyze integration points between levels"""
        try:
            source_name = self.level_names[source_level]
            target_name = self.level_names[target_level]

            result = f"Integration Points Analysis: {domain.title()}\n"
            result += f"From Level {source_level} ({source_name}) → Level {target_level} ({target_name})\n"
            result += "=" * 70 + "\n\n"

            # Get domain-specific information from both levels
            source_query = f"{domain} integration data output level {source_level}"
            target_query = f"{domain} integration data input level {target_level}"

            source_docs = self.retrieval_service.retrieve_documents(
                query=source_query, level=source_level, search_kwargs={"k": 2}
            )

            target_docs = self.retrieval_service.retrieve_documents(
                query=target_query, level=target_level, search_kwargs={"k": 2}
            )

            # Source level capabilities
            result += f"Source Level {source_level} ({source_name}) Capabilities:\n"
            result += "-" * 50 + "\n"
            if source_docs:
                for i, doc in enumerate(source_docs, 1):
                    content = doc.page_content[:200]
                    result += f"{i}. {content}...\n"
            else:
                result += f"General capabilities: Data collection, process control, monitoring\n"
            result += "\n"

            # Target level requirements
            result += f"Target Level {target_level} ({target_name}) Requirements:\n"
            result += "-" * 50 + "\n"
            if target_docs:
                for i, doc in enumerate(target_docs, 1):
                    content = doc.page_content[:200]
                    result += f"{i}. {content}...\n"
            else:
                result += f"General requirements: Aggregated data, reports, decision support\n"
            result += "\n"

            # Integration pattern analysis
            integration_key = (source_level, target_level)
            reverse_key = (target_level, source_level)

            if integration_key in self.integration_patterns:
                pattern = self.integration_patterns[integration_key]
                result += "Integration Pattern (Upward Flow):\n"
            elif reverse_key in self.integration_patterns:
                pattern = self.integration_patterns[reverse_key]
                result += "Integration Pattern (Downward Flow):\n"
            else:
                # Create generic pattern for unusual combinations
                pattern = {
                    "data_flow": "Custom data exchange requirements",
                    "protocols": "Standard industrial protocols",
                    "challenges": "Level mismatch, data transformation",
                }
                result += "Integration Pattern (Custom):\n"

            result += "-" * 30 + "\n"
            result += f"Typical Data Flow: {pattern['data_flow']}\n"
            result += f"Common Protocols: {pattern['protocols']}\n"
            result += f"Key Challenges: {pattern['challenges']}\n\n"

            # Recommendations
            result += "Integration Recommendations:\n"
            result += "-" * 30 + "\n"

            if source_level < target_level:
                result += "• Implement data aggregation and filtering\n"
                result += "• Ensure proper data validation and cleansing\n"
                result += "• Consider real-time vs batch processing requirements\n"
                result += "• Plan for data buffering during outages\n"
            else:
                result += "• Implement command validation and authorization\n"
                result += "• Ensure proper change management procedures\n"
                result += "• Consider impact on lower level operations\n"
                result += "• Plan for emergency override capabilities\n"

            if abs(source_level - target_level) > 1:
                result += "• Consider intermediate level involvement\n"
                result += "• Implement proper escalation procedures\n"
                result += "• Ensure audit trail for direct communications\n"

            return result

        except Exception as e:
            logger.error(f"Error in integration points analysis: {e}")
            return f"Error analyzing integration points: {str(e)}"


class ComplianceCheckInput(BaseModel):
    """Input for ISA-95 compliance checking"""

    topic: str = Field(description="Topic or process to check compliance for")
    compliance_standard: str = Field(
        default="ISA-95", description="Compliance standard to check against"
    )
    focus_levels: List[int] = Field(
        default=[1, 2, 3, 4], description="ISA-95 levels to check"
    )


class ISA95ComplianceTool(BaseTool):
    """Tool for checking ISA-95 compliance and best practices"""

    name: str = "isa95_compliance_check"
    description: str = """
    Check compliance with ISA-95 standards and best practices for a given topic or process.
    This tool helps ensure that implementations follow ISA-95 guidelines and identify
    potential compliance issues.
    
    Parameters:
    - topic: The topic, process, or system to check compliance for
    - compliance_standard: Standard to check against (default: ISA-95)
    - focus_levels: List of ISA-95 levels to check (default: all levels)
    
    Use this for:
    - Validating system designs against ISA-95
    - Identifying compliance gaps
    - Ensuring best practice implementation
    - Preparing for audits or assessments
    """
    args_schema: type[BaseModel] = ComplianceCheckInput

    def __init__(self, retrieval_service: RetrievalService):
        super().__init__()
        self.retrieval_service = retrieval_service
        self.level_names = {
            1: "Field & Control System",
            2: "Supervisory",
            3: "Planning",
            4: "Management",
        }

    def _run(
        self,
        topic: str,
        compliance_standard: str = "ISA-95",
        focus_levels: List[int] = [1, 2, 3, 4],
    ) -> str:
        """Check ISA-95 compliance for a topic"""
        try:
            result = f"ISA-95 Compliance Check: {topic.title()}\n"
            result += f"Standard: {compliance_standard}\n"
            result += f"Focus Levels: {focus_levels}\n"
            result += "=" * 60 + "\n\n"

            compliance_issues = []
            best_practices = []

            for level in focus_levels:
                if level not in self.level_names:
                    continue

                level_name = self.level_names[level]
                result += f"Level {level} - {level_name} Compliance:\n"
                result += "-" * 40 + "\n"

                # Search for compliance-related information
                compliance_query = (
                    f"{topic} ISA-95 compliance best practices level {level}"
                )
                docs = self.retrieval_service.retrieve_documents(
                    query=compliance_query,
                    level=level,
                    strategy="comprehensive",
                    search_kwargs={"k": 3},
                )

                if docs:
                    level_compliance = []
                    for doc in docs:
                        content = doc.page_content

                        # Look for compliance indicators
                        if any(
                            word in content.lower()
                            for word in [
                                "standard",
                                "compliance",
                                "requirement",
                                "specification",
                            ]
                        ):
                            sentences = [
                                s.strip()
                                for s in content.split(".")
                                if len(s.strip()) > 30
                            ]
                            if sentences:
                                level_compliance.extend(sentences[:2])

                    if level_compliance:
                        result += "Compliance Guidelines Found:\n"
                        for i, guideline in enumerate(level_compliance[:3], 1):
                            result += f"  {i}. {guideline}\n"
                        best_practices.extend(level_compliance[:2])
                    else:
                        result += "No specific compliance guidelines found.\n"
                        compliance_issues.append(
                            f"Level {level}: Missing compliance documentation"
                        )
                else:
                    result += "No compliance information available.\n"
                    compliance_issues.append(
                        f"Level {level}: No compliance information in knowledge base"
                    )

                result += "\n"

            # Summary
            result += "Compliance Summary:\n"
            result += "=" * 20 + "\n"

            if compliance_issues:
                result += "⚠️  Compliance Issues Identified:\n"
                for i, issue in enumerate(compliance_issues, 1):
                    result += f"  {i}. {issue}\n"
                result += "\n"

            if best_practices:
                result += "✅ Best Practices Found:\n"
                for i, practice in enumerate(best_practices[:5], 1):
                    result += f"  {i}. {practice[:100]}...\n"
                result += "\n"

            # General ISA-95 compliance recommendations
            result += "General ISA-95 Compliance Recommendations:\n"
            result += "-" * 45 + "\n"
            result += "• Ensure clear functional hierarchy separation\n"
            result += "• Implement proper data flow between levels\n"
            result += "• Maintain consistent object models\n"
            result += "• Document integration interfaces\n"
            result += "• Follow standard terminology and definitions\n"
            result += "• Implement proper security at each level\n"
            result += "• Ensure traceability and audit capabilities\n"

            return result

        except Exception as e:
            logger.error(f"Error in ISA-95 compliance check: {e}")
            return f"Error checking compliance: {str(e)}"


def create_isa95_specialized_tools(
    retrieval_service: RetrievalService,
) -> List[BaseTool]:
    """Create ISA-95 specialized tools"""
    return [
        ISA95LevelAnalysisTool(retrieval_service),
        IntegrationPointsTool(retrieval_service),
        ISA95ComplianceTool(retrieval_service),
    ]
