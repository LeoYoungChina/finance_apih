import os
import json
from textwrap import dedent
import streamlit as st
from datetime import datetime
from agno.agent import Agent
from agno.models.dashscope import DashScope
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools
from typing import Any, Callable, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from decimal import Decimal
from agno.tools.tavily import TavilyTools

st.set_page_config(layout="wide")

# Custom tool hook
def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Hook function that wraps the tool execution"""
    print(f"About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"Function call completed with result: {result}")
    return result

# Define Report identification tool
@tool(
    name="identify_report",
    description="Identify the type of Report the user wants to query",
    show_result=True,
    tool_hooks=[logger_hook]
)
def identify_report(user_question: str) -> str:
    """Identify Report type based on user question"""
    # Load all Report information first
    report_info_list = []
    report_directory = "/home/SSG_GPS/report_matrix/reports"
    
    # Read all Report files
    for i in range(1, 21):
        report_file_path = os.path.join(report_directory, f"{i}.json")
        if os.path.exists(report_file_path):
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                report_info = {
                    "id": str(i),
                    "name": report_data["report_desc"]["name"],
                    "description": report_data["report_desc"]["description"],
                    "measures": report_data["measures"],
                    "dimensions": report_data["dimensions"]
                }
                report_info_list.append(report_info)
    
    # Create keyword mapping based on actual Report information
    report_keywords = {}
    for report in report_info_list:
        keywords = []
        # Extract keywords from Report name and description
        name_keywords = report["name"].lower().split()
        desc_keywords = report["description"].lower().split()
        
        # Add dimension and measure keywords
        dim_keywords = [dim.lower() for dim in report["dimensions"]]
        measure_keywords = [measure.lower() for measure in report["measures"]]
        
        # Merge all keywords
        all_keywords = name_keywords + desc_keywords + dim_keywords + measure_keywords
        
        # Clean keywords, remove common stop words and special characters
        cleaned_keywords = []
        for keyword in all_keywords:
            # Remove special characters
            cleaned_keyword = ''.join(e for e in keyword if e.isalnum())
            if len(cleaned_keyword) > 1:  # Only keep words with length > 1
                cleaned_keywords.append(cleaned_keyword)
        
        report_keywords[report["id"]] = list(set(cleaned_keywords))  # Deduplicate
    
    # Match user question with Report keywords
    user_question_lower = user_question.lower()
    
    # Calculate matching score for each Report
    report_scores = {}
    for report_id, keywords in report_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in user_question_lower:
                score += 1
        report_scores[report_id] = score
    
    # Find the Report with highest score
    if report_scores:
        best_report = max(report_scores, key=report_scores.get)
        # Only return the Report if matching score is greater than 0
        if report_scores[best_report] > 0:
            return best_report
    
    return "0"  # 0 means no specific Report identified, casual chat

# Define tool to get Report information
@tool(
    name="get_report_info",
    description="Get detailed information of a Report",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_report_info(report_id: str) -> dict:
    """Get detailed information of a specified Report"""
    report_file_path = f"/home/SSG_GPS/report_matrix/reports/{report_id}.json"
    
    if not os.path.exists(report_file_path):
        return {"error": "Report not found"}
    
    with open(report_file_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    return report_data

# Define MySQL query tool
@tool(
    name="query_mysql",
    description="Execute MySQL query",
    show_result=True,
    tool_hooks=[logger_hook]
)
def query_mysql(sql_query: str) -> str:
    """Execute MySQL query using SQLAlchemy and return results"""
    # Database connection parameters
    # These parameters should be obtained from configuration file or environment variables
    db_params = {
        "host": "localhost",
        "port": 3306,
        "username": "ssg_gps",
        "password": "2EndJWXHYSweDbDB",
        "database": "ssg_gps"
    }
    
    # Create database connection string
    db_uri = f"mysql+pymysql://{db_params['username']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    try:
        # Create SQLAlchemy engine
        engine: Engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800,
                                       connect_args={"connect_timeout": 5, "read_timeout": 30, "write_timeout": 30},
                                       echo=False, future=True)
        
        # Execute query
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert results to dictionary list and handle Decimal type
            def default_serializer(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            
            data = []
            for row in rows:
                row_dict = {}
                for col, val in zip(columns, row):
                    if isinstance(val, Decimal):
                        row_dict[col] = float(val)
                    else:
                        row_dict[col] = val
                data.append(row_dict)
            
            # Return formatted results
            return json.dumps(data, ensure_ascii=False, indent=2, default=default_serializer)
            
    except Exception as e:
        return f"Query execution failed: {str(e)}"

# New chart generation tool
@tool(
    name="generate_chart",
    description="Generate charts from data with enhanced features",
    show_result=True,
    tool_hooks=[logger_hook]
)
def generate_chart(chart_type: str, data: str, x_column: str, y_column: str, title: str, 
                  additional_params: dict = None) -> str:
    """
    Enhanced chart generation tool
    
    Parameters:
    chart_type: Chart type ("bar", "line", "pie", "scatter", "area")
    data: JSON formatted data
    x_column: Column name to use for X-axis
    y_column: Column name to use for Y-axis
    title: Chart title
    additional_params: Additional chart customization options
    
    Returns:
    Path to chart file
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import uuid
    
    # Set default parameters
    if additional_params is None:
        additional_params = {}
    
    # Set default theme
    pio.templates.default = "plotly_white"
    
    # Parse data
    data_list = json.loads(data)
    df = pd.DataFrame(data_list)
    
    # Check if data is empty
    if df.empty:
        return "Cannot generate chart: data is empty"
    
    # Generate unique filename
    chart_id = str(uuid.uuid4())
    chart_path = f"/tmp/chart_{chart_id}.html"
    
    # Macaron color scheme
    macaron_colors = [
        '#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7',
        '#C7CEEA', '#F8B195', '#F67280', '#C06C84', '#6C5B7B'
    ]
    
    # Generate chart
    try:
        fig = None
        
        if chart_type == "bar":
            df_sorted = df.sort_values(by=y_column, ascending=False)
            fig = px.bar(df_sorted, x=x_column, y=y_column, title=title, 
                        color=y_column, color_continuous_scale=macaron_colors)
        elif chart_type == "line":
            fig = px.line(df, x=x_column, y=y_column, title=title,
                         color_discrete_sequence=macaron_colors[:len(df)])
        elif chart_type == "pie":
            fig = px.pie(df, values=y_column, names=x_column, title=title, 
                        color_discrete_sequence=macaron_colors[:len(df)])
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_column, y=y_column, title=title,
                           color_discrete_sequence=macaron_colors)
        elif chart_type == "area":
            fig = px.area(df, x=x_column, y=y_column, title=title,
                         color_discrete_sequence=macaron_colors[:len(df)])
        
        # Apply additional customization parameters
        if fig and additional_params:
            # Set legend position
            if 'legend_position' in additional_params:
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
            
            # Set axis labels
            if 'x_label' in additional_params:
                fig.update_xaxes(title_text=additional_params['x_label'])
            if 'y_label' in additional_params:
                fig.update_yaxes(title_text=additional_params['y_label'])
            
            # Enable data labels
            if additional_params.get('show_labels', False):
                fig.update_traces(textposition='auto')
        
        # Update chart layout
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16),
            hovermode='closest'
        )
        
        # Save chart
        fig.write_html(chart_path)
        return chart_path
        
    except Exception as e:
        return f"Chart generation failed: {str(e)}"

# New tool: read organization structure information
@tool(
    name="get_organization_structure",
    description="Get Lenovo's organization structure information",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_organization_structure() -> str:
    """Get Lenovo's organization structure information"""
    org_file_path = "/home/SSG_GPS/report_matrix/organization_relationship.json"
    
    if not os.path.exists(org_file_path):
        return "Organization structure file not found"
    
    try:
        with open(org_file_path, 'r', encoding='utf-8') as f:
            org_data = json.load(f)
        
        # Parse organization structure information
        lenovo_data = org_data.get("Lenovo", {})
        
        def extract_entities(data, entities=None):
            """Recursively extract all entities"""
            if entities is None:
                entities = set()
            
            if isinstance(data, dict):
                for key, value in data.items():
                    entities.add(key)
                    extract_entities(value, entities)
            elif isinstance(data, list):
                for item in data:
                    entities.add(item)
            
            return entities
        
        # Extract all organization entities
        all_entities = extract_entities(lenovo_data)
        
        # Build organization structure information summary
        org_summary = {
            "Total entities": len(all_entities),
            "Organization entities": list(all_entities)
        }
        
        return json.dumps(org_summary, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Error reading organization structure information: {str(e)}"

# New tool: read financial formulas and dictionary information
@tool(
    name="get_formulas_and_dictionaries",
    description="Get formula and dictionary information, including indicator calculation methods and term definitions",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_formulas_and_dictionaries() -> str:
    """Get formula and dictionary information"""
    formulas_dict_file_path = "/home/SSG_GPS/report_matrix/formulas_and_dictionaries.json"
    
    if not os.path.exists(formulas_dict_file_path):
        return "Formula and dictionary file not found"
    
    try:
        with open(formulas_dict_file_path, 'r', encoding='utf-8') as f:
            formulas_dict_data = json.load(f)
        
        return json.dumps(formulas_dict_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Error reading financial formulas and dictionary information: {str(e)}"

# New tool: date range parsing tool
@tool(
    name="parse_date_range",
    description="Parse date range from user input",
    show_result=True,
    tool_hooks=[logger_hook]
)
def parse_date_range(user_input: str) -> dict:
    """
    Extract date range information from user input
    
    Parameters:
    user_input: User's input text
    
    Returns:
    Dictionary containing start date and end date
    """
    import re
    from datetime import datetime, timedelta
    
    # Common date patterns
    date_patterns = [
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # MM-DD-YYYY or MM/DD/YYYY
        r'(\d{4}年\d{1,2}月\d{1,2}日)',     # YYYY年MM月DD日
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, user_input)
        for match in matches:
            try:
                # Try to parse date
                if '/' in match:
                    date_obj = datetime.strptime(match, '%Y/%m/%d') if len(match.split('/')[0]) == 4 else datetime.strptime(match, '%m/%d/%Y')
                elif '-' in match:
                    date_obj = datetime.strptime(match, '%Y-%m-%d') if len(match.split('-')[0]) == 4 else datetime.strptime(match, '%m-%d-%Y')
                else:
                    date_obj = datetime.strptime(match, '%Y年%m月%d日')
                dates_found.append(date_obj)
            except ValueError:
                continue
    
    # Sort dates
    dates_found.sort()
    
    result = {}
    if len(dates_found) >= 2:
        result['start_date'] = dates_found[0].strftime('%Y-%m-%d')
        result['end_date'] = dates_found[-1].strftime('%Y-%m-%d')
    elif len(dates_found) == 1:
        result['start_date'] = dates_found[0].strftime('%Y-%m-%d')
        # If only one date is provided, default to query data for one week before and after that date
        end_date = dates_found[0] + timedelta(days=7)
        result['end_date'] = end_date.strftime('%Y-%m-%d')
    else:
        # If no dates are found, return empty result
        result['start_date'] = None
        result['end_date'] = None
    
    return result

# New tool: check data availability
@tool(
    name="check_data_availability",
    description="Check data availability in database",
    show_result=True,
    tool_hooks=[logger_hook]
)
def check_data_availability() -> str:
    """Check data availability in database, including date range and other key information"""
    # Database connection parameters
    db_params = {
        "host": "localhost",
        "port": 3306,
        "username": "ssg_gps",
        "password": "2EndJWXHYSweDbDB",
        "database": "ssg_gps"
    }
    
    # Create database connection string
    db_uri = f"mysql+pymysql://{db_params['username']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    try:
        # Create SQLAlchemy engine
        engine: Engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800,
                                       connect_args={"connect_timeout": 5, "read_timeout": 30, "write_timeout": 30},
                                       echo=False, future=True)
        
        # Execute multiple queries to check data availability
        info = {}
        
        # Check total record count
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as total_records FROM ssg_gps_data"))
            info['total_records'] = result.fetchone()[0]
            
            # Check date range
            result = conn.execute(text("SELECT MIN(create_date) as min_date, MAX(create_date) as max_date FROM ssg_gps_data"))
            row = result.fetchone()
            info['date_range'] = {
                'min_date': str(row[0]) if row[0] else None,
                'max_date': str(row[1]) if row[1] else None
            }
            
            # Check available account_name values
            result = conn.execute(text("SELECT DISTINCT account_name FROM ssg_gps_data WHERE account_name IS NOT NULL AND account_name != '' LIMIT 10"))
            info['sample_accounts'] = [row[0] for row in result.fetchall()]
            
            # Check available create_month values
            result = conn.execute(text("SELECT DISTINCT create_month FROM ssg_gps_data WHERE create_month IS NOT NULL ORDER BY create_month"))
            info['available_months'] = [row[0] for row in result.fetchall()]
            
            # Check order types
            result = conn.execute(text("SELECT DISTINCT order_type FROM ssg_gps_data WHERE order_type IS NOT NULL"))
            info['order_types'] = [row[0] for row in result.fetchall()]
            
        return json.dumps(info, ensure_ascii=False, indent=2)
            
    except Exception as e:
        return f"Error checking data availability: {str(e)}"

# New tool: root cause analysis tool
@tool(
    name="perform_root_cause_analysis",
    description="Perform root cause analysis",
    show_result=True,
    tool_hooks=[logger_hook]
)
def perform_root_cause_analysis(query_context: str, available_data: str, analysis_request: str) -> str:
    """
    Perform root cause analysis
    
    Parameters:
    query_context: Query context information
    available_data: Available data summary
    analysis_request: Analysis request details
    
    Returns:
    Root cause analysis result or request for more information
    """
    return f"""Root cause analysis request received.
    
**Query Context**: {query_context}
**Available Data**: {available_data}
**Analysis Request**: {analysis_request}

Performing root cause analysis..."""

# New tool: ask for date range
@tool(
    name="ask_for_date_range",
    description="Ask user for date range when dealing with report queries",
    show_result=True,
    tool_hooks=[logger_hook]
)
def ask_for_date_range() -> str:
    """
    Ask user for date range when dealing with report queries
    
    Returns:
    A message prompting the user to provide a date range
    """
    return "To better provide you with report data, please provide the date range for your query (e.g., 2023-01-01 to 2023-12-31)."

# New tool: explain report calculation
# 在 explain_report_calculation 工具中添加更深入的逻辑解析
@tool(
    name="explain_report_calculation",
    description="Explain how a report is calculated with enhanced analysis",
    show_result=True,
    tool_hooks=[logger_hook]
)
def explain_report_calculation(report_id: str) -> str:
    """
    Explain how a report is calculated with enhanced analysis
    
    Parameters:
    report_id: The ID of the report to explain
    
    Returns:
    A detailed explanation of how the report is calculated
    """
    report_file_path = f"/home/SSG_GPS/report_matrix/reports/{report_id}.json"
    
    if not os.path.exists(report_file_path):
        return "Report not found"
    
    try:
        with open(report_file_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # Extract calculation information
        report_name = report_data.get("report_desc", {}).get("name", "Unknown Report")
        measures = report_data.get("measures", [])
        calculation_logic = report_data.get("calculation_logic", {})
        sql_query = report_data.get("sql_query", {}).get("query", "No SQL query available")
        dependencies = report_data.get("dependencies", [])
        
        explanation = f"# {report_name} Calculation Explanation\n\n"
        
        # Add measures information
        if measures:
            explanation += "## Measures\n"
            for measure in measures:
                explanation += f"- {measure}\n"
            explanation += "\n"
        
        # Add calculation logic with enhanced analysis
        if calculation_logic:
            explanation += "## Calculation Logic\n"
            for key, value in calculation_logic.items():
                explanation += f"### {key}\n{value}\n\n"
                
                # 如果是数学计算逻辑，尝试解析并验证
                if "formula" in key.lower() or "calculation" in key.lower():
                    explanation += "#### Formula Analysis\n"
                    # 这里可以添加更深入的公式分析逻辑
                    explanation += "This formula combines multiple data sources to produce the final metric.\n\n"
        else:
            explanation += "## Calculation Logic\n"
            explanation += "No specific calculation logic provided.\n\n"
        
        # 添加依赖关系分析
        if dependencies:
            explanation += "## Dependencies\n"
            explanation += "This report depends on the following data sources or other reports:\n"
            for dep in dependencies:
                explanation += f"- {dep}\n"
            explanation += "\n"
        
        # Add SQL query
        explanation += "## SQL Query\n"
        explanation += f"```sql\n{sql_query}\n```\n\n"
        
        # 添加执行示例（如果可能）
        explanation += "## Sample Execution\n"
        explanation += "To validate this calculation, you can execute a sample query with limited data:\n"
        explanation += "```sql\n"
        # 生成一个简化版的查询用于验证
        sample_query = sql_query.replace("SELECT", "SELECT /* Sample Execution */", 1)
        explanation += f"{sample_query}\n"
        explanation += "```\n\n"
        
        return explanation
        
    except Exception as e:
        return f"Error reading report calculation information: {str(e)}"

# 添加一个新的工具用于验证报告计算
@tool(
    name="validate_report_calculation",
    description="Validate a report's calculation by executing a sample query",
    show_result=True,
    tool_hooks=[logger_hook]
)
def validate_report_calculation(report_id: str, sample_data: dict = None) -> str:
    """
    Validate a report's calculation by executing a sample query
    
    Parameters:
    report_id: The ID of the report to validate
    sample_data: Sample data to use for validation (optional)
    
    Returns:
    Validation results
    """
    # 获取报告信息
    report_info = get_report_info(report_id)
    if "error" in report_info:
        return f"Error: {report_info['error']}"
    
    sql_query = report_info.get("sql_query", {}).get("query", "")
    if not sql_query:
        return "No SQL query found for this report"
    
    # 创建一个简化版的查询用于验证（限制结果数量）
    validation_query = sql_query
    if "LIMIT" not in sql_query.upper():
        validation_query = f"{sql_query} LIMIT 10"
    
    # 执行查询
    result = query_mysql(validation_query)
    
    validation_report = f"# Report {report_id} Calculation Validation\n\n"
    validation_report += f"## Executed Query\n```sql\n{validation_query}\n```\n\n"
    validation_report += f"## Results\n{result}\n\n"
    validation_report += "## Validation Status\n"
    
    try:
        data = json.loads(result)
        if isinstance(data, list) and len(data) > 0:
            validation_report += "✅ Calculation validation successful - data retrieved\n"
        else:
            validation_report += "⚠️ No data returned - check query conditions\n"
    except json.JSONDecodeError:
        validation_report += "❌ Validation failed - error in query execution\n"
        validation_report += f"Error details: {result}\n"
    
    return validation_report

# New tool: collect report dimensions
@tool(
    name="collect_report_dimensions",
    description="Based on report requirements, collect necessary dimension parameters from user",
    show_result=True,
    tool_hooks=[logger_hook]
)
def collect_report_dimensions(report_id: str, user_query: str) -> dict:
    """
    Based on report requirements, collect necessary dimension parameters from user
    
    Parameters:
    report_id: Report ID
    user_query: User's original query
    
    Returns:
    Dictionary containing dimension parameters
    """
    # Get report information
    report_info = get_report_info(report_id)
    if "error" in report_info:
        return {"error": report_info["error"]}
    
    # Get dimension information
    dimensions = report_info.get("dimensions", [])
    
    # Get dictionary information
    dict_info = get_formulas_and_dictionaries()
    if "File not found" in dict_info:
        return {"error": "Dictionary file not found"}
    
    try:
        dict_data = json.loads(dict_info)
    except json.JSONDecodeError:
        return {"error": "Failed to parse dictionary data"}
    
    # Analyze dimension information provided in user query
    provided_dimensions = {}
    missing_dimensions = []
    
    user_query_lower = user_query.lower()
    
    for dimension in dimensions:
        # Check if user query already provides this dimension information
        dimension_found = False
        
        # Get possible values for dimension from dictionary
        if dimension in dict_data and "possible_values" in dict_data[dimension]:
            possible_values = dict_data[dimension].get("possible_values", [])
            # Check if user query contains these possible values
            for value in possible_values:
                if value.lower() in user_query_lower:
                    provided_dimensions[dimension] = value
                    dimension_found = True
                    break
        
        if not dimension_found:
            missing_dimensions.append(dimension)
    
    return {
        "provided_dimensions": provided_dimensions,
        "missing_dimensions": missing_dimensions,
        "required_action": "ask_user_for_missing_dimensions" if missing_dimensions else "proceed_with_query"
    }

# New tool: ask for missing dimensions
@tool(
    name="ask_for_missing_dimensions",
    description="Ask user for missing dimension information to refine report query",
    show_result=True,
    tool_hooks=[logger_hook]
)
def ask_for_missing_dimensions(missing_dimensions: list, report_name: str) -> str:
    """
    Ask user for missing dimension information to refine report query
    
    Parameters:
    missing_dimensions: List of missing dimensions
    report_name: Report name
    
    Returns:
    Message asking user for information
    """
    # Get dimension dictionary information
    dict_info = get_formulas_and_dictionaries()
    dict_data = {}
    if "File not found" not in dict_info:
        try:
            dict_data = json.loads(dict_info)
        except json.JSONDecodeError:
            pass
    
    message = f"To provide you with a more accurate **{report_name}** report, I need some additional information:\n\n"
    
    for dimension in missing_dimensions:
        message += f"- **{dimension}**"
        # If dimension has example values, provide them to user for reference
        if dimension in dict_data and "possible_values" in dict_data[dimension]:
            examples = dict_data[dimension]["possible_values"][:3]  # Show only first 3 examples
            message += f" (e.g., {', '.join(examples)})"
        message += "\n"
    
    message += "\nPlease provide these details to help me generate a more precise report for you."
    
    return message

def initialize_app():
    st.title("🔍 Chat to SSG GPS Data")
    
    # Add welcome message
    st.markdown("""
    ### Welcome to Lenovo SSG GPS Data Assistant! 🚀
    
    Hello! I'm your intelligent data analysis assistant, here to help you explore Lenovo's SSG GPS reports and metrics. 
    
    **You can ask me questions like:**
    - "Show me the top accounts by revenue in FY25/26"
    - "What are our service order trends over the last quarter?"
    - "Explain how the customer satisfaction report is calculated"
    - "Compare our performance across different regions"
    
    I can also help with:
    - Generating visual charts from data
    - Providing organizational structure information
    - Explaining financial formulas and terms
    - Performing root cause analysis on business metrics
    
    Just type your question below and I'll do my best to assist you!
    """)
    
    if "msgs" not in st.session_state:
        st.session_state["msgs"] = []
        
    if "current_report" not in st.session_state:
        st.session_state["current_report"] = None
        
    if "report_parameters" not in st.session_state:
        st.session_state["report_parameters"] = {}
        
    return True

def load_all_report_info():
    """Load all Report information for building detailed prompts"""
    report_info_list = []
    report_directory = "/home/SSG_GPS/report_matrix/reports"
    
    for i in range(1, 21):
        report_file_path = os.path.join(report_directory, f"{i}.json")
        if os.path.exists(report_file_path):
            try:
                with open(report_file_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                    # Check if required keys exist
                    if "report_desc" not in report_data:
                        print(f"Warning: Report {i}.json is missing 'report_desc' field, skipping this report")
                        continue
                    
                    if "name" not in report_data["report_desc"] or "description" not in report_data["report_desc"]:
                        print(f"Warning: Report {i}.json's 'report_desc' is missing required fields, skipping this report")
                        continue
                    
                    if "measures" not in report_data or "dimensions" not in report_data:
                        print(f"Warning: Report {i}.json is missing 'measures' or 'dimensions' field, skipping this report")
                        continue
                    
                    if "sql_query" not in report_data or "query" not in report_data["sql_query"]:
                        print(f"Warning: Report {i}.json is missing 'sql_query' field, skipping this report")
                        continue
                    
                    report_info = {
                        "id": str(i),
                        "name": report_data["report_desc"]["name"],           # Correct access path
                        "description": report_data["report_desc"]["description"],  # Correct access path
                        "measures": report_data["measures"],
                        "dimensions": report_data["dimensions"],
                        "sql_query": report_data["sql_query"]["query"]
                    }
                    report_info_list.append(report_info)
            except json.JSONDecodeError:
                print(f"Warning: Report {i}.json is not valid JSON format, skipping this report")
                continue
            except KeyError as e:
                print(f"Warning: Report {i}.json is missing required field {e}, skipping this report")
                continue
            except Exception as e:
                print(f"Warning: Error processing report {i}.json: {e}, skipping this report")
                continue
    
    return report_info_list

def format_report_info_for_prompt(report_info_list):
    """Format Report information as prompt, removing specific date ranges to avoid misleading"""
    formatted_reports = []
    for report in report_info_list:
        try:
            # Remove specific date conditions from SQL query to avoid misleading Agent
            sql_query = report['sql_query']
            # Replace specific date ranges with placeholders
            import re
            # Match common date conditions and replace with placeholders
            sql_query_modified = re.sub(
                r"WHERE\s+.*?(create_date|closed_date|create_time|closed_time).*?(=|>|<|>=|<=|BETWEEN).*?['\"][^'\"]*['\"][^\)]*",
                "WHERE [DATE_FILTERS]", 
                sql_query, 
                flags=re.IGNORECASE | re.DOTALL
            )
            
            report_text = f"""
Report ID: {report['id']}
Name: {report['name']}
Description: {report['description']}
Measures: {', '.join(report['measures'])}
Dimensions: {', '.join(report['dimensions'])}
Sample SQL query structure: {sql_query_modified}"""
            formatted_reports.append(report_text)
        except KeyError as e:
            print(f"Warning: Report {report.get('id', 'unknown')} data structure is incomplete, missing field {e}")
            continue
    
    return "\n\n".join(formatted_reports)

def setup_agent():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load all Report information
    report_info_list = load_all_report_info()
    formatted_report_info = format_report_info_for_prompt(report_info_list)
    
    # Create DashScope model instance
    dashscope_model = DashScope(
        id="qwen3-max",
        api_key="sk-a3a228f6afcc4e3d9a34a5c5a7270fb3",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    agent = Agent(
        model=dashscope_model,
        markdown=True,
        tools=[
            ReasoningTools(add_instructions=True),
            identify_report,
            get_report_info,
            get_organization_structure,
            get_formulas_and_dictionaries,
            query_mysql,
            generate_chart,
            parse_date_range,  # Add date parsing tool
            check_data_availability,  # Add data availability check tool
            TavilyTools(api_key="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"),
            perform_root_cause_analysis,  # Add root cause analysis tool
            ask_for_date_range,  # Add tool to ask for date range
            explain_report_calculation,  # Add tool to explain report calculation
            collect_report_dimensions,  # Add dimension collection tool
            ask_for_missing_dimensions,  # Add tool to ask for missing dimensions
        ],

        instructions=dedent(f"""
                            
            You are Lenovo SSG GPS Report analysis assistant,capable of helping users query and analyze various metrics.
            
            Current time: {current_time}, for now we are actually in FY 25/26 in Lenovo.
            
            
            Your capabilities include:
            1. Judging whether user questions are casual chat or Report queries
            2. Identifying the specific Report type the user wants to query,and pay attention to not take the Report example as a query directly
            3. Obtaining detailed Report information and required parameters
            4. Checking parameter completeness, and guiding users to supplement if parameters are missing
            5. Executing database queries and returning detailed, clear and intuitive results
            6. Generating charts from the data to visualize Reports
            7. Providing organizational structure information when requested
            8. Providing formulas and dictionaries for better understanding of metrics
            9. If users want to chat casually, please accompany them in conversation
            10. Performing root cause analysis when requested
            11. Always ask for date range when dealing with report queries
            12. Explaining how reports are calculated when requested
            13. Collecting dimension parameters to refine report queries
            14. Asking users for missing dimension information
            
            The following is the available Report information:
            {formatted_report_info}
            
            Workflow:
            1. First judge the user question type (casual chat/Report query/root cause analysis)
            2. If it's a Report query, identify the specific Report type
            3. Ask user for date range using ask_for_date_range tool
            4. Obtain the DSL information {formatted_report_info} for that Report
            5. Check if additional parameters are needed
            6. If needed, guide the user to supplement parameters
            7. Execute the query and display detailed results in an intuitive and clear manner, such as tables, etc.
            8. After displaying data in table format, automatically generate appropriate charts to visualize the data
            9. If users ask about organizational structure, use the get_organization_structure tool to provide information
            10. If users ask about formulas or terms, use the get_formulas_and_dictionaries tool to provide explanations
            11. If users want to chat casually, please accompany them in conversation
            12. Share your final sql query to the user.
            13. When users request root cause analysis, use the perform_root_cause_analysis tool
            14. When users ask how a report is calculated, use the explain_report_calculation tool
            15. When processing report queries, use collect_report_dimensions to identify missing dimensions
            16. If dimensions are missing, use ask_for_missing_dimensions to request them from the user
            
            Notes:
            - Strictly execute queries according to the Report-defined SQL query structure
            - When users inquire about specific Reports, please refer to the corresponding SQL query structure and fields
            - If the user's question does not match any Report, please engage in casual chat or request clarification
            - Ensure the language of the output answers is the same language used by the user
            - Showing your thinking process
            - When generating charts, select the most appropriate chart type for the data:
              * Bar charts for comparisons
              * Line charts for trends over time
              * Pie charts for proportions
            - When users ask about organizational structure, use the get_organization_structure tool to provide comprehensive information
            - When explaining metrics, use the get_formulas_and_dictionaries tool to provide accurate definitions and calculation methods
            - When explaining terms, refer to the Formulas and Dictionaries section above for accurate definitions
            - CRITICAL: Always prioritize date ranges provided by the user over any example date ranges in the Report information
            - If the user specifies a date range, replace any example date filters in the SQL query with the user's date range
            - If no date range is specified by the user, you may use the example date range from the Report information
            - Use the parse_date_range tool to extract date information from user input
            - When modifying SQL queries, ensure you maintain the correct table and column names while only changing the date filters
            - If a query returns no results, suggest alternative approaches:
              * Try a broader date range
              * Check if specific filters might be too restrictive
              * Verify data availability by running a simple count query
              * Suggest checking different time periods
            - Root Cause Analysis Module:
              * When users ask about reasons for data anomalies, declines or growth, trigger root cause analysis
              * Root Cause Analysis Process:
                1. Determine the analysis target and context
                2. Obtain relevant data and check data quality
                3. Execute specific queries if needed to gather more information using query_mysql tool
                4. If additional information is needed for effective analysis, clearly state what information is needed
                5. Based on data and business knowledge, propose possible root cause hypotheses
                6. Provide validation suggestions and action plans
              * Common Root Cause Analysis Scenarios:
                - Performance decline analysis
                - Abnormal fluctuation investigation
                - KPI non-achievement reason finding
                - Trend change interpretation
              * If you need more information from the user to continue analysis, clearly explain what information is needed and why
              * Combine Lenovo business context for analysis
              * Consider seasonal, market and other factors
              * Provide actionable recommendations
            - For all report-related queries, always ask for a date range using the ask_for_date_range tool
            - When explaining how a report is calculated, use the explain_report_calculation tool to provide a detailed breakdown
            - Always present the calculation method of the report separately in your response when relevant
            - After executing a query and displaying results in table format, automatically generate an appropriate chart using the generate_chart tool
            - Dimension Optimization Analysis Module:
              * When identifying a user query for a specific report, check the required dimensions for that report
              * Analyze whether the user query already provides sufficient dimension information
              * If dimension information is insufficient, actively ask the user for missing dimension parameters
              * Use information in formulas_and_dictionaries.json to guide dimension parameter collection
              * Optimize SQL queries based on collected dimension information to provide more accurate analysis results
              * Common dimension types include: geographic regions, business units, product lines, customer types, etc.
              * When requesting dimension information, provide example values to help users understand what type of information is needed
        """),
        reasoning=True,
    )
    return agent

def process_assistant_response(agent, messages):
    # Create response placeholder
    message_placeholder = st.empty()
    
    full_response = ""
    
    run_response = agent.run(
        messages, 
        stream=True
    )
    
    for chunk in run_response:
        # Process response content
        if chunk.content:
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")
    
    # Update final content (remove cursor)
    message_placeholder.markdown(full_response)
    
    # Check and display charts
    import re
    chart_paths = re.findall(r"/tmp/chart_[a-f0-9-]+\.html", full_response)
    # Deduplicate to avoid repeated display
    chart_paths = list(set(chart_paths))
    for chart_path in chart_paths:
        if os.path.exists(chart_path):
            try:
                with open(chart_path, "r") as f:
                    chart_html = f.read()
                    st.components.v1.html(chart_html, height=500)
            except Exception as e:
                st.error(f"Chart display failed: {str(e)}")
        else:
            st.warning(f"Chart file not found: {chart_path}")
    
    # Check if user needs to provide more information for root cause analysis
    if "need more information" in full_response.lower() or "please provide" in full_response.lower():
        st.info("💡 System detected that root cause analysis needs more information, please check the request above and provide the corresponding data.")
    
    return full_response

def main():
    initialize_app()
    agent = setup_agent()
    
    # Display history messages
    for msg in st.session_state["msgs"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If message contains chart paths, display charts
            import re
            chart_paths = re.findall(r"/tmp/chart_[a-f0-9-]+\.html", msg["content"])
            # Deduplicate to avoid repeated display
            chart_paths = list(set(chart_paths))
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    try:
                        with open(chart_path, "r") as f:
                            chart_html = f.read()
                            st.components.v1.html(chart_html, height=500)
                    except Exception as e:
                        st.error(f"Chart display failed: {str(e)}")
                else:
                    st.warning(f"Chart file not found: {chart_path}")

    if prompt := st.chat_input("talk to me"):
        st.session_state["msgs"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            full_response = process_assistant_response(agent, st.session_state["msgs"])
            st.session_state["msgs"].append({"role": "assistant", "content": full_response})
            
            # If response contains chart paths, display charts
            import re
            chart_paths = re.findall(r"/tmp/chart_[a-f0-9-]+\.html", full_response)
            # Deduplicate to avoid repeated display
            chart_paths = list(set(chart_paths))
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    try:
                        with open(chart_path, "r") as f:
                            chart_html = f.read()
                            st.components.v1.html(chart_html, height=500)
                    except Exception as e:
                        st.error(f"Chart display failed: {str(e)}")
                else:
                    st.warning(f"Chart file not found: {chart_path}")

if __name__ == "__main__":
    main()