import os
import json
from textwrap import dedent
import streamlit as st
from datetime import datetime
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools
from typing import Any, Callable, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from decimal import Decimal
from agno.tools.tavily import TavilyTools

st.set_page_config(layout="wide")

# 自定义工具钩子
def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Hook function that wraps the tool execution"""
    print(f"About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"Function call completed with result: {result}")
    return result

# 定义KPI识别工具
@tool(
    name="identify_kpi",
    description="识别用户想要查询的KPI类型",
    show_result=True,
    tool_hooks=[logger_hook]
)
def identify_kpi(user_question: str) -> str:
    """根据用户问题识别KPI类型"""
    # 先加载所有KPI信息
    kpi_info_list = []
    kpi_directory = "/home/Finance_v2/kpi_matrix/kpis"
    
    # 读取所有KPI文件
    for i in range(1, 11):
        kpi_file_path = os.path.join(kpi_directory, f"{i}.json")
        if os.path.exists(kpi_file_path):
            with open(kpi_file_path, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                kpi_info = {
                    "id": str(i),
                    "name": kpi_data["kpi_desc"]["name"],
                    "description": kpi_data["kpi_desc"]["description"],
                    "measures": kpi_data["measures"],
                    "dimensions": kpi_data["dimensions"]
                }
                kpi_info_list.append(kpi_info)
    
    # 创建关键词映射，基于实际的KPI信息
    kpi_keywords = {}
    for kpi in kpi_info_list:
        keywords = []
        # 从KPI名称和描述中提取关键词
        name_keywords = kpi["name"].lower().split()
        desc_keywords = kpi["description"].lower().split()
        
        # 添加维度和度量关键词
        dim_keywords = [dim.lower() for dim in kpi["dimensions"]]
        measure_keywords = [measure.lower() for measure in kpi["measures"]]
        
        # 合并所有关键词
        all_keywords = name_keywords + desc_keywords + dim_keywords + measure_keywords
        
        # 清理关键词，移除常见停用词和特殊字符
        cleaned_keywords = []
        for keyword in all_keywords:
            # 移除特殊字符
            cleaned_keyword = ''.join(e for e in keyword if e.isalnum())
            if len(cleaned_keyword) > 1:  # 只保留长度大于1的词
                cleaned_keywords.append(cleaned_keyword)
        
        kpi_keywords[kpi["id"]] = list(set(cleaned_keywords))  # 去重
    
    # 匹配用户问题与KPI关键词
    user_question_lower = user_question.lower()
    
    # 为每个KPI计算匹配分数
    kpi_scores = {}
    for kpi_id, keywords in kpi_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in user_question_lower:
                score += 1
        kpi_scores[kpi_id] = score
    
    # 找到最高分的KPI
    if kpi_scores:
        best_kpi = max(kpi_scores, key=kpi_scores.get)
        # 只有当匹配分数大于0时才返回该KPI
        if kpi_scores[best_kpi] > 0:
            return best_kpi
    
    return "0"  # 0表示未识别出特定KPI，闲聊

# 定义获取KPI信息工具
@tool(
    name="get_kpi_info",
    description="获取KPI的详细信息",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_kpi_info(kpi_id: str) -> dict:
    """获取指定KPI的详细信息"""
    kpi_file_path = f"/home/Finance_v2/kpi_matrix/kpis/{kpi_id}.json"
    
    if not os.path.exists(kpi_file_path):
        return {"error": "KPI not found"}
    
    with open(kpi_file_path, 'r', encoding='utf-8') as f:
        kpi_data = json.load(f)
    
    return kpi_data

# 定义MySQL查询工具
@tool(
    name="query_mysql",
    description="执行MySQL查询",
    show_result=True,
    tool_hooks=[logger_hook]
)
def query_mysql(sql_query: str) -> str:
    """使用SQLAlchemy执行MySQL查询并返回结果"""
    # 数据库连接参数
    # 这些参数应该从配置文件或环境变量中获取
    db_params = {
        "host": "localhost",
        "port": 3306,
        "username": "finance",
        "password": "ieWdNpW487t5xcTY",
        "database": "finance"
    }
    
    # 创建数据库连接字符串
    db_uri = f"mysql+pymysql://{db_params['username']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    try:
        # 创建SQLAlchemy引擎
        engine: Engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800,
                                       connect_args={"connect_timeout": 5, "read_timeout": 30, "write_timeout": 30},
                                       echo=False, future=True)
        
        # 执行查询
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            
            # 将结果转换为字典列表，并处理 Decimal 类型
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
            
            # 返回格式化的结果
            return json.dumps(data, ensure_ascii=False, indent=2, default=default_serializer)
            
    except Exception as e:
        return f"查询执行失败: {str(e)}"

# 新增图表生成工具
@tool(
    name="generate_chart",
    description="根据数据生成图表",
    show_result=True,
    tool_hooks=[logger_hook]
)
def generate_chart(chart_type: str, data: str, x_column: str, y_column: str, title: str) -> str:
    """
    生成图表并保存为HTML文件
    
    参数:
    chart_type: 图表类型 ("bar", "line", "pie")
    data: JSON格式的数据
    x_column: X轴使用的列名
    y_column: Y轴使用的列名
    title: 图表标题
    
    返回:
    图表文件的路径
    """
    import pandas as pd
    import plotly.express as px
    import plotly.io as pio
    import uuid
    
    # 设置默认主题为plotly_white，确保图表有明亮的背景
    pio.templates.default = "plotly_white"
    
    # 解析数据
    data_list = json.loads(data)
    df = pd.DataFrame(data_list)
    
    # 生成唯一文件名
    chart_id = str(uuid.uuid4())
    chart_path = f"/tmp/chart_{chart_id}.html"
    
    # 定义马卡龙色系调色板，从重色到轻色
    macaron_colors = [
        '#FF9AA2',  # 珊瑚粉
        '#FFB7B2',  # 浅珊瑚
        '#FFDAC1',  # 淡桃色
        '#E2F0CB',  # 薄荷绿
        '#B5EAD7',  # 淡青绿
        '#C7CEEA',  # 淡紫
        '#F8B195',  # 橙粉
        '#F67280',  # 玫瑰粉
        '#C06C84',  # 紫红
        '#6C5B7B',  # 淡紫灰
        '#F16D7A',  # 马卡龙草莓奶霜
        '#E27386',  # 马卡龙玫瑰
        '#F55066',  # 马卡龙玫瑰粉红
        '#EF5464',  # 马卡龙基尔Anperiaru
        '#AE716E',  # 大巧克力杏仁饼
        '#CB8E85',  # 巧克力马卡龙
        '#CF8878',  # 杏仁饼，果仁糖，巧克力
        '#C86F67',  # 覆盆子马卡龙
        '#F1CCB8',  # 盐焦糖杏仁饼
        '#F2DEBD',  # 杏仁饼香草奶油
        '#B7D28D',  # 马卡龙抹茶奶霜
        '#DCFF93',  # 开心果杏仁杏仁饼
        '#FF9B6A',  # 牛奶咖啡马卡龙
        '#F1B8E4',  # 马卡龙粉
        '#D9B8F1',  # 杏仁饼紫
        '#F1F1B8',  # 杏仁饼黄色
        '#B8F1ED',  # 杏仁饼海洋蓝
        '#B8F1CC',  # 杏仁饼绿色
        '#E7DAC9',  # 马卡龙枫武
        '#E1622F',  # 杏仁饼蔓藤
        '#F3D64E',  # 杏仁饼，柚子
        '#FD7D36',  # 马卡龙玫瑰果
        '#FE9778',  # 马卡龙覆盆子乳酪蛋糕
        '#C38E9E',  # 马卡龙枫薰衣草
        '#F28860',  # 巧克力马卡龙马龙
        '#DE772C',  # 马卡龙达米安
        '#E96A25',  # 马卡龙果仁巧克力
        '#CA7497',  # 马卡龙黑醋栗
        '#E29E4B',  # 马卡龙激情巧克力
        '#EDBF2B',  # 马卡龙Powaru
        '#FECF45',  # 马卡龙芒果激情
        '#F9B747',  # 马卡龙柚子，覆盆子
        '#C17E61',  # 马卡龙让JA斗
        '#ED9678',  # 杏仁饼牛轧糖
        '#FFE543',  # 马卡龙可可凤梨
        '#E37C5B',  # 马卡龙伯爵茶
        '#FF8240',  # 马卡龙巧克力橙
        '#AA5B71',  # 马卡龙巧克力樱桃色
        '#F0B631',  # 巧克力马卡龙Bananu
        '#CF8888'   # 马卡龙Kurakkure
    ]
    
    # 根据类型生成图表，使用马卡龙色系
    if chart_type == "bar":
        # 对数据按y值排序，实现从高到低的颜色渐变效果
        df_sorted = df.sort_values(by=y_column, ascending=False)
        # 使用马卡龙色系的连续色阶
        fig = px.bar(df_sorted, x=x_column, y=y_column, title=title, 
                     color=y_column, color_continuous_scale=macaron_colors)
    elif chart_type == "line":
        # 线图使用马卡龙色系
        fig = px.line(df, x=x_column, y=y_column, title=title,
                      color_discrete_sequence=macaron_colors[:len(df)])
    elif chart_type == "pie":
        # 饼图使用马卡龙色系
        fig = px.pie(df, values=y_column, names=x_column, title=title, 
                     color_discrete_sequence=macaron_colors[:len(df)])
    else:
        # 默认使用柱状图
        df_sorted = df.sort_values(by=y_column, ascending=False)
        fig = px.bar(df_sorted, x=x_column, y=y_column, title=title,
                     color=y_column, color_continuous_scale=macaron_colors)
    
    # 更新图表布局，确保使用明亮主题
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        title_font=dict(size=16)
    )
    
    # 保存图表
    fig.write_html(chart_path)
    return chart_path

# 新增工具：读取组织结构信息
@tool(
    name="get_organization_structure",
    description="获取联想公司的组织结构信息",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_organization_structure() -> str:
    """获取联想公司的组织结构信息"""
    org_file_path = "/home/Finance_v2/kpi_matrix/organization_relationship.json"
    
    if not os.path.exists(org_file_path):
        return "组织结构文件未找到"
    
    try:
        with open(org_file_path, 'r', encoding='utf-8') as f:
            org_data = json.load(f)
        
        # 解析组织结构信息
        lenovo_data = org_data.get("Lenovo", {})
        
        def extract_entities(data, entities=None):
            """递归提取所有实体"""
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
        
        # 提取所有组织实体
        all_entities = extract_entities(lenovo_data)
        
        # 构建组织结构信息摘要
        org_summary = {
            "总实体数": len(all_entities),
            "组织实体": list(all_entities)
        }
        
        return json.dumps(org_summary, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"读取组织结构信息时出错: {str(e)}"

# 新增工具：读取财务公式和字典信息
@tool(
    name="get_financial_formulas_and_dictionaries",
    description="获取财务公式和字典信息，包括指标计算方法和术语定义",
    show_result=True,
    tool_hooks=[logger_hook]
)
def get_financial_formulas_and_dictionaries() -> str:
    """获取财务公式和字典信息"""
    formulas_dict_file_path = "/home/Finance_v2/kpi_matrix/Financial_formulas_and_dictionaries.json"
    
    if not os.path.exists(formulas_dict_file_path):
        return "财务公式和字典文件未找到"
    
    try:
        with open(formulas_dict_file_path, 'r', encoding='utf-8') as f:
            formulas_dict_data = json.load(f)
        
        return json.dumps(formulas_dict_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"读取财务公式和字典信息时出错: {str(e)}"

def initialize_app():
    st.title("🔍 Chat to Financial Data")
    
    if "msgs" not in st.session_state:
        st.session_state["msgs"] = []
        
    if "current_kpi" not in st.session_state:
        st.session_state["current_kpi"] = None
        
    if "kpi_parameters" not in st.session_state:
        st.session_state["kpi_parameters"] = {}
        
    return True

def load_all_kpi_info():
    """加载所有KPI信息用于构建更详细的提示词"""
    kpi_info_list = []
    kpi_directory = "/home/Finance_v2/kpi_matrix/kpis"
    
    for i in range(1, 11):
        kpi_file_path = os.path.join(kpi_directory, f"{i}.json")
        if os.path.exists(kpi_file_path):
            with open(kpi_file_path, 'r', encoding='utf-8') as f:
                kpi_data = json.load(f)
                kpi_info = {
                    "id": str(i),
                    "name": kpi_data["kpi_desc"]["name"],
                    "description": kpi_data["kpi_desc"]["description"],
                    "measures": kpi_data["measures"],
                    "dimensions": kpi_data["dimensions"],
                    "sql_query": kpi_data["sql_query"]["query"]
                }
                kpi_info_list.append(kpi_info)
    
    return kpi_info_list

def format_kpi_info_for_prompt(kpi_info_list):
    """将KPI信息格式化为提示词"""
    formatted_kpis = []
    for kpi in kpi_info_list:
        kpi_text = f"""
KPI ID: {kpi['id']}
名称: {kpi['name']}
描述: {kpi['description']}
度量指标: {', '.join(kpi['measures'])}
维度: {', '.join(kpi['dimensions'])}
示例SQL查询: {kpi['sql_query']}"""
        formatted_kpis.append(kpi_text)
    
    return "\n\n".join(formatted_kpis)
# 替换 setup_agent() 函数中的模型初始化部分
# 替换 setup_agent() 函数中的模型初始化部分
def setup_agent():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 加载所有KPI信息
    kpi_info_list = load_all_kpi_info()
    formatted_kpi_info = format_kpi_info_for_prompt(kpi_info_list)
    
    # 加载财务公式和字典信息
    formulas_dict_file_path = "/home/Finance_v2/kpi_matrix/Financial_formulas_and_dictionaries.json"
    try:
        with open(formulas_dict_file_path, 'r', encoding='utf-8') as f:
            financial_formulas_dict = json.load(f)
        formatted_formulas_dict = json.dumps(financial_formulas_dict, ensure_ascii=False, indent=2)
    except Exception as e:
        formatted_formulas_dict = f"无法加载财务公式和字典: {str(e)}"
    
    # 创建自定义的联想API模型类
    from agno.models.base import Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    class LenovoAPIModel(Model):
        def __init__(self):
            super().__init__(id="gpt-4o-mini")
            self.api_key = "VBOyFcVkgJuGwhTROgbB1atm7QlVhI1z"
            self.token_url = "https://apihub-us.lenovo.com/token"
            self.api_url = "https://apihub-us.lenovo.com/prod/v1/services/aiverse-ukm/ics-apps/projects/115/ukm/aiverse/endpoint/v1/chat/completions"
            self.access_token = None
            self._get_token()
        
        def _get_token(self):
            """获取联想API访问令牌"""
            token_payload = {
                'username': 'api_akm_aiesa',
                'password': 'YB=&f%6$C-'
            }
            
            token_headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Harbor/0.0.1 (Python)"
            }
            
            try:
                response = requests.post(
                    self.token_url, 
                    headers=token_headers, 
                    data=token_payload, 
                    verify=False, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.access_token = token_data.get('access_token')
                else:
                    raise Exception(f"Failed to get token: {response.status_code} - {response.text}")
                    
            except Exception as e:
                raise Exception(f"Error getting token: {str(e)}")
        
        def _parse_provider_response(self, response: dict) -> ModelResponse:
            """解析提供商响应"""
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                return ModelResponse(content=content)
            else:
                raise Exception("No response content from API")
        
        def _parse_provider_response_delta(self, response: dict) -> ModelResponse:
            """解析流式响应增量"""
            if 'choices' in response and len(response['choices']) > 0:
                delta = response['choices'][0].get('delta', {})
                content = delta.get('content', '')
                return ModelResponse(content=content)
            else:
                return ModelResponse(content='')
        
        def invoke(self, messages, **kwargs):
            """调用联想API"""
            if not self.access_token:
                self._get_token()
                
            api_payload = {
                "model": "gpt-4o-mini",
                "stream": False,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            api_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Harbor/0.0.1 (Python)"
            }
            
            try:
                response = requests.post(
                    self.api_url, 
                    headers=api_headers, 
                    json=api_payload, 
                    verify=False, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    api_data = response.json()
                    return self._parse_provider_response(api_data)
                else:
                    # 如果是认证错误，尝试重新获取token
                    if response.status_code == 401:
                        self._get_token()
                        # 重新尝试一次请求
                        api_headers["Authorization"] = f"Bearer {self.access_token}"
                        response = requests.post(
                            self.api_url, 
                            headers=api_headers, 
                            json=api_payload, 
                            verify=False, 
                            timeout=60
                        )
                        if response.status_code == 200:
                            api_data = response.json()
                            return self._parse_provider_response(api_data)
                    
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                raise Exception(f"Error calling Lenovo API: {str(e)}")
        
        async def ainvoke(self, messages, **kwargs):
            """异步调用联想API"""
            # 简单实现同步调用的异步版本
            return self.invoke(messages, **kwargs)
        
        def invoke_stream(self, messages, **kwargs):
            """流式调用联想API"""
            if not self.access_token:
                self._get_token()
                
            api_payload = {
                "model": "gpt-4o-mini",
                "stream": True,  # 启用流式传输
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            api_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Harbor/0.0.1 (Python)"
            }
            
            try:
                response = requests.post(
                    self.api_url, 
                    headers=api_headers, 
                    json=api_payload, 
                    verify=False, 
                    timeout=60,
                    stream=True
                )
                
                if response.status_code == 200:
                    # 处理流式响应
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data: '):
                                data = decoded_line[6:]  # 移除 'data: ' 前缀
                                if data != '[DONE]':
                                    try:
                                        json_data = json.loads(data)
                                        yield self._parse_provider_response_delta(json_data)
                                    except json.JSONDecodeError:
                                        continue
                else:
                    # 如果是认证错误，尝试重新获取token
                    if response.status_code == 401:
                        self._get_token()
                        # 重新尝试一次请求
                        api_headers["Authorization"] = f"Bearer {self.access_token}"
                        response = requests.post(
                            self.api_url, 
                            headers=api_headers, 
                            json=api_payload, 
                            verify=False, 
                            timeout=60,
                            stream=True
                        )
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if line:
                                    decoded_line = line.decode('utf-8')
                                    if decoded_line.startswith('data: '):
                                        data = decoded_line[6:]
                                        if data != '[DONE]':
                                            try:
                                                json_data = json.loads(data)
                                                yield self._parse_provider_response_delta(json_data)
                                            except json.JSONDecodeError:
                                                continue
                    
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                raise Exception(f"Error calling Lenovo API: {str(e)}")
        
        async def ainvoke_stream(self, messages, **kwargs):
            """异步流式调用联想API"""
            # 简单实现同步流式调用的异步版本
            for response in self.invoke_stream(messages, **kwargs):
                yield response
    
    # 使用自定义的联想API模型
    lenovo_model = LenovoAPIModel()
    
    agent = Agent(
        model=lenovo_model,
        markdown=True,
        tools=[
            ReasoningTools(add_instructions=True),
            identify_kpi,
            get_kpi_info,
            get_organization_structure,
            get_financial_formulas_and_dictionaries,
            query_mysql,
            generate_chart,
            TavilyTools(api_key="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"),
        ],

        instructions=dedent(f"""
                            
            You are Lenovo SSG Financail KPI analysis assistant,your name is Superman ,capable of helping users query and analyze various Financial metrics.
            
            Current time: {current_time},
            Time peroid: in Lenovo ,we are actually in FY 25/26
            
            Your capabilities include:
            1. Judging whether user questions are casual chat or KPI queries
            2. Identifying the specific KPI type the user wants to query,and pay attention to not take the KPI example as a query directly
            3. Obtaining detailed KPI information and required parameters
            4. Checking parameter completeness, and guiding users to supplement if parameters are missing
            5. Executing database queries and returning detailed, clear and intuitive results
            6. Generating charts from the data to visualize KPIs
            7. Providing organizational structure information when requested
            8. Providing financial formulas and dictionaries for better understanding of financial metrics
            9. If users want to chat casually, please accompany them in conversation
            
            The following is the available KPI information:
            {formatted_kpi_info}
            
            Workflow:
            1. First judge the user question type (casual chat/KPI query)
            2. If it's a KPI query, identify the specific KPI type
            3. Obtain the DSL information {formatted_kpi_info} for that KPI
            4. Check if additional parameters are needed
            5. If needed, guide the user to supplement parameters
            6. Execute the query and display detailed results in an intuitive and clear manner, such as tables, etc.
            7. If appropriate, generate charts to visualize the data
            8. If users ask about organizational structure, use the get_organization_structure tool to provide information
            9. If users ask about financial formulas or terms, use the get_financial_formulas_and_dictionaries tool to provide explanations
            10. If users want to chat casually, please accompany them in conversation
            11. Share your final sql query to the user.
            
            Notes:
            - Strictly execute queries according to the KPI-defined SQL query structure
            - When users inquire about specific KPIs, please refer to the corresponding SQL query structure and fields
            - If the user's question does not match any KPI, please engage in casual chat or request clarification
            - Ensure the language of the output answers is the same language used by the user
            - Showing your thinking process
            - When generating charts, select the most appropriate chart type for the data:
              * Bar charts for comparisons
              * Line charts for trends over time
              * Pie charts for proportions
            - When users ask about organizational structure, use the get_organization_structure tool to provide comprehensive information
            - When explaining financial metrics, use the get_financial_formulas_and_dictionaries tool to provide accurate definitions and calculation methods
            - When explaining financial terms, refer to the Financial Formulas and Dictionaries section above for accurate definitions
        """),
        reasoning=False,
    )
    return agent
def process_assistant_response(agent, messages):
    # 创建回答占位符
    message_placeholder = st.empty()
    
    full_response = ""
    
    run_response = agent.run(
        messages, 
        stream=True
    )
    
    for chunk in run_response:
        # 处理回答内容
        if chunk.content:
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")
    
    # 更新最终内容（去掉光标）
    message_placeholder.markdown(full_response)
    
    return full_response

def main():
    initialize_app()
    agent = setup_agent()
    
    # 显示历史消息
    for msg in st.session_state["msgs"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # 如果消息中包含图表路径，则显示图表
            import re
            chart_paths = re.findall(r"/tmp/chart_[a-f0-9-]+\.html", msg["content"])
            # 去重以避免重复显示
            chart_paths = list(set(chart_paths))
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    with open(chart_path, "r") as f:
                        chart_html = f.read()
                        st.components.v1.html(chart_html, height=500)

    if prompt := st.chat_input("talk to me"):
        st.session_state["msgs"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            full_response = process_assistant_response(agent, st.session_state["msgs"])
            st.session_state["msgs"].append({"role": "assistant", "content": full_response})
            
            # 如果响应中包含图表路径，则显示图表
            import re
            chart_paths = re.findall(r"/tmp/chart_[a-f0-9-]+\.html", full_response)
            # 去重以避免重复显示
            chart_paths = list(set(chart_paths))
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    with open(chart_path, "r") as f:
                        chart_html = f.read()
                        st.components.v1.html(chart_html, height=500)

if __name__ == "__main__":
    main()