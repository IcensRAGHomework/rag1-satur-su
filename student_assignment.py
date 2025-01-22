import requests
import base64
from mimetypes import guess_type

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

history = ChatMessageHistory()

def get_llm():
    return AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

def get_holiday_tool():
    def get_holiday(year: int, month: int) -> str:
        url = f"https://calendarific.com/api/v2/holidays?&api_key=NWLS9VUCWzUG0H0jpGNem9yIR59dbftW&country=tw&year={year}&month={month}"
        response = requests.get(url)
        return response.json().get('response')
    class GetHoliday(BaseModel):
        year: int = Field(description="specific year")
        month: int = Field(description="specific month")
    return StructuredTool.from_function(
        name="get_holiday",
        description="Fetch holidays for specific year and month",
        func=get_holiday,
        args_schema=GetHoliday)

def get_agent(llm: BaseChatModel):
    def get_history() -> ChatMessageHistory:
        return history
    tools = [get_holiday_tool()]
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def get_holiday_output(llm: BaseChatModel, data: str):
    response_schemas = [
        ResponseSchema(
            name="date",
            description="該紀念日的日期",
            type="YYYY-MM-DD"),
        ResponseSchema(
            name="name",
            description="該紀念日的名稱")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將我提供的資料整理成指定格式,使用台灣語言,{format_instructions},有幾個答案就回答幾次,將所有答案放進同個list"),
        ("human","{data}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format_messages(data=data)).content
    return response

def get_result_output(llm: BaseChatModel, data: str, isList: bool):
    if isList:
        response_schemas = [
            ResponseSchema(
                name="Result",
                description="json內的所有內容",
                type="list")
        ]
    else:
        response_schemas = [
            ResponseSchema(
                name="Result",
                description="json內的所有內容")
        ]

    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將提供的json內容輸出成指定json格式,{format_instructions}"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format_messages(question=data)).content

    examples = [
        {"input": """```json
                    {
                            "Result": [
                                    content
                            ]
                    }
                    ```""",
        "output": """{
                            "Result": [
                                    content
                            ]
                    }"""}
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "將我提供的文字進行處理"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    response = llm.invoke(prompt.invoke(input = response)).content
    return response

def generate_hw01(question):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,用格式化的答案呈現,答案的日期要包含年月日,除了答案本身以外不要回答其他語句"),
        ("human","{question}")
        ])
    response = llm.invoke(prompt.format_messages(question = question)).content
    response = get_holiday_output(llm, response)
    response = get_result_output(llm, response, True)
    return response

def generate_hw02(question):
    llm = get_llm()
    agent = get_agent(llm)
    response = agent.invoke({"input": question}).get('output')
    response = get_holiday_output(llm, response)
    response = get_result_output(llm, response, True)
    return response
    
def generate_hw03(question2, question3):
    generate_hw02(question2)

    llm = get_llm()
    agent = get_agent(llm)

    response_schemas = [
        ResponseSchema(
            name="add",
            description="該紀念日是否需要加入先前的清單內,若月份相同且該紀念日不被包含在清單內則為true,否則為false",
            type = "boolean"),
        ResponseSchema(
            name="reason",
            description="決定該紀念日是否加入清單的理由")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","使用台灣語言並回答問題,{format_instructions}"),
        ("human","{question}")
        ])
    prompt = prompt.partial(format_instructions=format_instructions)
    response = agent.invoke({"input":prompt.format_messages(question=question3)}).get('output')
    response = get_result_output(llm, response, False)
    return response

def local_image_to_data_url():
    # Example usage
    image_path = './baseball.png'
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    llm = get_llm()
    image_url = local_image_to_data_url()
    response_schemas = [
        ResponseSchema(
            name="score",
            description="圖片文字表格中顯示的指定隊伍的積分數",
            type="integer")
    ]
    output_parser = StructuredOutputParser(response_schemas=response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "辨識圖片中的文字表格,{format_instructions}"),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                ],
            ),
            ("human", "{question}")
        ]
    )
    prompt = prompt.partial(format_instructions=format_instructions)
    response = llm.invoke(prompt.format_messages(question=question)).content
    response = get_result_output(llm, response, False)
    return response

def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response


print(generate_hw01("2024年台灣10月紀念日有哪些?"))
# print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'))
# print(generate_hw04('請問中華台北的積分是多少'))
