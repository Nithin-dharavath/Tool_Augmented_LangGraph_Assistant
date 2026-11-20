from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from rag_tool import rag_tool_func
from langgraph.types import interrupt, Command

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.2
)

load_dotenv()

@tool
def calculator(a : float, b : float, operation : str) -> dict:
    """Given the two numbers a and b perform arthermetic operations avaliable:
    add, sub, multiply, division"""
    decision = interrupt({
        "type" : "approval",
        "question" : f"""AI is going to use {operation} appprove or restictate it ?"""
    })

    if decision["approved"] == "NO":
        return {"message" : "Tool is rejected by user"}
        
    if operation == "addition":
        return {"result" : (a + b)}
    elif operation == "subtraction":
        return {"result" : (a - b)}
    elif operation == "multiply":
        return {"result" : (a*b)}
    elif operation == "division":
        if b == 0:
            return {"error" : "cant divide by the 0"}
        return {"result" : (a / b)}
    return {"error" : "invalid operation to perform"}

ducksearch = DuckDuckGoSearchRun()


toollist = [calculator, ducksearch, rag_tool_func]

llm_with_tools = llm.bind_tools(toollist)

tool_node = ToolNode(toollist)
