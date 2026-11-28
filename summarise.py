from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from main import llm
from schema import WorkflowState


def summarise_stlm(state: WokflowState):
    """
    Summarize conversation messages when they exceed a threshold.
    
    Correct pattern:
    1. Check if summary already exists (avoid duplicates)
    2. Only summarize when messages > 10 AND no summary exists
    3. Generate summary and return state update
    4. The summary message is added once; on next turn, message count
       resets because summary replaces old messages
    """
    messages = state["messages"]
    summary = state.get("summary", "")
    
    # Only summarize if BOTH conditions are met:
    # 1. No summary exists yet
    # 2. Message count exceeds threshold (10 messages)
    if not summary and len(messages) > 10:
        # Build conversation text from all messages
        conversation_text = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ])
        
        # Generate summary via LLM
        prompt = f"""Please summarize the conversation above, keeping only the most important information, 
        key topics, and any critical context that would be needed to continue the conversation.
        Provide a concise summary in 2-3 sentences."""
        
        result = llm.invoke(prompt.format(conversation=conversation_text))
        summary_content = result.content if hasattr(result, 'content') else str(result)
        
        # Return state update: add summary to state AND add summary message
        # The summary message gets added to the message list via LangGraph's add_messages
        # On next turn, the message list starts with the summary, keeping count low
        return {
            "summary": summary_content,
            "messages": [HumanMessage(content=summary_content)]
        }
    
    # No summary needed - return state unchanged
    return {
        "summary": summary,
        "messages": []
    }