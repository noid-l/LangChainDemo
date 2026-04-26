import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

def test_deepseek_reasoning():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL")
    
    print(f"Testing model: {model}")
    print(f"Base URL: {base_url}")
    
    llm = ChatDeepSeek(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.2
    )
    
    res = llm.invoke("你好，请简要介绍一下济南。")
    print("-" * 20)
    print("Content:", res.content)
    print("Additional Kwargs:", res.additional_kwargs)
    print("Response Metadata:", res.response_metadata)

if __name__ == "__main__":
    test_deepseek_reasoning()
