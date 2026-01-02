from langchain.agents import create_react_agent, AgentExecutor
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from tools import search_documents


def build_agent():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    tools = [search_documents]

    agent = create_react_agent(
        llm=llm,
        tools=tools
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3
    )
