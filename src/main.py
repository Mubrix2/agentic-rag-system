from agent import build_agent

if __name__ == "__main__":
    agent_executor = build_agent()

    question = "What is the refund policy for digital products?"

    response = agent_executor.invoke({"input": question})

    print(response["output"])
