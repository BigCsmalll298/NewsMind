from newsmind_agent import NewsMindAgent

if __name__ == "__main__":
    agent = NewsMindAgent(
        use_collector=True,
        max_articles=5,
        clear_log=True,
    )
    agent.run("最近的新闻？")
