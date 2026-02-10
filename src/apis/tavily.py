from tavily import TavilyClient

def search_tavily(query, "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"):
    """
    Searches the web using Tavily and returns structured results.
    """
    # Initialize the client
    tavily = TavilyClient(api_key=api_key)
    
    # Execute the search
    # search_depth="advanced" includes more context/content
    # max_results determines how many links you get back
    response = tavily.search(query=query, search_depth="advanced", max_results=3)
    
    return response['results']
