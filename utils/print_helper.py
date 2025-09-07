def print_response(response):
    """
    Pretty print the response from the LLM/vector search.
    """
    print("\n=== Result ===")
    result = response.content

    if isinstance(result, str):
        for line in result.strip().split("\n"):
            print(line.strip())
    else:
        print(result)