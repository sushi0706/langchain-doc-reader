def print_response(response: dict):
    """
    Pretty print the response from the LLM/vector search.
    """
    print("\n=== Result ===")
    result = response.get("answer", "")
    if isinstance(result, str):
        for line in result.strip().split("\n"):
            print(line.strip())
    else:
        print(result)