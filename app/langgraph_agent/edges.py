def should_rewrite(state):

    if state["needs_rewrite"]:
        return "rewrite"

    return "rag"