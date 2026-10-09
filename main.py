import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from typing import TypedDict, Any, Optional, Literal
from pydantic import BaseModel
from tavily import TavilyClient
from langgraph.graph import END, StateGraph, START

load_dotenv()

api_key = os.getenv("API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("Sorry Invalid API KEY")

if not tavily_api_key:
    raise ValueError("Sorry Invalid Tavily API KEY")


client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"
tavily = TavilyClient(api_key=tavily_api_key)


# initialize the state of the research agent
class ResearchState(TypedDict):
    user_query: str
    query_analysis: dict[str, Any]
    search_results: list[dict[str, Any]]
    calculation_result: str
    final_result: str
    cited_result: str
    citation_sources: list[dict[str, str]]
    search_status: str


# create a pydantic model
class QueryAnalysis(BaseModel):
    topic: str
    intent: str
    key_points: list[str]
    search_queries: list[str]
    answer_requirements: list[str]
    tool: Literal["web_search", "news_search", "calculator"]
    calculation_expression: Optional[str] = None




# Helper functions/tools


def ask_llm(user_prompt, system_prompt=None):

    if system_prompt is None:
        system_prompt = (
            "Analyze the user's question and produce a structured research plan."
        )

    system_message = {"role": "system", "content": system_prompt}
    user_message = {"role": "user", "content": user_prompt}

    messages = [system_message, user_message]

    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer


def check_search_status(state: ResearchState):
    if state["search_status"] == "success":
        return "process_sources"
    return "end"


def check_calculation_status(state: ResearchState):
    if state["search_status"] == "success":
        return "generate_result"
    return "end"


def route_tool(state: ResearchState):
    tool = state["query_analysis"]["tool"]

    if tool == "web_search":
        return "web_search"

    if tool == "news_search":
        return "news_search"

    if tool == "calculator":
        return "calculator"

    return "end"


# making nodes
def analyze_query(state: ResearchState):
    print("\n[Agent] Analyzing your research question...")

    system_prompt = """
        You are a research query analyzer.

        Your job is to analyze the user's question and create a structured research plan.

        You have access to the following tools:

        1. web_search
        - Use this for general information that requires searching the internet.
        - Examples:
            - "What is Django?"
            - "How does React work?"
            - "Explain the history of Python."

        2. news_search
        - Use this for recent, current, latest, or news-related information.
        - Examples:
            - "What are the latest AI developments?"
            - "What happened in the latest OpenAI news?"
            - "What are today's major technology news stories?"

        3. calculator
        - Use this for mathematical calculations or numerical expressions.
        - Examples:
            - "What is 125 * 8 + 50?"
            - "Calculate 15% of 500."
            - "What is (100 + 50) * 2?"

        Tool selection rules:

        - Choose web_search for general research questions.
        - Choose news_search when the user asks about recent, current, latest,
        today's, or news-related information.
        - Choose calculator when the user's question requires mathematical calculation.
        - Choose exactly ONE tool for each user question.
        - Do not choose calculator for questions that only mention numbers but do not
        require calculation.
        - Do not choose news_search unless the question requires recent or
        news-related information.

        Return ONLY valid JSON.

        The JSON must contain exactly these fields:

        {
            "topic": "string",
            "intent": "string",
            "key_points": ["string"],
            "search_queries": ["string"],
            "answer_requirements": ["string"],
            "tool": "web_search",
            "calculation_expression": null
        }

        The "tool" field must contain exactly ONE of these values:

        - "web_search"
        - "news_search"
        - "calculator"

        For calculator queries:
        - Set "tool" to "calculator".
        - Put the mathematical expression in "calculation_expression".
        - Set "search_queries" to an empty list.
        - Set "calculation_expression" to null for non-calculator queries.

        Do not invent tool names.

        Do not include markdown.
        Do not include ```json.
        Do not include explanations outside the JSON.
    """

    user_prompt = state["user_query"]

    result = ask_llm(user_prompt=user_prompt, system_prompt=system_prompt)

    try:
        # convert llm result into json response
        data = json.loads(result)

        # validate the json response and create a pydantic object
        query_analysis = QueryAnalysis.model_validate(data)

        # convert pydantic object into dictionary
        query_analysis_dict = query_analysis.model_dump()

        print("[Agent] Research question analyzed successfully.")

        return {"query_analysis": query_analysis_dict}

    except Exception as e:
        raise ValueError(f"Invalid LLM response: {e}")


def web_search(state: ResearchState):
    print("\n[Agent] Searching the web for relevant information...")

    queries = state["query_analysis"]["search_queries"]
    search_results = []

    try:
        for query in queries:
            print(f"[Agent] Searching for: {query}")

            response = tavily.search(query=query)
            results = response.get("results", [])

            for result in results:
                if len(search_results) >= 10:
                    break

                search_results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:2000],
                    }
                )

            if len(search_results) >= 10:
                break

        if search_results:
            print(
                f"[Agent] Web search completed. "
                f"Found {len(search_results)} relevant results."
            )

            return {"search_results": search_results, "search_status": "success"}

        print("[Agent] No search results found.")

        return {"search_results": [], "search_status": "no_results"}

    except Exception as e:
        raise ValueError(f"Invalid web search response: {e}")


def news_search(state: ResearchState):
    print("\n[Agent] Searching the web for relevant news...")

    queries = state["query_analysis"]["search_queries"]
    search_results = []

    try:
        for query in queries:
            response = tavily.search(
                query=query,
                topic="news",
            )

            results = response.get("results", [])

            for result in results:
                if len(search_results) >= 10:
                    break

                search_results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:2000],
                    }
                )

            if len(search_results) >= 10:
                break

        if search_results:
            print(
                f"[Agent] News search completed. "
                f"Found {len(search_results)} relevant results."
            )

            return {"search_results": search_results, "search_status": "success"}

        print("[Agent] No search results found.")

        return {"search_results": [], "search_status": "no_results"}

    except Exception as e:
        raise ValueError(f"Invalid news search response: {e}")


def calculator(state: ResearchState):
    print("\n[Agent] Performing calculation...")

    expression = state["query_analysis"].get("calculation_expression", "")

    if not expression:
        return {"calculation_result": "No calculation expression was provided."}

    try:
        # Only allow basic mathematical characters
        allowed_characters = "0123456789+-*/().% "

        if not all(char in allowed_characters for char in expression):
            raise ValueError("Invalid characters in calculation expression.")

        result = eval(expression, {"__builtins__": {}}, {})

        print(f"[Agent] Calculation completed: {result}")

        if result:
            return {"calculation_result": str(result), "search_status": "success"}

        return {"calculation_result": "", "search_status": "no_results"}

    except Exception as e:
        print("[Agent] Calculation failed.")

        return {
            "calculation_result": f"Calculation error: {e}",
            "search_status": "no_results",
        }


def remove_duplicates(state: ResearchState):
    print("\n[Agent] Removing duplicate sources...")

    search_results = state["search_results"]

    unique_results = []
    seen_urls = set()

    for result in search_results:
        url = result.get("url", "").strip()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        unique_results.append(result)

    print(
        f"[Agent] Removed {len(search_results) - len(unique_results)} duplicate sources."
    )

    return {"search_results": unique_results}


def clean_sources(state: ResearchState):
    print("\n[Agent] Cleaning research sources...")

    search_results = state["search_results"]

    cleaned_results = []

    for result in search_results:
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()

        if not title or not url or not content:
            continue

        cleaned_results.append(
            {
                "title": title,
                "url": url,
                "content": content,
            }
        )

    print(
        f"[Agent] Source cleaning completed {len(cleaned_results)} valid sources remain from {len(search_results)}."
    )

    return {"search_results": cleaned_results}


def limit_content(state: ResearchState):
    print("\n[Agent] Limiting source content...")

    search_results = state["search_results"]

    limited_results = []

    for result in search_results:
        content = result.get("content", "")
        title = result.get("title", "")
        url = result.get("url", "")

        limited_results.append(
            {
                "title": title,
                "url": url,
                "content": content[:1200],
            }
        )

    print(
        f"[Agent] Source content has been limited to {len(limited_results)} from {len(search_results)} sources."
    )

    return {"search_results": limited_results}


def filter_relevant_sources(state: ResearchState):
    print("\n[Agent] Filtering irrelevant sources...")

    search_results = state["search_results"]
    user_query = state["user_query"]

    query_words = set(word.lower() for word in user_query.split() if len(word) > 3)

    relevant_results = []

    for result in search_results:
        text = (result.get("title", "") + " " + result.get("content", "")).lower()

        matches = 0

        for word in query_words:
            if word in text:
                matches += 1

        if matches > 0:
            relevant_results.append(result)

    print(
        f"[Agent] Relevance filtering completed. "
        f"{len(relevant_results)} relevant sources remain."
    )

    return {"search_results": relevant_results}


def generate_result(state: ResearchState):
    print("\n[Agent] Generating the final research answer...")

    system_prompt = """
        You are a research assistant.

        Use the provided research information to answer the user's question.

        I will send you user_query which will be question that user will be asking, query_analysis a structure
        variant or form of the user_query, and search_results a dictionary or key value pair that contain all results of the
        queries that find through the web search.

        Base your answer on the search results when relevant.
        Do not invent facts that are not supported by the available information.
        Give a clear and useful answer.
    """

    user_query = state["user_query"]
    query_analysis = state["query_analysis"]
    search_results = state["search_results"]

    # make the state readable texts
    query_analysis_text = json.dumps(query_analysis)
    search_results_text = json.dumps(search_results)

    user_prompt = f"""
        User question:
        {user_query}

        Research plan:
        {query_analysis_text}

        Web search results:
        {search_results_text}
    """

    final_result = ask_llm(user_prompt=user_prompt, system_prompt=system_prompt)

    print("[Agent] Final research answer generated successfully.")

    return {"final_result": final_result}


def add_citations(state: ResearchState):
    print("\n[Agent] Adding citations to the research answer...")

    final_result = state["final_result"]
    search_results = state.get("search_results", [])

    # Calculator-only answers may not have web sources.
    if not search_results:
        print("[Agent] No sources available for citations.")

        return {"cited_result": final_result, "citation_sources": []}

    # Give every source a stable number.
    numbered_sources = []

    for index, source in enumerate(search_results, start=1):
        numbered_sources.append(
            {
                "id": index,
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "content": source.get("content", ""),
            }
        )

    system_prompt = """
    You are a research citation assistant.

    Your job is to add citations to the provided research answer.

    Rules:
    1. Use only the supplied sources.
    2. Each source has a unique numeric ID.
    3. Add citations such as [1] or [2] after factual claims
       that are supported by the corresponding source.
    4. Only cite a source when its content supports the claim.
    5. Never invent source IDs, URLs, titles, or facts.
    6. If a claim is not supported by the sources, do not
       attach a citation to it.
    7. Preserve the meaning and structure of the original answer.
    8. Return only the cited answer.
    9. Do not generate a references section. It will be
       constructed separately from the supplied source data.
    """

    user_prompt = f"""
    Original user question:
    {state["user_query"]}

    Research answer:
    {final_result}

    Available sources:
    {json.dumps(numbered_sources, ensure_ascii=False)}
    """

    try:
        cited_result = ask_llm(user_prompt=user_prompt, system_prompt=system_prompt)

        # Find citation numbers used in the answer.
        used_ids = {int(match) for match in re.findall(r"\[(\d+)\]", cited_result)}

        # Keep only references that correspond to real sources.
        valid_ids = {source["id"] for source in numbered_sources}

        # Remove invalid citation markers, if the LLM produced any.
        cited_result = re.sub(
            r"\[(\d+)\]",
            lambda match: (match.group(0) if int(match.group(1)) in valid_ids else ""),
            cited_result,
        )

        citation_sources = [
            {"title": source["title"], "url": source["url"]}
            for source in numbered_sources
            if source["id"] in used_ids and source["id"] in valid_ids
        ]

        # Add references using the original source metadata.
        if citation_sources:
            references = "\n\n## References\n"

            for index, source in enumerate(numbered_sources, start=1):
                if source["id"] not in used_ids:
                    continue

                if source["id"] not in valid_ids:
                    continue

                references += f'\n[{index}] {source["title"]} - {source["url"]}'

            cited_result += references

        print("[Agent] Citations added successfully.")

        return {"cited_result": cited_result, "citation_sources": citation_sources}

    except Exception as e:
        print(f"[Agent] Citation generation failed: {e}")
        raise ValueError(f"Failed to add citations: {e}") from e


def generate_report(state: ResearchState):
    print("\n[Agent] Generating the final research report...")

    cited_result = state.get("cited_result", "")
    user_query = state["user_query"]

    # Make sure there is an answer to format.
    if not cited_result.strip():
        return {
            "final_report": "Unable to generate a report because no research answer is available."
        }

    system_prompt = """
    You are a professional research report writer.

    Convert the provided answer into a clear, well-organized report.

    Follow these rules:
    1. Use a suitable title based on the user's question.
    2. Start with a short introduction.
    3. Organize the main information under meaningful headings.
    4. Use bullet points or numbered lists when helpful.
    5. Add a conclusion or summary at the end.
    6. Preserve all existing inline citations, such as [1] and [2].
    7. Preserve the References section and its original URLs.
    8. Do not invent facts, statistics, sources, or URLs.
    9. Do not add new factual claims that are absent from the provided answer.
    10. If the answer is a simple calculation, keep the report short and
        clearly show the result instead of forcing a long report.
    11. Return the report in Markdown format only.
    """

    user_prompt = f"""
    User's original question:
    {user_query}

    Cited research answer:
    {cited_result}

    Format this answer as the final research report.
    Preserve the existing citations and references.
    """

    try:
        final_report = ask_llm(user_prompt=user_prompt, system_prompt=system_prompt)

        print("[Agent] Final research report generated successfully.")

        return {"final_result": final_report}

    except Exception as e:
        print(f"[Agent] Report generation failed: {e}")
        raise ValueError(f"Failed to generate research report: {e}") from e


# initilize the state
builder = StateGraph(ResearchState)

# add nodes in the graph
builder.add_node("analyze_query", analyze_query)
builder.add_node("web_search", web_search)
builder.add_node("generate_result", generate_result)
builder.add_node("news_search", news_search)
builder.add_node("calculator", calculator)
builder.add_node("remove_duplicates", remove_duplicates)
builder.add_node("clean_sources", clean_sources)
builder.add_node("limit_content", limit_content)
builder.add_node("filter_relevant_sources", filter_relevant_sources)
builder.add_node("add_citations", add_citations)
builder.add_node("generate_report", generate_report)

# create the entry point for the graph
builder.add_edge(START, "analyze_query")

# connect nodes using conditional edge
builder.add_conditional_edges(
    "analyze_query",
    route_tool,
    {
        "web_search": "web_search",
        "news_search": "news_search",
        "calculator": "calculator",
        "end": END,
    },
)


# connect web_search to remove_duplicates using coniditional edge
builder.add_conditional_edges(
    "web_search",
    check_search_status,
    {"process_sources": "remove_duplicates", "end": END},
)


# connect news_search to remove_duplicates using coniditional edge
builder.add_conditional_edges(
    "news_search",
    check_search_status,
    {"process_sources": "remove_duplicates", "end": END},
)


# connect calculator to remove_duplicates using coniditional edge
builder.add_conditional_edges(
    "calculator",
    check_calculation_status,
    {"generate_result": "generate_result", "end": END},
)


# connect remove_duplicates to clean_sources using edge
builder.add_edge("remove_duplicates", "clean_sources")


# connect clean_sources to limit_content using edge
builder.add_edge("clean_sources", "limit_content")


# connect limit_content to filter_relevant_sources using edge
builder.add_edge("limit_content", "filter_relevant_sources")


# connect filter_relevant_sources to generate_result using edge
builder.add_edge("filter_relevant_sources", "generate_result")


# connect generate_result to add_citations
builder.add_edge("generate_result", "add_citations")


# connect add_citations to generate_report
builder.add_edge("add_citations", "generate_report")

# compile the graph
graph = builder.compile()

# invoke the graph and get final result
if __name__ == "__main__":
    user_query = input("Ask your research question: ")

    initial_state = {
        "user_query": user_query,
        "query_analysis": {},
        "search_results": [],
        "final_result": "",
        "search_status": "",
        "calculation_result": "",
        "cited_result": "",
        "citation_sources": []
    }

    result = graph.invoke(initial_state)

    print("\n===== FINAL RESULT =====\n")
    print(result["final_result"])

