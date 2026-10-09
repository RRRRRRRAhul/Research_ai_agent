## Project Overview

An AI-powered research agent built with LangGraph that analyzes user queries, searches the web or recent news, performs calculations, and generates structured research reports with citations.

## Features

- Web search for relevant information.
- News search for recent updates.
- Calculator for mathematical expressions.
- Source filtering to remove duplicates and irrelevant content.
- Automatic citations for research findings.
- Structured research report generation in Markdown.


## Tech Stack

- **Python** – Core programming language.
- **LangGraph** – Agent workflow orchestration.
- **Groq API** – LLM-powered query analysis and report generation.
- **Tavily API** – Web and news search.
- **Pydantic** – Structured data validation.
- **python-dotenv** – Environment variable management.

## Project Workflow

1. **Query Analysis** – Analyzes the user's query and identifies the required tool.
2. **Tool Selection** – Routes the query to web search, news search, or calculator.
3. **Source Processing** – Removes duplicate sources, cleans content, and filters irrelevant results.
4. **Answer Generation** – Generates an answer using the collected information or calculation result.
5. **Citation Addition** – Adds citations to research claims using available sources.
6. **Report Generation** – Formats the cited answer into a structured Markdown report.


## Project Structure

```text
ai-research-agent/
├── main.py              # Main application and LangGraph workflow
├── .env                 # API keys and environment variables
├── .env.example         # Example environment configuration
├── .gitignore           # Files excluded from Git
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```


## Prerequisites

- **Python 3.10+** installed.
- **Groq API key** for LLM access.
- **Tavily API key** for web and news search.
- Required Python packages listed in `requirements.txt`.
- Basic knowledge of Python and environment variables.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-research-agent
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

1. Create a `.env` file in the project root directory.
2. Add your API keys:
   ```env
   API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
3. Load environment variables using `python-dotenv`.
4. Never commit your `.env` file to GitHub.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Enter your research query when prompted.
3. The agent selects the appropriate tool and processes the results.
4. View the generated research report with citations in the terminal.

## Example Output

**Sample Query:**
```text
How is artificial intelligence transforming healthcare?
```

**Generated Research Report:**
```markdown
# AI in Healthcare

## Introduction
AI is helping healthcare professionals improve diagnosis and patient care.

## Key Findings
- AI assists in medical image analysis.
- Predictive models help identify health risks.
- AI supports clinical decision-making.

## Conclusion
AI has the potential to improve healthcare efficiency and outcomes.

## References
[1] Example Source - https://example.com
```

## Error Handling

- **Missing Search Results:** Detects empty search results and stops the research workflow.
- **Invalid Calculations:** Catches calculation errors and returns an appropriate error message.
- **API Errors:** Handles API failures and reports errors during search or LLM requests.
- **Input Validation:** Uses Pydantic to validate structured query analysis data.

## Limitations

- **Search Quality:** Results depend on the relevance and availability of web sources.
- **Token Limits:** Large search results may exceed the LLM's token limits.
- **Citation Accuracy:** Citations may not always fully support the associated claims.
- **Source Filtering:** Keyword-based filtering may exclude relevant information.
- **Calculation Support:** The calculator supports only the mathematical expressions implemented in the agent.

## Future Improvements

- **Better Relevance Ranking:** Improve source selection using semantic search.
- **Citation Verification:** Verify that citations accurately support research claims.
- **Report Export:** Add PDF and DOCX export options.
- **Improved Error Handling:** Strengthen recovery from API failures and invalid inputs.
- **Enhanced Reports:** Support customizable report formats and more detailed analysis.
