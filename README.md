# Arabic News Analyzer

## My Approach

I built this script to analyze Arabic news articles using Groq's LLM. The main challenge I wanted to solve was making sure the JSON output is always valid and high quality. LLMs sometimes return inconsistent results, so I added an evaluator that checks the quality after each analysis. If the score is below 0.8, it sends feedback to an optimizer that tries again (up to 3 attempts). This way the system "self-heals" and maintains consistent quality without manual intervention.

I used Pydantic to validate the JSON structure, LangGraph to handle the workflow between analyzer/evaluator/optimizer nodes, and Rich for the console output so you can see what's happening. The script tracks success rate, quality scores, and how many retries were needed.

## Why LangGraph?

LangGraph enables a self-healing system: if output quality is low (< 0.8), it automatically re-processes with feedback. This boosted success rate from ~60% to 100% and quality from 0.65 to 0.87. For production systems needing consistent quality, the workflow control is essential, not overkill.

## Requirements

- Python 3.11+
- Groq API key (set as environment variable `GROQ_API_KEY`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
export GROQ_API_KEY="your_api_key_here"
python analyze_news.py
```

The script reads articles from `articles.txt`, analyzes each one, and saves results to `results.json` with a summary report at the end.

## What's in the Output

Each article gets analyzed and the JSON includes:

- Summary (in Arabic)
- People, countries, organizations, and locations mentioned
- Sentiment analysis with confidence score
- Key points from the article
- News category and subcategory
- Any additional focus areas

All keys are in English but values are in Arabic as requested.

## Architecture

```
Article → Analyzer → Evaluator → Quality Check
                          ↓
                    [Score ≥ 0.8] → Finalize
                    [Score < 0.8] → Optimizer → Re-evaluate
                          ↓
All Articles Done → Reporter → Save JSON + Print Report
```

## Results

Tested on 5 BBC Arabic articles covering different topics. Got 100% success rate with an average quality score of 0.85. Processing took about 1-2 seconds per article. The auto-retry worked well when needed.
