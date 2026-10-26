# Women's Expert Voices

Development prototype for analysing representation in Spanish technology news.

## Current features

- Extract people, professional descriptions and quotations using Ollama.
- Check whether supporting evidence appears in the article.
- Review and edit people and quotations.
- Export original predictions and human corrections separately.

Predictions require human review. Matching evidence does not prove that a classification is correct. Aggregate metrics and the spaCy baseline are not implemented yet.

## Requirements

- Python 3.11
- Ollama installed and running
- The model specified below

## Setup on Windows

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Download the model if it is not already installed:
ollama pull hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q3_K_M

Keep Ollama running, then launch the application:
.\.venv\Scripts\python.exe -m streamlit run app.py

Open the Local URL shown in the terminal.
Reviewing an article
1. Enter the headline, introduction and article body.
2. Click Analyse article.
3. Inspect the original prediction and evidence warnings.
4. Correct the people and quotation tables.
5. Ensure each quotation's speaker_id matches a person_id.
6. Download the prediction and review.
Reviews are not saved automatically. Download them before analysing another article or closing the app.
Current limitations
- The model can miss people and quotations or invent unsupported labels.
- Evidence checks handle whitespace differences but do not validate meaning.
- The current model connection uses local Ollama only.
- The full team metrics and independent evaluation remain under development.
Do not commit credentials or copyrighted article collections without checking permissions.

```

## Version 2 — OVH gpt-oss-120b

The `v2` branch uses the organisers’ OVH API with gpt-oss-120b.
The local Ollama version remains on `main`.

To try v2:

```powershell
git fetch origin
git switch v2
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Enter the organisers’ API base URL (ending in `/v1`) and team token
in the sidebar. Ollama is not required for v2.

Never put the token in the code or commit it to GitHub.
Articles are sent to OVH for analysis. Human review is still required.

To return to the local version, stop Streamlit, run
`git switch main`, then restart the app.
