# 0) unzip, cd into folder
unzip fleet_insights_rag.zip
cd fleet_insights_rag

# 1) venv & install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) API key
cp .env.example .env
# edit .env to set OPENAI_API_KEY

# 3) add docs
# put PDFs or .txt into data/raw/

# 4) build index
python -m src.ingest

# 5) run app
cd /Users/ywchen/project/fleet_insights_rag
PYTHONPATH=. streamlit run src/app.py
