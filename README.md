# Local Voice Agent Example

This is a modification of the [Cartesia Voice Agent](https://github.com/livekit-examples/cartesia-voice-agent) using faster_whisper, ollama, and xtts for local inference.

The example includes a custom Next.js frontend, Python agent and a modification of [xtts-streaming-server](https://github.com/coqui-ai/xtts-streaming-server).

## Running the example

### Prerequisites

- Node.js
- Python 3.9-3.12
- LiveKit Cloud account (or OSS LiveKit server)
- Ollama (for LLM)

### Frontend

Copy `.env.example` to `.env.local` and set the environment variables. Then run:

```bash
cd frontend
npm install
npm run dev
```

### Agent

Copy `.env.example` to `.env` and set the environment variables. Then run:

```bash
cd agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py dev
```

### TTS Server

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install --use-deprecated=legacy-resolver -r requirements.txt
python -m unidic download
python main.py
```