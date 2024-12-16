import asyncio
import json
import os
import requests

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, tokenize
from livekit.agents import tts as base_tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.log import logger
from livekit.plugins import silero, openai
from typing import List, Any

from dotenv import load_dotenv

from plugins import whisper, xtts


load_dotenv()


def prewarm(proc: JobProcess):
    # preload models when process starts to speed up first interaction
    proc.userdata["vad"] = silero.VAD.load()

    # fetch xtts voices
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.get("http://localhost:8020/studio_speakers", headers=headers)
    if response.status_code == 200:
        proc.userdata["xtts_voices"] = response.json()
    else:
        logger.warning(f"Failed to fetch XTTS voices: {response.status_code}")


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are a voice assistant created by LiveKit. Your interface with users will be voice. Pretend we're having a conversation, no special formatting or headings, just natural speech. Use short and concise responses.",
            )
        ]
    )

    xtts_voices: List[str] = ctx.proc.userdata["xtts_voices"]

    tts = xtts.TTS()

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=whisper.STT(),
        llm=openai.LLM.with_ollama(model="llama3.2"),
        tts=base_tts.StreamAdapter(
            tts=tts,
            sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
        ),
        chat_ctx=initial_ctx,
    )

    is_user_speaking = False
    is_agent_speaking = False

    @ctx.room.on("participant_attributes_changed")
    def on_participant_attributes_changed(
        changed_attributes: dict[str, str], participant: rtc.Participant
    ):
        # check for attribute changes from the user itself
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
            return

        if "voice" in changed_attributes:
            voice_id = participant.attributes.get("voice")
            logger.info(
                f"participant {participant.identity} requested voice change: {voice_id}"
            )
            if not voice_id:
                return

            if voice_id not in xtts_voices:
                logger.warning(f"Voice {voice_id} not found")
                return

            tts._opts.voice = voice_id
            # allow user to confirm voice change as long as no one is speaking
            if not (is_agent_speaking or is_user_speaking):
                asyncio.create_task(
                    agent.say("How do I sound now?", allow_interruptions=True)
                )

    await ctx.connect()

    @agent.on("agent_started_speaking")
    def agent_started_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = True

    @agent.on("agent_stopped_speaking")
    def agent_stopped_speaking():
        nonlocal is_agent_speaking
        is_agent_speaking = False

    @agent.on("user_started_speaking")
    def user_started_speaking():
        nonlocal is_user_speaking
        is_user_speaking = True

    @agent.on("user_stopped_speaking")
    def user_stopped_speaking():
        nonlocal is_user_speaking
        is_user_speaking = False

    # set voice listing as attribute for UI
    voices = []
    for voice in xtts_voices:
        voices.append(
            {
                "id": voice,
                "name": voice,
            }
        )
    voices.sort(key=lambda x: x["name"])
    await ctx.room.local_participant.set_attributes({"voices": json.dumps(voices)})

    agent.start(ctx.room)
    await asyncio.sleep(3)
    await agent.say("Hi there, how are you doing today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
