import asyncio
import numpy as np

from faster_whisper import WhisperModel
from typing import Optional

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.stt import STTCapabilities
from livekit.agents.utils import AudioBuffer


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: Optional[str] = "base",
        no_speech_prob: float = 0.4,
    ):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self._no_speech_prob = no_speech_prob
        self._model = WhisperModel(model, device="cpu", compute_type="int8")

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = await asyncio.to_thread(
                self._model.transcribe, 
                audio_np,
                language=language,
                no_speech_threshold=self._no_speech_prob)
            
            text: str = ""
            for segment in segments:
                if segment.no_speech_prob < self._no_speech_prob:
                    text += f"{segment.text} "
            
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(text=text or "", language=language or "")
                ],
            )
        except Exception as e:
            raise APIConnectionError() from e
