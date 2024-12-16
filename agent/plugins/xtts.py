import aiohttp
import asyncio

from dataclasses import dataclass

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)


@dataclass
class _TTSOptions:
    base_url: str
    language: str
    voice: str

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8020",
        voice: str = "Damien Black",
        language: str = "en",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=24_000,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            base_url=base_url,
            language=language,
            voice=voice,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session())

class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""
    def __init__(
        self, 
        *,
        tts: TTS,
        text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        session: aiohttp.ClientSession
    ) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=24_000, num_channels=1
        )

        data = {
            "text": self._input_text,
            "language": self._opts.language,
            "speaker": self._opts.voice,
            "add_wav_header": True,
            "stream_chunk_size": 150,
        }

        try:
            async with self._session.post(
                url=f"{self._opts.base_url}/tts_stream",
                json=data,
            ) as resp:
                async for bytes_data, _ in resp.content.iter_chunks():
                    for frame in bstream.write(bytes_data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=frame,
                        )
                    )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
