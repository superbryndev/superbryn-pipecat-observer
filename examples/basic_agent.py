"""
Minimal Pipecat agent wired with SuperbrynObserver.

Run with:
    export SUPERBRYN_API_KEY="sb_live_..."
    export SUPERBRYN_WEBHOOK_URL="https://api.superbryn.com/webhooks/obs/pipecat"
    export OPENAI_API_KEY="..."
    export DEEPGRAM_API_KEY="..."
    export CARTESIA_API_KEY="..."
    python examples/basic_agent.py

The observer captures audio in-pipeline (transport-agnostic), uploads
the WAV directly to S3 via a presigned PUT URL fetched from the
SuperBryn API, then POSTs the call payload to the webhook URL when
the session ends. No carrier credentials required.
"""

import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport

from superbryn_pipecat_observer import SuperbrynObserver


async def main(transport: FastAPIWebsocketTransport) -> None:
    pipeline = Pipeline(
        [
            transport.input(),
            DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"]),
            OpenAILLMService(
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4o-mini",
            ),
            CartesiaTTSService(
                api_key=os.environ["CARTESIA_API_KEY"],
                voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
            ),
            transport.output(),
        ]
    )

    observer = SuperbrynObserver(
        agent_id=os.environ.get("SUPERBRYN_AGENT_ID", "basic-pipecat-bot"),
        agent_name="basic-pipecat-bot",
    )

    task = observer.observe_and_create_task(
        pipeline,
        transport=transport,
    )

    await PipelineRunner().run(task)
