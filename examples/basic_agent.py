"""
Minimal Pipecat agent wired with SuperbrynObserver.

Run with:
    export SUPERBRYN_API_KEY="sb_live_..."
    export OPENAI_API_KEY="..."
    export DEEPGRAM_API_KEY="..."
    export CARTESIA_API_KEY="..."
    python examples/basic_agent.py
"""

import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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

    task = PipelineTask(
        pipeline,
        observers=[
            SuperbrynObserver(
                agent_name="basic-pipecat-bot",
                transport="fastapi_websocket",
            ),
        ],
        params=PipelineParams(
            enable_usage_metrics=True,
        ),
    )

    await PipelineRunner().run(task)
