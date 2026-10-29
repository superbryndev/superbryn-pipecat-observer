"""
Minimal Pipecat agent reported to SuperBryn with one attach() call.

Run with:
    export SUPERBRYN_API_KEY="sbryn_..."   # one key with test_data:write and obs:write
    export SUPERBRYN_AGENT_ID="..."        # the agent's SuperBryn ID (not needed with an agent key)
    export OPENAI_API_KEY="..."
    export DEEPGRAM_API_KEY="..."
    export CARTESIA_API_KEY="..."

`bot(runner_args)` is the entry point Pipecat's runner calls for each call. A SuperBryn test call is
reported for testing; every other call as a production call, recorded in stereo in the pipeline and
uploaded straight to S3. No carrier credentials required.
"""

import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

from superbryn_pipecat_observer import attach


async def run_bot(transport: FastAPIWebsocketTransport, runner_args) -> None:
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
    task = PipelineTask(pipeline, params=PipelineParams())

    # once per call, after the task is built and before it runs
    report = attach(task, runner_args, custom_data={"agent_name": "basic-pipecat-bot"})

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        report.update_custom_data({"hung_up_by_caller": True})
        await task.cancel()

    await PipelineRunner().run(task)
