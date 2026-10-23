"""
Real-pipeline harness for the SuperBryn Pipecat SDK — runs fully on this machine.

A real Pipecat pipeline (aggregators, function calling, output transport,
audio recorder) runs a scripted support call with stand-in STT / LLM / TTS.
Both SDK paths are attached, as a customer would: the obs observer
(`SuperbrynObserver`) and the evals path (`attach_test_data`). They ship to a
local catcher instead of SuperBryn, and a checklist compares what each sent
against what actually happened.

Run (no API keys, no network, ~20s per scenario):
    uv run python examples/local_harness/run.py                 # every scenario
    uv run python examples/local_harness/run.py -s barge_in     # one scenario
    uv run python examples/local_harness/run.py --dump out/     # save payloads
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from catcher import Catcher  # noqa: E402
from checklist import Check, Expected, print_report, print_summary, run_checks  # noqa: E402
from loguru import logger as loguru_logger  # noqa: E402
from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.frames.frames import EndFrame, TTSSpeakFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies  # noqa: E402
from scripted_services import (  # noqa: E402
    CallTool,
    GroundTruth,
    LocalMockTransport,
    Say,
    ScriptedCaller,
    ScriptedLLMService,
    ScriptedSTTService,
    ScriptedTTSService,
)

from superbryn_pipecat_observer import SuperbrynObserver, attach_test_data  # noqa: E402

API_KEY = "sb_local_harness"
AGENT_ID = "6f1c2a8e-3b4d-4e5f-8a9b-0c1d2e3f4a5b"
AGENT_NAME = "local-harness-bot"
CUSTOM_METADATA = {"campaign": "harness", "tenant_id": "acme-local"}

SYSTEM_PROMPT = "You are Acme's support agent. Look up orders with lookup_order."
GREETING = "Hello, thanks for calling Acme support. How can I help you today?"
USER_LINES = [
    "Hi, I need help with my recent order.",
    "It's order number 884291.",
    "Great, thanks for your help.",
]
FIRST_REPLY = "Sure, I can help with that. What's your order number?"
# Second sentence is never heard: the caller talks over the first one
INTERRUPTED_REPLY = (
    "Sure, I can help with that. Could you read me the order number from your confirmation email?"
)
ORDER_RESULT = {"order_id": "884291", "status": "shipped", "eta": "Friday"}
STDLIB_LOG_LINE = "lookup_order called via logging"
LOGURU_LOG_LINE = "lookup_order called via loguru"


def llm_steps(first_reply: str) -> list[Say | CallTool]:
    return [
        Say(first_reply, 180, 14),
        CallTool("lookup_order", {"order_id": "884291"}, 210, 18),
        Say("Your order 884291 has shipped and should arrive on Friday.", 260, 16),
        Say("You're welcome. Have a great day. Goodbye!", 290, 11),
    ]


@dataclass
class Scenario:
    name: str
    about: str
    word_timestamps: bool = False
    own_params: bool = False
    barge_in: set[int] = field(default_factory=set)
    token_delay: float = 0.01
    first_reply: str = FIRST_REPLY


SCENARIOS = [
    Scenario("default", "README setup, HTTP-style TTS (one text frame per sentence)"),
    Scenario("word_tts", "TTS sends words in playback order (Cartesia / ElevenLabs)", True),
    Scenario(
        "own_params",
        "customer passes PipelineParams(enable_metrics=True) to the task",
        own_params=True,
    ),
    Scenario(
        "barge_in",
        "caller talks over the first reply; its second sentence is never heard",
        word_timestamps=True,
        barge_in={1},
        first_reply=INTERRUPTED_REPLY,
    ),
    Scenario(
        "slow_llm",
        "LLM still streaming when the bot starts speaking",
        word_timestamps=True,
        token_delay=0.12,
    ),
]


def expected_for(scenario: Scenario) -> Expected:
    spoken = [GREETING, scenario.first_reply, *[s.text for s in llm_steps("")[2:]]]
    unspoken: list[str] = []
    cut_off = None
    if scenario.barge_in:
        cut_off = 1
        first, second = INTERRUPTED_REPLY.split(". ", 1)
        spoken[1] = first + "."
        unspoken.append(second)
    return Expected(
        api_key=API_KEY,
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        custom_metadata=CUSTOM_METADATA,
        agent_lines=spoken,
        unspoken=unspoken,
        cut_off=cut_off,
        interruptions=len(scenario.barge_in),
        usage_metrics_requested=not scenario.own_params,
        log_lines={"logging": STDLIB_LOG_LINE, "loguru": LOGURU_LOG_LINE},
    )


async def run_call(scenario: Scenario, dump: Path | None = None) -> list[Check]:
    catcher = Catcher()
    await catcher.start()
    truth = GroundTruth()
    task = None

    async def end_call() -> None:
        await task.queue_frame(EndFrame())

    async def lookup_order(params: FunctionCallParams) -> None:
        # Agents log from tools: Pipecat apps mostly use loguru, some use logging
        logging.getLogger("agent.tools").info(STDLIB_LOG_LINE)
        loguru_logger.info(LOGURU_LOG_LINE)
        truth.tool_calls[-1]["result"] = ORDER_RESULT
        await params.result_callback(ORDER_RESULT)

    caller = ScriptedCaller(USER_LINES, truth, end_call, barge_in=scenario.barge_in)
    transport = LocalMockTransport(caller)
    stt = ScriptedSTTService(USER_LINES)
    llm = ScriptedLLMService(
        llm_steps(scenario.first_reply), truth, token_delay=scenario.token_delay
    )
    llm.register_function("lookup_order", lookup_order)
    tts = ScriptedTTSService(truth, word_timestamps=scenario.word_timestamps)

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup_order",
                    description="Look up an order by its number.",
                    properties={"order_id": {"type": "string"}},
                    required=["order_id"],
                )
            ]
        ),
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    observer = SuperbrynObserver(
        agent_id=AGENT_ID,
        agent_name=AGENT_NAME,
        api_key=API_KEY,
        webhook_url=catcher.webhook_url,
        api_base_url=catcher.base_url,
        custom_metadata=CUSTOM_METADATA,
    )
    extra = {"params": PipelineParams(enable_metrics=True)} if scenario.own_params else {}
    task = observer.monitor_and_create_task(pipeline, transport=transport, **extra)
    attach_test_data(
        task,
        api_key=API_KEY,
        agent_id=AGENT_ID,
        base_url=catcher.base_url,
        superbryn_call_id=f"harness-{scenario.name}",
    )
    await task.queue_frames([TTSSpeakFrame(GREETING)])

    started = time.monotonic()
    try:
        await asyncio.wait_for(PipelineRunner(handle_sigint=False).run(task), timeout=90)
    except TimeoutError:
        truth.anomalies.append("call did not finish within 90s")
    wall_seconds = time.monotonic() - started
    await catcher.stop()

    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        out = {"obs": catcher.caught.webhooks, "evals": catcher.caught.test_events}
        (dump / f"{scenario.name}.json").write_text(json.dumps(out, indent=2, default=str))

    return run_checks(
        catcher.caught,
        truth,
        expected_for(scenario),
        session_id=observer.session_id,
        wall_seconds=wall_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-s", "--scenario", choices=[s.name for s in SCENARIOS], action="append")
    parser.add_argument("--dump", type=Path, help="write each scenario's payloads to this folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Pipecat logs")
    args = parser.parse_args()
    if not args.verbose:
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="ERROR")
    logging.basicConfig(level=logging.WARNING)

    chosen = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    results: dict[str, list[Check]] = {}
    for scenario in chosen:
        print(f"\n=== {scenario.name}: {scenario.about}")
        results[scenario.name] = asyncio.run(run_call(scenario, args.dump))
        print_report(results[scenario.name])
    if len(results) > 1:
        print_summary(results)
    sys.exit(0 if all(c.ok for checks in results.values() for c in checks) else 1)


if __name__ == "__main__":
    main()
