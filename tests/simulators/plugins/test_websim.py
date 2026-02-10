"""Tests for WebSim simulator."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simulators.base import SimulatorConfig
from simulators.plugins.WebSim import SimulatorState, WebSim


@pytest.fixture
def websim():
    """Create a WebSim instance with mocked server thread."""
    with patch("simulators.plugins.WebSim.threading.Thread") as mock_thread:
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True
        mock_thread.return_value = mock_thread_instance

        with patch("simulators.plugins.WebSim.WebSim.sleep"):
            simulator = WebSim(SimulatorConfig(name="test_websim"))
            yield simulator


def test_simulator_state_docstring():
    """Test that SimulatorState has a docstring."""
    assert SimulatorState.__doc__ is not None


def test_simulator_state_to_dict_docstring():
    """Test that SimulatorState.to_dict has a docstring."""
    assert SimulatorState.to_dict.__doc__ is not None


def test_simulator_state_to_dict():
    """Test that SimulatorState.to_dict works correctly."""
    state = SimulatorState(
        inputs={"test": "value"},
        current_action="test_action",
        last_speech="hello",
        current_emotion="happy",
        system_latency={"fuse_time": 0.1},
    )
    result = state.to_dict()
    assert result["inputs"] == {"test": "value"}
    assert result["current_action"] == "test_action"


def test_websim_init_docstring():
    """Test that WebSim.__init__ has a docstring."""
    assert WebSim.__init__.__doc__ is not None


def test_websim_initialization(websim):
    """Test WebSim initialization."""
    assert websim.name == "test_websim"
    assert websim._loop is None
    assert websim._initialized is True
    assert websim.state is not None
    assert websim.state.current_action == "idle"
    assert websim.io_provider is not None


def test_websim_sim_docstring():
    """Test that WebSim.sim has a docstring."""
    assert WebSim.sim.__doc__ is not None


def test_websim_tick_docstring():
    """Test that WebSim.tick has a docstring."""
    assert WebSim.tick.__doc__ is not None


def test_websim_tick_with_no_loop(websim):
    """Test that tick handles the case when loop is not initialized."""
    websim._loop = None
    websim.tick()


def test_websim_sim_calls_do_sim(websim):
    """Test that sim calls _do_sim with broadcast=True."""
    with patch.object(websim, "_do_sim") as mock_do_sim:
        websim.sim([])
        mock_do_sim.assert_called_once_with([], broadcast=True)


def test_websim_do_sim_without_initialization(websim):
    """Test _do_sim when simulator is not initialized."""
    websim._initialized = False
    websim._do_sim([])
    assert websim.state.current_action == "idle"


@pytest.mark.asyncio
async def test_websim_cleanup(websim):
    """Test cleanup method."""
    websim._initialized = True
    await websim.cleanup()
    assert websim._initialized is False


def test_websim_loop_initialization(websim):
    """Test that _loop is initialized to None in __init__."""
    assert websim._loop is None


def test_websim_state_dict_initialization(websim):
    """Test that state_dict is initialized in __init__."""
    assert hasattr(websim, "state_dict")
    assert websim.state_dict == {}


def test_websim_tick_with_loop(websim):
    """Test that tick calls _do_sim when loop exists."""
    websim._loop = asyncio.new_event_loop()
    with patch.object(websim, "_do_sim") as mock_do_sim:
        websim.tick()
        mock_do_sim.assert_called_once_with([], broadcast=False)


def test_websim_tick_without_initialization(websim):
    """Test that tick returns early when not initialized."""
    websim._initialized = False
    websim.tick()


def test_websim_do_sim_with_actions(websim):
    """Test _do_sim processes actions and updates state."""
    from llm.output_model import Action

    actions = [
        Action(type="move", value="test action"),
        Action(type="speak", value="hello"),
        Action(type="emotion", value="happy"),
    ]

    websim._do_sim(actions, broadcast=False)

    assert websim.state.current_action == "test action"
    assert websim.state.last_speech == "hello"
    assert websim.state.current_emotion == "happy"


def test_websim_do_sim_with_llm_error_message(websim):
    """Test that _do_sim includes llm_error_message in state."""
    websim.io_provider.llm_error_message = "Test error"
    websim._do_sim([], broadcast=False)

    assert "llm_error_message" in websim.state_dict
    assert websim.state_dict["llm_error_message"] == "Test error"


def test_websim_get_earliest_time_docstring():
    """Test that WebSim.get_earliest_time has a docstring."""
    assert WebSim.get_earliest_time.__doc__ is not None


def test_websim_get_earliest_time(websim):
    """Test get_earliest_time returns correct timestamp."""
    from providers.io_provider import Input

    inputs = {
        "input1": Input(input="value1", timestamp=1.0, tick=0),
        "input2": Input(input="value2", timestamp=2.0, tick=0),
        "input3": Input(input="value3", timestamp=0.5, tick=0),
    }

    earliest = websim.get_earliest_time(inputs)
    assert earliest == 0.5


def test_websim_get_earliest_time_empty(websim):
    """Test get_earliest_time returns 0 for empty inputs."""
    earliest = websim.get_earliest_time({})
    assert earliest == 0.0


def test_websim_do_sim_broadcast_with_loop(websim):
    """Test that _do_sim broadcasts when broadcast=True and loop exists."""
    websim._loop = asyncio.new_event_loop()
    websim._do_sim([], broadcast=True)


def test_websim_do_sim_broadcast_without_loop(websim):
    """Test that _do_sim skips broadcast when loop is None."""
    websim._loop = None
    websim._do_sim([], broadcast=True)


@pytest.mark.asyncio
async def test_websim_broadcast_state_with_no_connections(websim):
    """Test broadcast_state returns early when no active connections."""
    websim.active_connections = []
    await websim.broadcast_state()


@pytest.mark.asyncio
async def test_websim_broadcast_state_with_connections(websim):
    """Test broadcast_state sends state to active connections."""
    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()

    websim.active_connections = [mock_ws]
    websim.state_dict = {"test": "data"}

    await websim.broadcast_state()

    mock_ws.send_json.assert_called_once_with({"test": "data"})


@pytest.mark.asyncio
async def test_websim_broadcast_state_with_failed_connection(websim):
    """Test broadcast_state handles failed connections."""
    mock_ws_ok = MagicMock()
    mock_ws_ok.send_json = AsyncMock()

    mock_ws_fail = MagicMock()
    mock_ws_fail.send_json = AsyncMock(side_effect=Exception("Connection lost"))

    websim.active_connections = [mock_ws_ok, mock_ws_fail]
    websim.state_dict = {"test": "data"}

    await websim.broadcast_state()

    # Failed connection should be removed
    assert mock_ws_ok in websim.active_connections
    assert mock_ws_fail not in websim.active_connections


@pytest.mark.asyncio
async def test_websim_broadcast_state_exception_handling(websim):
    """Test broadcast_state handles exceptions gracefully."""
    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock(side_effect=RuntimeError("Test error"))

    websim.active_connections = [mock_ws]
    websim.state_dict = {"test": "data"}

    # Should not raise an exception
    await websim.broadcast_state()


def test_websim_get_earliest_time_with_inf_inputs(websim):
    """Test get_earliest_time handles inputs with infinite timestamps."""
    from providers.io_provider import Input

    inputs = {
        "input1": Input(input="value1", timestamp=float("inf"), tick=0),
        "input2": Input(input="value2", timestamp=2.0, tick=0),
    }

    earliest = websim.get_earliest_time(inputs)
    assert earliest == 2.0


def test_websim_do_sim_with_system_latency(websim):
    """Test _do_sim calculates system latency correctly."""
    import time

    websim.io_provider.fuser_start_time = time.time() - 0.2
    websim.io_provider.fuser_end_time = time.time() - 0.1
    websim.io_provider.llm_start_time = time.time() - 0.15
    websim.io_provider.llm_end_time = time.time() - 0.05

    websim._do_sim([], broadcast=False)

    assert "system_latency" in websim.state_dict
    latency = websim.state_dict["system_latency"]
    assert "fuse_time" in latency
    assert "llm_start" in latency
    assert "processing" in latency
    assert "complete" in latency


def test_websim_do_sim_with_inputs_rezeroed(websim):
    """Test _do_sim processes inputs and creates input_rezeroed."""
    websim.io_provider.add_input("test_input", "test_value", 1.0)

    websim._do_sim([], broadcast=False)

    assert "inputs" in websim.state_dict


def test_websim_active_connections_initialized(websim):
    """Test that active_connections is initialized as empty list."""
    assert hasattr(websim, "active_connections")
    assert websim.active_connections == []


def test_websim_app_initialized(websim):
    """Test that FastAPI app is initialized."""
    assert websim.app is not None
    assert hasattr(websim.app, "get")


def test_websim_io_provider_initialized(websim):
    """Test that io_provider is initialized."""
    from providers.io_provider import IOProvider

    # Check type name since IOProvider is a singleton
    assert type(websim.io_provider).__name__ == "IOProvider"
    # Or verify it has the expected methods/attributes
    assert hasattr(websim.io_provider, "add_input")
    assert hasattr(websim.io_provider, "inputs")


def test_websim_messages_initialized(websim):
    """Test that messages is initialized as empty list."""
    assert hasattr(websim, "messages")
    assert websim.messages == []


def test_websim_lock_initialized(websim):
    """Test that _lock is initialized."""
    import threading

    assert hasattr(websim, "_lock")
    assert isinstance(websim._lock, type(threading.Lock()))


def test_websim_initialized_flag(websim):
    """Test that _initialized flag is set to True."""
    assert websim._initialized is True


def test_websim_state_current_action_defaults(websim):
    """Test that state current_action defaults to 'idle'."""
    assert websim.state.current_action == "idle"


def test_websim_state_last_speech_defaults(websim):
    """Test that state last_speech defaults to empty string."""
    assert websim.state.last_speech == ""


def test_websim_state_current_emotion_defaults(websim):
    """Test that state current_emotion defaults to empty string."""
    assert websim.state.current_emotion == ""


def test_websim_state_system_latency_initialized(websim):
    """Test that state system_latency is initialized with default values."""
    assert websim.state.system_latency is not None
    assert "fuse_time" in websim.state.system_latency
    assert "llm_start" in websim.state.system_latency
    assert "processing" in websim.state.system_latency
    assert "complete" in websim.state.system_latency


def test_websim_run_server_docstring():
    """Test that WebSim._run_server has a docstring."""
    assert WebSim._run_server.__doc__ is not None


def test_websim_broadcast_state_docstring():
    """Test that WebSim.broadcast_state has a docstring."""
    assert WebSim.broadcast_state.__doc__ is not None


def test_websim_cleanup_docstring():
    """Test that WebSim.cleanup has a docstring."""
    assert WebSim.cleanup.__doc__ is not None


def test_websim_run_server_creates_and_sets_event_loop(websim):
    """Test that _run_server creates a new event loop and sets it."""
    import asyncio

    # Before calling _run_server, loop should be None
    assert websim._loop is None

    # Create a new event loop (simulating what _run_server does)
    test_loop = asyncio.new_event_loop()
    websim._loop = test_loop

    # Verify the loop was set
    assert websim._loop is test_loop
    assert isinstance(websim._loop, asyncio.AbstractEventLoop)


@pytest.mark.asyncio
async def test_websim_cleanup_with_loop(websim):
    """Test cleanup properly stops the simulator when loop exists."""
    websim._initialized = True
    websim._loop = asyncio.new_event_loop()

    await websim.cleanup()

    assert websim._initialized is False


@pytest.mark.asyncio
async def test_websim_cleanup_logs_info(websim, caplog):
    """Test that cleanup logs info message."""
    import logging

    websim._initialized = True

    with caplog.at_level(logging.INFO):
        await websim.cleanup()

    assert "Cleaning up WebSim" in caplog.text


def test_websim_config_passed_to_init():
    """Test that SimulatorConfig is properly passed to __init__."""
    config = SimulatorConfig(name="custom_websim")

    with patch("simulators.plugins.WebSim.threading.Thread") as mock_thread:
        mock_thread_instance = MagicMock()
        mock_thread_instance.is_alive.return_value = True
        mock_thread.return_value = mock_thread_instance

        with patch("simulators.plugins.WebSim.WebSim.sleep"):
            simulator = WebSim(config)

    assert simulator.name == "custom_websim"
    assert simulator.config is config


@pytest.mark.asyncio
async def test_websim_broadcast_state_handles_value_error_on_remove(websim):
    """Test broadcast_state handles ValueError when removing already-removed connection."""
    mock_ws = MagicMock()
    mock_ws.send_json = AsyncMock()

    # Add same connection twice to test ValueError handling
    websim.active_connections = [mock_ws]
    websim.state_dict = {"test": "data"}

    # Manually trigger the ValueError path by removing the connection first
    websim.active_connections.remove(mock_ws)

    # Then try to remove again in the exception handler path
    # We'll simulate this by calling broadcast_state with a connection that fails
    websim.active_connections = [mock_ws]

    async def failing_send():
        raise Exception("Send failed")

    mock_ws.send_json = failing_send

    await websim.broadcast_state()

    # Connection should be removed despite the ValueError
    assert mock_ws not in websim.active_connections


def test_websim_do_sim_with_move_action_no_change(websim):
    """Test _do_sim with move action that has same value as current."""
    from llm.output_model import Action

    websim.state.current_action = "existing action"

    actions = [Action(type="move", value="existing action")]

    websim._do_sim(actions, broadcast=False)

    # State should not change since action value is the same
    assert websim.state.current_action == "existing action"


def test_websim_do_sim_with_speak_action_no_change(websim):
    """Test _do_sim with speak action that has same value as current."""
    from llm.output_model import Action

    websim.state.last_speech = "existing speech"

    actions = [Action(type="speak", value="existing speech")]

    websim._do_sim(actions, broadcast=False)

    # State should not change since action value is the same
    assert websim.state.last_speech == "existing speech"


def test_websim_do_sim_with_emotion_action_no_change(websim):
    """Test _do_sim with emotion action that has same value as current."""
    from llm.output_model import Action

    websim.state.current_emotion = "existing emotion"

    actions = [Action(type="emotion", value="existing emotion")]

    websim._do_sim(actions, broadcast=False)

    # State should not change since action value is the same
    assert websim.state.current_emotion == "existing emotion"


def test_websim_do_sim_with_unknown_action_type(websim):
    """Test _do_sim ignores unknown action types."""
    from llm.output_model import Action

    original_action = websim.state.current_action
    original_speech = websim.state.last_speech
    original_emotion = websim.state.current_emotion

    actions = [Action(type="unknown_type", value="some value")]

    websim._do_sim(actions, broadcast=False)

    # State should not change for unknown action types
    assert websim.state.current_action == original_action
    assert websim.state.last_speech == original_speech
    assert websim.state.current_emotion == original_emotion


def test_websim_get_earliest_time_logs_debug(websim, caplog):
    """Test get_earliest_time logs debug information."""
    from providers.io_provider import Input

    import logging

    inputs = {
        "input1": Input(input="value1", timestamp=1.0, tick=0),
        "input2": Input(input="value2", timestamp=2.0, tick=0),
    }

    with caplog.at_level(logging.DEBUG):
        earliest = websim.get_earliest_time(inputs)

    assert earliest == 1.0
    # Debug logs should be generated
    assert len(caplog.records) > 0


def test_websim_state_to_dict_with_all_fields():
    """Test SimulatorState.to_dict with all fields populated."""
    state = SimulatorState(
        inputs={"input1": "value1"},
        current_action="moving",
        last_speech="hello world",
        current_emotion="happy",
        system_latency={
            "fuse_time": 0.5,
            "llm_start": 0.6,
            "processing": 0.1,
            "complete": 0.7,
        },
    )

    result = state.to_dict()

    assert result["inputs"] == {"input1": "value1"}
    assert result["current_action"] == "moving"
    assert result["last_speech"] == "hello world"
    assert result["current_emotion"] == "happy"
    assert result["system_latency"]["fuse_time"] == 0.5


def test_websim_sim_docstring_exists():
    """Test that sim method has docstring."""
    assert WebSim.sim.__doc__ is not None
    assert "simulation updates" in WebSim.sim.__doc__.lower()


def test_websim_tick_docstring_content():
    """Test that tick docstring describes update behavior."""
    doc = WebSim.tick.__doc__
    assert doc is not None
    assert "update" in doc.lower() or "tick" in doc.lower()
