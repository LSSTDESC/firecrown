"""Unit tests for firecrown.fctools.tracer module.

Tests the method tracing facility.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=import-outside-toplevel
# Test helper classes don't need docstrings
# types module imported inside tests to create mock frame objects

import subprocess
import sys
from pathlib import Path

import pytest

from firecrown.fctools.tracer import TracerState, settrace, untrace


@pytest.fixture(autouse=True)
def cleanup_tracing():
    """Ensure sys.settrace is always cleaned up after each test.

    This fixture runs automatically for all tests and ensures that tracing
    is disabled after each test completes, preventing interference between
    parallel test runs.
    """
    # Setup: save current trace function
    old_trace = sys.gettrace()

    yield

    # Teardown: always disable tracing and restore original state
    sys.settrace(old_trace)


# Tests for TracerState.__init__()


def test_tracer_state_init_creates_file(tmp_path):
    """Test that TracerState creates a trace file."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    try:
        assert trace_file.exists()
        assert tracer.level == 0
        assert tracer.entry == 0
    finally:
        tracer.close()


def test_tracer_state_init_writes_header(tmp_path):
    """Test that TracerState writes header line."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "entry\tevent\tlevel\tfunction\tvalue\textra" in content
    finally:
        tracer.close()


def test_tracer_state_default_filename(tmp_path, monkeypatch):
    """Test that TracerState uses default filename."""
    monkeypatch.chdir(tmp_path)
    tracer = TracerState()

    try:
        trace_file = Path("trace.tsv")
        assert trace_file.exists()
    finally:
        tracer.close()


# Tests for TracerState.trace_call() - call events


def test_trace_call_event(tmp_path):
    """Test tracing a function call event."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    # Create a simple frame-like object for testing
    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(
            co_argcount=2,
            co_varnames=("x", "y"),
            co_qualname="test_func",
        ),
        f_locals={"x": 1, "y": 2},
    )

    tracer.trace_call(frame_obj, "call", None)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "call" in content
        assert "test_func" in content
        assert tracer.entry == 1
        assert tracer.level == 1
    finally:
        tracer.close()


def test_trace_call_with_self_argument(tmp_path):
    """Test tracing a method call with self argument."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    class TestClass:
        def method(self):
            pass

    obj = TestClass()

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(
            co_argcount=1,
            co_varnames=("self",),
            co_qualname="TestClass.method",
        ),
        f_locals={"self": obj},
    )

    tracer.trace_call(frame_obj, "call", None)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "TestClass" in content
        assert "method" in content
    finally:
        tracer.close()


def test_trace_call_no_arguments(tmp_path):
    """Test tracing a function with no arguments."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(
            co_argcount=0, co_varnames=(), co_qualname="no_args_func"
        ),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "call", None)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "call" in content
        assert "no_args_func" in content
    finally:
        tracer.close()


# Tests for TracerState.trace_call() - return events


def test_trace_return_event(tmp_path):
    """Test tracing a function return event."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1  # Simulate being inside a function

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "return", 42)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "return" in content
        assert "42" in content
        assert "int" in content  # type of return value
        assert tracer.level == 0  # level decremented
    finally:
        tracer.close()


def test_trace_return_with_none(tmp_path):
    """Test tracing a return with None value."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "return", None)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "return" in content
        assert "None" in content
        assert "NoneType" in content
    finally:
        tracer.close()


def test_trace_return_with_unprintable_object(tmp_path):
    """Test tracing a return with object that can't be converted to str."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1

    class UnprintableObject:
        def __str__(self):
            raise AttributeError("Cannot convert to string")

    obj = UnprintableObject()

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "return", obj)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "return" in content
        assert "UnprintableObject" in content
    finally:
        tracer.close()


def test_trace_return_with_recursion_error(tmp_path):
    """Test tracing a return with object that causes RecursionError on str()."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1

    class RecursiveObject:
        def __str__(self):
            raise RecursionError("Maximum recursion depth exceeded")

    obj = RecursiveObject()

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "return", obj)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "return" in content
        assert "RecursiveObject" in content
    finally:
        tracer.close()


def test_trace_return_with_type_error(tmp_path):
    """Test tracing a return with object that causes TypeError on str()."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1

    class TypeErrorObject:
        def __str__(self):
            raise TypeError("Cannot convert to string")

    obj = TypeErrorObject()

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "return", obj)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "return" in content
        assert "TypeErrorObject" in content
    finally:
        tracer.close()


# Tests for TracerState.trace_call() - exception events


def test_trace_exception_event(tmp_path):
    """Test tracing an exception event."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))
    tracer.level = 1

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_qualname="test_func"),
        f_locals={},
    )

    tracer.trace_call(frame_obj, "exception", None)

    try:
        tracer.tracefile.flush()
        content = trace_file.read_text()
        assert "exception" in content
        assert "test_func" in content
        assert tracer.entry == 1
    finally:
        tracer.close()


def test_trace_call_returns_itself(tmp_path):
    """Test that trace_call returns itself for continued tracing."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    import types

    frame_obj = types.SimpleNamespace(
        f_code=types.SimpleNamespace(
            co_argcount=0, co_varnames=(), co_qualname="test_func"
        ),
        f_locals={},
    )

    result = tracer.trace_call(frame_obj, "call", None)

    try:
        assert result == tracer.trace_call
    finally:
        tracer.close()


# Tests for TracerState.close()


def test_tracer_state_close(tmp_path):
    """Test that close() closes the trace file."""
    trace_file = tmp_path / "test.tsv"
    tracer = TracerState(str(trace_file))

    tracer.close()

    # File should be closed
    assert tracer.tracefile.closed


# Tests for settrace()


def test_settrace_creates_tracer(tmp_path):
    """Test that settrace creates and returns a TracerState."""
    trace_file = tmp_path / "test.tsv"
    tracer = settrace(str(trace_file))

    try:
        assert isinstance(tracer, TracerState)
        assert trace_file.exists()
    finally:
        sys.settrace(None)
        tracer.close()


def test_settrace_enables_tracing(tmp_path):
    """Test that settrace enables sys.settrace."""
    trace_file = tmp_path / "test.tsv"
    tracer = settrace(str(trace_file))

    try:
        # sys.settrace should be set
        assert sys.gettrace() is not None
    finally:
        sys.settrace(None)
        tracer.close()


def test_settrace_default_filename(tmp_path, monkeypatch):
    """Test that settrace uses default filename."""
    monkeypatch.chdir(tmp_path)
    tracer = settrace()

    try:
        trace_file = Path("trace.tsv")
        assert trace_file.exists()
    finally:
        sys.settrace(None)
        tracer.close()


# Tests for untrace()


def test_untrace_disables_tracing(tmp_path):
    """Test that untrace disables sys.settrace."""
    trace_file = tmp_path / "test.tsv"
    tracer = settrace(str(trace_file))

    untrace(tracer)

    # sys.settrace should be disabled
    assert sys.gettrace() is None


def test_untrace_closes_file(tmp_path):
    """Test that untrace closes the trace file."""
    trace_file = tmp_path / "test.tsv"
    tracer = settrace(str(trace_file))

    untrace(tracer)

    assert tracer.tracefile.closed


# Tests for main() - script execution


def test_main_traces_script(tmp_path):
    """Test main traces a Python script."""
    # Create a simple test script
    script_file = tmp_path / "test_script.py"
    script_file.write_text(
        """
def simple_function():
    return 42

result = simple_function()
"""
    )

    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Tracing script:" in result.stdout
    assert "Trace complete" in result.stdout
    assert trace_file.exists()

    # Verify trace file has content
    content = trace_file.read_text()
    assert "entry\tevent\tlevel\tfunction\tvalue\textra" in content


def test_main_with_nonexistent_script(tmp_path):
    """Test the main CLI with a nonexistent script."""
    trace_file = tmp_path / "trace_output.txt"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            "nonexistent.py",
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # The tracer handles file not found gracefully and prints an error message
    assert "Error" in result.stdout
    assert "not found" in result.stdout


def test_main_module_mode_with_cli_runner(tmp_path):
    """Test the main CLI in module mode with subprocess."""
    trace_file = tmp_path / "trace_output.txt"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            "--module",
            "json.tool",
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert trace_file.exists()
    assert trace_file.stat().st_size > 0


def test_main_module_tracing(tmp_path):
    """Test main with module tracing (all module-related scenarios).

    Uses subprocess to avoid Click CliRunner I/O conflicts with sys.settrace()
    when running tests in parallel with pytest-xdist.

    Tests:
    - Tracing a valid module with --module flag
    - Tracing with short -m flag
    - Tracing an invalid/nonexistent module
    """
    # Test 1: Trace a valid module with --module flag
    trace_file1 = tmp_path / "output1.tsv"
    result1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "firecrown.fctools.tracer",
            "json.tool",
            "--module",
            "--output",
            str(trace_file1),
        ],
        cwd=tmp_path,
        input=b'{"test": "data"}\n',
        capture_output=True,
        check=False,
    )
    assert result1.returncode == 0
    assert b"Tracing module:" in result1.stdout
    assert b"Trace complete" in result1.stdout
    assert trace_file1.exists()

    # Test 2: Trace with short -m flag
    trace_file2 = tmp_path / "output2.tsv"
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "firecrown.fctools.tracer",
            "-m",
            "json.tool",
            "-o",
            str(trace_file2),
        ],
        cwd=tmp_path,
        input=b'{"test": "data"}\n',
        capture_output=True,
        check=False,
    )
    assert result2.returncode == 0
    assert b"Tracing module:" in result2.stdout
    assert trace_file2.exists()

    # Test 3: Trace an invalid module
    trace_file3 = tmp_path / "output3.tsv"
    result3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "firecrown.fctools.tracer",
            "nonexistent_module_xyz",
            "--module",
            "--output",
            str(trace_file3),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result3.returncode == 0  # tracer catches the error
    assert b"Tracing module:" in result3.stdout
    assert b"Error during traced execution" in result3.stdout
    assert trace_file3.exists()


def test_main_default_output_filename(tmp_path, monkeypatch):
    """Test main uses default output filename."""
    monkeypatch.chdir(tmp_path)

    script_file = tmp_path / "test_script.py"
    script_file.write_text("x = 1\n")

    # Get absolute path to tracer script
    tracer_script = Path(__file__).parent.parent.parent / "firecrown" / "fctools"
    tracer_script = tracer_script / "tracer.py"

    result = subprocess.run(
        [sys.executable, str(tracer_script), str(script_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert Path("trace.tsv").exists()


def test_main_short_output_option(tmp_path):
    """Test main with -o short option."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text("x = 1\n")
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "-o",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert trace_file.exists()


def test_main_handles_script_with_system_exit(tmp_path):
    """Test main handles scripts that call sys.exit()."""
    script_file = tmp_path / "exit_script.py"
    script_file.write_text("import sys\nsys.exit(0)\n")
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Should complete successfully even though script exits
    assert "Trace complete" in result.stdout
    assert trace_file.exists()


def test_main_with_subprocess(tmp_path):
    """Test that the script can be executed directly via subprocess.

    This test verifies that the __main__ block works correctly.
    """
    script_file = tmp_path / "test_script.py"
    script_file.write_text("x = 42\n")
    trace_file = tmp_path / "output.tsv"

    tracer_script = "firecrown/fctools/tracer.py"
    result = subprocess.run(
        [sys.executable, tracer_script, str(script_file), "-o", str(trace_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Tracing script:" in result.stdout
    assert "Trace complete" in result.stdout
    assert trace_file.exists()


def test_main_with_oserror_in_script(tmp_path):
    """Test main handles OSError during script execution."""
    script_file = tmp_path / "error_script.py"
    script_file.write_text("raise OSError('Test error')\n")
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "Error during traced execution" in result.stdout
    assert "Trace complete" in result.stdout
    assert trace_file.exists()


def test_main_with_runtime_error_in_script(tmp_path):
    """Test main handles RuntimeError during script execution."""
    script_file = tmp_path / "error_script.py"
    script_file.write_text("raise RuntimeError('Test error')\n")
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "Error during traced execution" in result.stdout
    assert "Trace complete" in result.stdout
    assert trace_file.exists()


def test_main_with_value_error_in_script(tmp_path):
    """Test main handles ValueError during script execution."""
    script_file = tmp_path / "error_script.py"
    script_file.write_text("raise ValueError('Test error')\n")
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "Error during traced execution" in result.stdout
    assert "Trace complete" in result.stdout
    assert trace_file.exists()


# Integration tests


def test_full_tracing_workflow(tmp_path):
    """Test complete tracing workflow with actual function execution."""
    trace_file = tmp_path / "trace.tsv"

    # Start tracing
    tracer = settrace(str(trace_file))

    try:
        # Execute some code to trace
        def add(a, b):
            return a + b

        def multiply(x, y):
            return x * y

        result1 = add(2, 3)
        result2 = multiply(4, 5)

        assert result1 == 5
        assert result2 == 20

    finally:
        # Stop tracing
        untrace(tracer)

    # Verify trace file exists and has content
    assert trace_file.exists()
    content = trace_file.read_text()

    # Should have header
    assert "entry\tevent\tlevel\tfunction\tvalue\textra" in content

    # Should have traced function calls and returns
    lines = content.strip().split("\n")
    assert len(lines) > 1  # Header + at least some trace entries


def test_tracing_with_nested_calls(tmp_path):
    """Test tracing with nested function calls."""
    trace_file = tmp_path / "trace.tsv"
    tracer = settrace(str(trace_file))

    try:

        def outer():
            def inner():
                return "inner"

            return inner()

        result = outer()
        assert result == "inner"

    finally:
        untrace(tracer)

    content = trace_file.read_text()

    # Should trace both outer and inner calls
    assert "outer" in content or "inner" in content


def test_cli_produces_valid_tsv(tmp_path):
    """Test that CLI produces valid TSV file."""
    script_file = tmp_path / "test_script.py"
    script_file.write_text(
        """
def func():
    return 1

func()
"""
    )
    trace_file = tmp_path / "output.tsv"

    result = subprocess.run(
        [
            sys.executable,
            "firecrown/fctools/tracer.py",
            str(script_file),
            "--output",
            str(trace_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    # Verify TSV format
    content = trace_file.read_text()
    lines = content.strip().split("\n")

    # Should have at least header line
    assert len(lines) >= 1

    # Header should have 6 columns with tabs
    header = lines[0]
    assert "\t" in header
    columns = header.split("\t")
    assert len(columns) == 6
    assert columns == ["entry", "event", "level", "function", "value", "extra"]


# Direct main() function tests for coverage


def test_main_function_with_script(tmp_path, capsys):
    """Test main function directly with a script file."""
    from firecrown.fctools.tracer import main

    # Create a simple test script
    script_file = tmp_path / "test_script.py"
    script_file.write_text("x = 1 + 1\n")
    trace_file = tmp_path / "trace.tsv"

    # Call main directly
    main(target=str(script_file), output=str(trace_file), module=False)

    # Verify trace file was created
    assert trace_file.exists()
    captured = capsys.readouterr()
    assert "Tracing script:" in captured.out
    assert "Trace complete" in captured.out


def test_main_function_with_nonexistent_script(tmp_path, capsys):
    """Test main function with nonexistent script."""
    from firecrown.fctools.tracer import main

    trace_file = tmp_path / "trace.tsv"

    # Call main with nonexistent script - should exit with error
    try:
        main(target="nonexistent_script.py", output=str(trace_file), module=False)
    except SystemExit as e:
        assert e.code == 1

    captured = capsys.readouterr()
    assert "Error: Script file" in captured.out
    assert "not found" in captured.out


def test_main_function_with_module(tmp_path, capsys):
    """Test main function in module mode."""
    from firecrown.fctools.tracer import main

    trace_file = tmp_path / "trace.tsv"

    # Use json.tool as a test module (part of standard library)
    main(target="json.tool", output=str(trace_file), module=True)

    # Verify trace file was created
    assert trace_file.exists()
    captured = capsys.readouterr()
    assert "Tracing module:" in captured.out
    assert "Trace complete" in captured.out


def test_main_function_with_invalid_module(tmp_path, capsys):
    """Test main function with invalid module name."""
    from firecrown.fctools.tracer import main

    trace_file = tmp_path / "trace.tsv"

    # Try to import nonexistent module
    main(target="nonexistent_module_xyz", output=str(trace_file), module=True)

    # Should handle ImportError gracefully
    captured = capsys.readouterr()
    assert "Error during traced execution" in captured.out
    assert "Trace complete" in captured.out
    assert trace_file.exists()


def test_main_function_with_script_system_exit(tmp_path, capsys):
    """Test main function with script that calls sys.exit()."""
    from firecrown.fctools.tracer import main

    script_file = tmp_path / "exit_script.py"
    script_file.write_text("import sys\nsys.exit(0)\n")
    trace_file = tmp_path / "trace.tsv"

    # Should handle SystemExit gracefully
    main(target=str(script_file), output=str(trace_file), module=False)

    captured = capsys.readouterr()
    assert "Trace complete" in captured.out
    assert trace_file.exists()


def test_main_function_with_script_import_error(tmp_path, capsys):
    """Test main function with script that has import error."""
    from firecrown.fctools.tracer import main

    script_file = tmp_path / "bad_import.py"
    script_file.write_text("import nonexistent_module\n")
    trace_file = tmp_path / "trace.tsv"

    # Should handle ImportError gracefully
    main(target=str(script_file), output=str(trace_file), module=False)

    captured = capsys.readouterr()
    assert "Error during traced execution" in captured.out
    assert "Trace complete" in captured.out
    assert trace_file.exists()


def test_main_function_with_script_value_error(tmp_path, capsys):
    """Test main function with script that raises ValueError."""
    from firecrown.fctools.tracer import main

    script_file = tmp_path / "value_error.py"
    script_file.write_text("raise ValueError('test error')\n")
    trace_file = tmp_path / "trace.tsv"

    # Should handle ValueError gracefully
    main(target=str(script_file), output=str(trace_file), module=False)

    captured = capsys.readouterr()
    assert "Error during traced execution" in captured.out
    assert "Trace complete" in captured.out
    assert trace_file.exists()
