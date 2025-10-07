#!/usr/bin/env python
"""Method tracing facility.

This module provides a facility to capture and record tracing data, using the
`sys.settrace` method. It causes a tab-separated value file to be written, with
one record (line) for each captured event.

The columns in the file are:

    entry:
        a sequential entry number, for each event
    event:
        the event type (call, return, exception)
    level:
        the call nesting level
    function:
        the function name
    value:
        for a 'call' entry, the names of the arguments;
        for a 'return' entry, the return value
    extra:
        for a 'call' entry, the type self, if that is the first argument;
        for a 'return' entry, the type of the return value

N.B.: This tracer should be used only for debugging and development purposes.
      It interferes with the pytest test coverage measurement process.
"""

import runpy
import sys
from pathlib import Path

import click


class TracerState:
    """Encapsulates tracing state to avoid global variables."""

    def __init__(self, filename: str = "trace.tsv") -> None:
        """Initialize tracer state with output file."""
        # File must remain open for the duration of tracing session (many callbacks)
        # and is properly closed via close() method called from untrace()
        # pylint: disable=consider-using-with
        self.tracefile = Path(filename).open(mode="w", encoding="utf8")  # noqa: SIM115
        self.level = 0  # the call nesting level
        self.entry = 0  # sequential entry number for each record
        print("entry\tevent\tlevel\tfunction\tvalue\textra", file=self.tracefile)

    def trace_call(self, fr, ev, arg):
        """Callback used by settrace.

        :param fr: the frame object
        :param ev: the event type
        :param arg: the argument
        """
        code = fr.f_code
        extra = ""
        match ev:
            case "call":
                self.entry += 1
                self.level += 1
                nargs = code.co_argcount
                # slice the tuple to get only argument names
                argnames = code.co_varnames[:nargs]
                if nargs > 0 and code.co_varnames[0] == "self":
                    val = fr.f_locals["self"]
                    extra = f"{type(val).__name__}"
                print(
                    f"{self.entry}\tcall\t{self.level}\t{code.co_qualname}\t"
                    f"{argnames}\t{extra}",
                    file=self.tracefile,
                )
            case "return":
                self.entry += 1
                extra = f"{type(arg).__name__}"
                # Handle special cases where arg conversion might cause issues
                try:
                    arg_str = str(arg)
                except (AttributeError, RecursionError, TypeError):
                    arg_str = f"<{type(arg).__name__} object>"
                print(
                    f"{self.entry}\treturn\t{self.level}\t{code.co_qualname}\t"
                    f"{arg_str}\t{extra}",
                    file=self.tracefile,
                )
                self.level -= 1
            case "exception":
                self.entry += 1
                print(
                    f"{self.entry}\texception\t{self.level}\t{code.co_qualname}\t"
                    f"\t{extra}",
                    file=self.tracefile,
                )
        return self.trace_call

    def close(self) -> None:
        """Close the trace file."""
        self.tracefile.close()


def settrace(filename: str = "trace.tsv") -> TracerState:
    """Start the tracer, with log being written to a new file with the given name.

    :param filename: the name of the new file to be created
    :return: TracerState instance managing the trace
    """
    tracer = TracerState(filename)
    sys.settrace(tracer.trace_call)
    return tracer


def untrace(tracer: TracerState) -> None:
    """Turn off tracing, and close the specified trace file.

    :param tracer: TracerState instance, as returned by settrace.
    """
    sys.settrace(None)
    tracer.close()


@click.command()
@click.argument("target")
@click.option(
    "--output", "-o", default="trace.tsv", help="Output trace file (default: trace.tsv)"
)
@click.option(
    "--module",
    "-m",
    is_flag=True,
    help="Run target as a module (like python -m module)",
)
def main(target: str, output: str, module: bool):
    """Trace execution of a Python script or module.

    This tool enables method tracing for Python code, recording function
    calls, returns, and exceptions to a TSV file for analysis.

    TARGET  Python script file or module name to trace
    """
    # Start tracing
    tracer = settrace(output)

    try:
        if module:
            # Run as module (like python -m)
            click.echo(f"Tracing module: {target}")
            click.echo(f"Trace output: {output}")
            runpy.run_module(target, run_name="__main__", alter_sys=True)
        else:
            # Run as script file
            script_path = Path(target)
            if not script_path.exists():
                click.echo(f"Error: Script file '{target}' not found.", err=True)
                sys.exit(1)

            click.echo(f"Tracing script: {target}")
            click.echo(f"Trace output: {output}")
            runpy.run_path(str(script_path), run_name="__main__")

    except SystemExit:
        # Allow normal script exit
        pass
    except (OSError, ImportError, ValueError, RuntimeError) as e:
        click.echo(f"Error during traced execution: {e}", err=True)
    finally:
        # Stop tracing and close file
        untrace(tracer)
        click.echo(f"Trace complete. Output saved to: {output}")


if __name__ == "__main__":
    # Click decorators inject arguments automatically from sys.argv
    main()  # pylint: disable=no-value-for-parameter
