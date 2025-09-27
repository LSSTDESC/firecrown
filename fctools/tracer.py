#!/usr/bin/env python
"""Method tracing facility.

This module provides a facility to capture and record tracing data, using the
`sys.settrace` method. It causes a tab-separated value file to be written, with
one record (line) for each captured event.

The columns in the file are:

    entry: a sequential entry number, for each event
    event: the event type (call, return, exception)
    level: the call nesting level
    function: the function name
    value: for a 'call' entry, the names of the arguments.
           forl a 'return' entry, the return value
    extra: for a 'call' entry, the type self, if that is the first argument
           for a 'return' entry, the type of the return value

N.B.: This tracer should be used only for debugging and development purposes.
      It interferes with the pytest test coverage measurement process.
"""

import click
import runpy
import sys
from pathlib import Path
from typing import TextIO

# some global context to be used in the tracing. We are relying on
# 'trace_call' to act as a closure that captures these names.
tracefile: TextIO | None = None  # the file used for logging
level = 0  # the call nesting level
entry = 0  # sequential entry number for each record


def settrace(filename: str = "trace.tsv") -> TextIO:
    """Start the tracer, with log being written to a new file with the given name.

    :param filename: the name of the new file to be created
    """
    global tracefile
    tracefile = open(filename, mode="w", encoding="utf8")
    print("entry\tevent\tlevel\tfunction\tvalue\textra", file=tracefile)
    sys.settrace(trace_call)
    return tracefile


def untrace(trace_file: TextIO) -> None:
    """Turn off tracing, and close the specified trace file.

    :param trace_file: an open file, as returned by setttrace.
    """
    sys.settrace(None)
    trace_file.close()


def trace_call(fr, ev, arg):
    """Callback used by settrace.

    :param fr: the frame object
    :param ev: the event type
    :param arg: the argument
    """
    code = fr.f_code
    extra = ""
    global entry
    global level
    match ev:
        case "call":
            entry += 1
            level += 1
            nargs = code.co_argcount
            # slice the tuple to get only argument names
            argnames = code.co_varnames[:nargs]
            if nargs > 0 and code.co_varnames[0] == "self":
                val = fr.f_locals["self"]
                extra = f"{type(val).__name__}"
            print(
                f"{entry}\tcall\t{level}\t{code.co_qualname}\t{argnames}\t{extra}",
                file=tracefile,
            )
        case "return":
            entry += 1
            extra = f"{type(arg).__name__}"
            print(
                f"{entry}\treturn\t{level}\t{code.co_qualname}\t{arg}\t{extra}",
                file=tracefile,
            )
            level -= 1
        case "exception":
            entry += 1
            print(
                f"{entry}\texception\t{level}\t{code.co_qualname}\t\t{extra}",
                file=tracefile,
            )
    return trace_call


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
    trace_file = settrace(output)

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
    except Exception as e:
        click.echo(f"Error during traced execution: {e}", err=True)
    finally:
        # Stop tracing and close file
        untrace(trace_file)
        click.echo(f"Trace complete. Output saved to: {output}")


if __name__ == "__main__":
    main()
