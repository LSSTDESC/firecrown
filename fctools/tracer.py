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

import sys
from typing import TextIO

# some global context to be used in the tracing. We are relying on
# 'trace_call' to act as a closure that captures these names.
tracefile = None  # the file used for logging
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


def untrace(tracefile: TextIO) -> None:
    """Turn off tracing, and close the specified trace file.

    :param tracefile: an open file, as returned by setttrace.
    """
    sys.settrace(None)
    tracefile.close()


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
