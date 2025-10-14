from . import match_wrapped


def test_match_wrapped():
    text = "Reading coverage data from \n/tmp/pytest-of-runner/pytest-0/popen-gw0/test_main_direct_with_timing0/coverage.\njson...\nReading timing data from \n/tmp/pytest-of-runner/pytest-0/popen-gw0/test_main_direct_with_timing0/timing.tx\nt...\nLoaded timing data for 1 tests\nExtracting function-level coverage data...\nWriting 1 records to \n/tmp/pytest-of-runner/pytest-0/popen-gw0/test_main_direct_with_timing0/output.ts\nv...\nSuccessfully converted coverage data to TSV format!\nOutput file: \n/tmp/pytest-of-runner/pytest-0/popen-gw0/test_main_direct_with_timing0/output.ts\nv\nRecords written: 1\nRecords with timing data: 1\n"
    assert match_wrapped(text, "Reading coverage data from")
    assert match_wrapped(text, "Reading timing data from")
