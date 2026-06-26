"""Phase 1.7 — data-quality dashboard.

``queries`` holds pure, testable functions that read the DB and return
DataFrames; ``app`` is a thin Streamlit layer that renders them. Keeping
the SQL out of the Streamlit script lets us unit-test the reporting logic
without spinning up a browser.
"""
