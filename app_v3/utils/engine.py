import time, streamlit as st
from contextlib import contextmanager
from utils.log import audit_event

ENGINE_KEY = "engine_state"

def _init():
    if ENGINE_KEY not in st.session_state:
        st.session_state[ENGINE_KEY] = {"state":"Idle","last_action":"","last_ms":0}

def get_engine_state():
    _init(); return st.session_state[ENGINE_KEY]

@contextmanager
def action(title: str, audit_name: str, **audit_meta):
    _init()
    st.session_state[ENGINE_KEY]["state"] = "Running"
    st.session_state[ENGINE_KEY]["last_action"] = title
    t0 = time.perf_counter()
    try:
        yield
        ms = int((time.perf_counter()-t0)*1000)
        st.session_state[ENGINE_KEY]["state"] = "Idle"
        st.session_state[ENGINE_KEY]["last_ms"] = ms
        audit_event(audit_name, ok=True, ms=ms, **audit_meta)
    except Exception as e:
        st.session_state[ENGINE_KEY]["state"] = "Error"
        audit_event(audit_name, ok=False, error=str(e), **audit_meta)
        raise
