import streamlit as st
from contextlib import contextmanager

def _inject_base_css(theme):
    css = f"""
    <style>
      .app-shell {{
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: {theme['space']['lg']}px;
        background: {theme['color']['bg']};
        color: {theme['color']['text']};
        font-family: {theme['font']['family']};
      }}
      .panel {{
        background: {theme['color']['panel']};
        border: 1px solid {theme['color']['border']};
        border-radius: {theme['radius']}px;
        box-shadow: {theme['shadow']};
        padding: {theme['space']['lg']}px;
      }}
      .topbar {{
        display:flex; justify-content:space-between; align-items:center;
        background: {theme['color']['surface']};
        border:1px solid {theme['color']['border']};
        border-radius: {theme['radius']}px;
        padding: {theme['space']['md']}px {theme['space']['lg']}px;
        margin-bottom: {theme['space']['lg']}px;
      }}
      .status-dot {{
        width:10px; height:10px; border-radius:50%;
        background: #22c55e; display:inline-block; margin-right:8px;
      }}
      .footer {{
        margin-top:{theme['space']['lg']}px; 
        opacity:0.8; font-size:{theme['font']['size']['sm']}px;
      }}
      .sidenav-item {{
        display:flex; align-items:center; gap:10px;
        padding:10px 12px; border-radius:{theme['radius']}px; cursor:pointer;
        border:1px solid transparent;
      }}
      .sidenav-item.active {{
        background: rgba(110,231,255,0.08);
        border-color: {theme['color']['accent']};
      }}
      .sidenav-item:hover {{ background: rgba(255,255,255,0.04); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@contextmanager
def render_shell(title: str, theme: dict):
    _inject_base_css(theme)

    # Top bar
    top = st.container()
    with top:
        st.markdown(
            f"""
            <div class="topbar">
              <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:{theme['font']['size']['xxl']}px; font-weight:700;">{title}</span>
                <span style="opacity:.7;">| Global Forecast OS</span>
              </div>
              <div>
                <span class="status-dot"></span>
                <span style="opacity:.85;">Engine: Idle</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Two-column shell
    left_col, right_col = st.columns([0.28, 0.72], gap="large")
    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

    class _Shell:
        def __init__(self, left, right):
            self.left = left
            self.right = right
        def __enter__(self):
            self.left_container = self.left.container()
            self.right_container = self.right.container()
            return self
        def __exit__(self, exc_type, exc, tb):
            # close panel wrappers
            with self.left:
                st.markdown('</div>', unsafe_allow_html=True)
            with self.right:
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="footer">© PunchLogic Command — Phase 1 UI Shell</div>', unsafe_allow_html=True)

    shell = _Shell(left_col, right_col)
    yield shell
