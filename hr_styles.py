# =========================
# 페이지/스타일 (Clean & Minimal UI)
# =========================
import streamlit as st
import streamlit.components.v1 as _components
import plotly.io as pio

pio.templates.default = "plotly_white"

# 폰트 및 컬러 팔레트 정의
PRIMARY_COLOR = "#2563EB"
BG_COLOR = "#F8F9FA"       # Standard Light Gray
CARD_BG = "#FFFFFF"        # Solid White
TEXT_COLOR = "#333333"

# 차트 컬러 팔레트 (Corporate Custom: Purple & Cyan)
COLORS = {
    "primary": "#48C0D8",      # Corporate Cyan (Bar Charts)
    "secondary": "#5548C7",    # Corporate Purple (Line/Trend)
    "success": "#48C0D8",      # Cyan
    "danger": "#5548C7",       # Purple (Emphasis/Warning)
    "warning": "#F59E0B",      # Amber
    "info": "#48C0D8",         # Cyan
    "light": "#F8FAFC",        # Light
    "dark": "#334155",         # Dark
    "sequence": ["#48C0D8", "#5548C7", "#7DD3FC", "#A5B4FC", "#C4B5FD"]  # Cyan & Purple Mix
}

# 위험 등급 기준 (전 페이지 공통)
RISK_HIGH = 0.70   # 고위험
RISK_MID = 0.30    # 중위험


def risk_color(prob: float) -> str:
    """예측 확률에 따른 등급 색상"""
    if prob >= RISK_HIGH:
        return COLORS["warning"]      # 고위험 (Amber)
    elif prob >= RISK_MID:
        return "#7DD3FC"              # 중위험 (Light Cyan)
    return COLORS["primary"]          # 저위험 (Cyan)


def set_font(fig):
    layout_updates = {
        'font': dict(family="Pretendard, -apple-system, system-ui, sans-serif", size=14, color=TEXT_COLOR),
        'paper_bgcolor': "rgba(0,0,0,0)",
        'plot_bgcolor': "rgba(0,0,0,0)",
        'margin': dict(t=40, b=20, l=20, r=20)
    }

    # title이 있는 경우에만 title 관련 폰트 설정 추가
    if fig.layout.title and fig.layout.title.text:
        layout_updates['title_font_size'] = 18
        layout_updates['title_font_family'] = "Pretendard, sans-serif"
        layout_updates['title_font_color'] = "#111827"

    fig.update_layout(**layout_updates)
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    return fig


def inject_global_styles():
    """Clean Minimal CSS Injection"""
    st.markdown(f"""
        <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

        /* 전체 배경 및 폰트 */
        .stApp {{
            background-color: {BG_COLOR};
            font-family: 'Pretendard', sans-serif;
            color: {TEXT_COLOR};
        }}

        /* 헤더 내부 도구만 숨김 (헤더 자체와 사이드바 펼치기 버튼은 유지) */
        header [data-testid="stToolbar"],
        header [data-testid="stDecoration"],
        header [data-testid="stStatusWidget"],
        header [data-testid="stDeployButton"],
        header .stDeployButton,
        [data-testid="stMainMenu"] {{
            visibility: hidden !important;
        }}
        header,
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* 헤더는 보이게 (사이드바 펼치기 버튼이 헤더 안에 있음) */
        header,
        [data-testid="stHeader"] {{
            visibility: visible !important;
            display: block !important;
            height: auto !important;
            min-height: 3.5rem !important;
            z-index: 999998 !important;
        }}

        /* 사이드바 펼치기/접기 버튼 - 모든 Streamlit 버전 대응 */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"],
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarHeader"] button,
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"],
        button[aria-label*="sidebar"],
        button[aria-label*="Sidebar"],
        button[kind="header"],
        button[kind="headerNoPadding"] {{
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
            z-index: 999999 !important;
            pointer-events: auto !important;
            position: relative !important;
        }}

        /* 사이드바가 접혔을 때 펼치기 버튼을 좌측 상단에 고정 표시 */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {{
            position: fixed !important;
            top: 0.75rem !important;
            left: 0.75rem !important;
            width: auto !important;
            height: auto !important;
            transform: none !important;
        }}

        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stExpandSidebarButton"] svg,
        button[aria-label="Open sidebar"] svg,
        button[aria-label="Close sidebar"] svg {{
            fill: #2A9BB0 !important;
            color: #2A9BB0 !important;
            width: 24px !important;
            height: 24px !important;
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }}
        [data-testid="collapsedControl"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stExpandSidebarButton"],
        button[aria-label="Open sidebar"],
        button[aria-label="Close sidebar"] {{
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
            padding: 6px !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            min-height: 36px !important;
            cursor: pointer !important;
        }}
        [data-testid="collapsedControl"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="stExpandSidebarButton"]:hover,
        button[aria-label="Open sidebar"]:hover,
        button[aria-label="Close sidebar"]:hover {{
            background: #F0FAFC !important;
            border-color: #48C0D8 !important;
        }}

        /* 메인 컨테이너 */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1200px;
        }}

        /* 카드 스타일 (Clean Flat) */
        div[data-testid="stMetric"], div.stDataFrame, .plotly-graph-div {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* 아주 가벼운 그림자 */
            border: 1px solid #E5E7EB;
        }}

        /* 호버 효과 제거 또는 아주 약하게 */
        div[data-testid="stMetric"]:hover, .plotly-graph-div:hover {{
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}

        /* 메트릭 텍스트 스타일 */
        div[data-testid="stMetricLabel"] {{
            font-size: 0.9rem;
            color: #6B7280;
            font-weight: 500;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #111827;
        }}

        /* 데이터프레임 헤더 스타일 */
        div[data-testid="stDataFrame"] table th {{
            background-color: #F3F4F6 !important;
            color: #374151 !important;
            font-weight: 600 !important;
            border-bottom: 1px solid #E5E7EB !important;
            text-align: center !important;
        }}
        div[data-testid="stDataFrame"] table td {{
            text-align: center !important;
            color: #4B5563 !important;
            font-size: 0.95rem;
            border-bottom: 1px solid #F3F4F6 !important;
        }}

        /* 사이드바 스타일 */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #48C0D8 0%, #3BADC7 100%);
            border-right: none;
            padding-top: 0;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 0;
        }}
        /* 사이드바 내부 텍스트 색상 */
        section[data-testid="stSidebar"] * {{
            color: rgba(255, 255, 255, 0.85) !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #FFFFFF !important;
        }}
        /* 사이드바 메뉴 버튼 */
        section[data-testid="stSidebar"] .stButton > button {{
            width: 100%;
            text-align: left;
            padding: 14px 18px !important;
            border-radius: 8px;
            border: none;
            font-size: 15px;
            font-weight: 500;
            font-family: 'Pretendard', sans-serif;
            background: transparent !important;
            color: rgba(255, 255, 255, 0.85) !important;
            transition: all 0.15s ease;
            margin-bottom: 2px;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.15) !important;
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:focus {{
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: #FFFFFF !important;
            font-weight: 700 !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1) !important;
        }}
        /* 사이드바 파일 업로더 */
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.3);
        }}
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {{
            background: rgba(255, 255, 255, 0.9) !important;
            border: none !important;
            color: #2A9BB0 !important;
            font-weight: 600 !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover {{
            background: rgba(255, 255, 255, 1) !important;
        }}
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small,
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span,
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] p,
        section[data-testid="stSidebar"] div[data-testid="stFileUploader"] div {{
            color: #6B7280 !important;
        }}
        /* 사이드바 구분선 */
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255, 255, 255, 0.2);
            margin: 1.5rem 0;
        }}

        /* 버튼 스타일 */
        button {{
            border-radius: 6px !important;
            box-shadow: none !important;
        }}

        /* 타이틀 스타일 */
        h1, h2, h3 {{
            font-family: 'Pretendard', sans-serif;
            font-weight: 700;
            color: #111827;
        }}

        /* 구분선 스타일 */
        hr {{
            margin: 2rem 0;
            border-color: #E5E7EB;
        }}
        </style>
        """, unsafe_allow_html=True)


def inject_sidebar_toggle():
    """사이드바 펼치기 버튼 안전장치 (Streamlit 내장 버튼이 안 보이는 경우 대비)"""
    _components.html("""
<script>
(function() {
    var doc = window.parent.document;
    if (!doc) return;

    // 토글 버튼 스타일 주입 (메인 문서)
    if (!doc.getElementById('custom-sidebar-toggle-style')) {
        var style = doc.createElement('style');
        style.id = 'custom-sidebar-toggle-style';
        style.textContent = `
            #custom-sidebar-toggle {
                position: fixed;
                top: 0.75rem;
                left: 0.75rem;
                width: 36px;
                height: 36px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
                display: none;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                z-index: 999999;
                padding: 0;
                transition: all 0.15s ease;
            }
            #custom-sidebar-toggle:hover {
                background: #F0FAFC;
                border-color: #48C0D8;
            }
            #custom-sidebar-toggle svg {
                width: 20px;
                height: 20px;
                fill: #2A9BB0;
            }
        `;
        doc.head.appendChild(style);
    }

    function ensureToggleButton() {
        var existing = doc.getElementById('custom-sidebar-toggle');
        if (!existing) {
            var btn = doc.createElement('button');
            btn.id = 'custom-sidebar-toggle';
            btn.title = '사이드바 열기';
            btn.innerHTML = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M3 6h18v2H3V6zm0 5h18v2H3v-2zm0 5h18v2H3v-2z"/></svg>';
            btn.onclick = function() {
                var openBtn = doc.querySelector('[data-testid="stSidebarCollapsedControl"] button')
                            || doc.querySelector('[data-testid="collapsedControl"] button')
                            || doc.querySelector('[data-testid="stExpandSidebarButton"]')
                            || doc.querySelector('button[aria-label="Open sidebar"]');
                if (openBtn) {
                    openBtn.click();
                } else {
                    var sb = doc.querySelector('section[data-testid="stSidebar"]');
                    if (sb) {
                        sb.style.transform = 'none';
                        sb.style.visibility = 'visible';
                        sb.style.marginLeft = '0';
                        sb.setAttribute('aria-expanded', 'true');
                    }
                }
            };
            doc.body.appendChild(btn);
            existing = btn;
        }

        var sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            var aria = sidebar.getAttribute('aria-expanded');
            var rect = sidebar.getBoundingClientRect();
            var collapsed = (aria === 'false') || rect.width < 50 || (rect.left + rect.width) <= 5;
            existing.style.display = collapsed ? 'flex' : 'none';
        }
    }

    ensureToggleButton();
    setInterval(ensureToggleButton, 500);
})();
</script>
""", height=0)


def add_pdf_button():
    _components.html("""
    <script>
    (function() {
        var parentDoc = window.parent.document;

        // 이미 버튼이 있으면 중복 생성 방지
        if (parentDoc.getElementById('pdf-download-btn')) return;

        // Font Awesome 로드
        if (!parentDoc.querySelector('link[href*="font-awesome"]')) {
            var fa = parentDoc.createElement('link');
            fa.rel = 'stylesheet';
            fa.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
            parentDoc.head.appendChild(fa);
        }

        // 프린트용 CSS + 버튼/메뉴 스타일 주입
        var style = parentDoc.createElement('style');
        style.textContent = `
            #pdf-download-btn {
                position: fixed;
                top: 60px;
                right: 20px;
                z-index: 999999;
                background-color: #48C0D8;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: 'Pretendard', sans-serif;
            }
            #pdf-download-btn:hover {
                background-color: #3BADC7;
            }
            #pdf-orient-menu {
                position: fixed;
                top: 100px;
                right: 20px;
                z-index: 999999;
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                box-shadow: 0 6px 16px rgba(0,0,0,0.12);
                padding: 6px 0;
                display: none;
                min-width: 140px;
                font-family: 'Pretendard', sans-serif;
            }
            #pdf-orient-menu button {
                display: block;
                width: 100%;
                text-align: left;
                background: transparent;
                border: none;
                padding: 10px 18px;
                font-size: 14px;
                cursor: pointer;
                color: #334155;
                font-family: 'Pretendard', sans-serif;
            }
            #pdf-orient-menu button:hover {
                background: #F0FAFC;
                color: #2A9BB0;
            }

            @media print {
                /* 사이드바 숨김 */
                section[data-testid="stSidebar"],
                [data-testid="stSidebarNav"],
                [data-testid="collapsedControl"] {
                    display: none !important;
                }
                /* Streamlit 헤더/푸터 숨김 */
                header, footer,
                .stDeployButton,
                [data-testid="stToolbar"],
                [data-testid="stDecoration"],
                [data-testid="stStatusWidget"] {
                    display: none !important;
                }
                /* PDF 버튼/메뉴 자체 숨김 */
                #pdf-download-btn,
                #pdf-orient-menu {
                    display: none !important;
                }
                /* iframe(components) 영역 숨김 - 빈 공간 제거 */
                iframe[title="streamlit_components.v1.components.html"] {
                    display: none !important;
                }
                /* 본문 영역을 전체 너비로 */
                section[data-testid="stMain"],
                .main,
                [data-testid="stAppViewContainer"] {
                    margin-left: 0 !important;
                    padding-left: 0 !important;
                    width: 100% !important;
                    max-width: 100% !important;
                }
                .block-container {
                    max-width: 100% !important;
                    padding: 1rem !important;
                }
                /* 배경색 유지 */
                .stApp {
                    background-color: white !important;
                }
            }
        `;
        parentDoc.head.appendChild(style);

        // 메인 버튼
        var btn = parentDoc.createElement('button');
        btn.id = 'pdf-download-btn';
        btn.innerHTML = '<i class="fa fa-print"></i> PDF 저장 <span style="font-size:10px;margin-left:2px;">▾</span>';
        parentDoc.body.appendChild(btn);

        // 방향 선택 메뉴
        var menu = parentDoc.createElement('div');
        menu.id = 'pdf-orient-menu';
        menu.innerHTML = ''
            + '<button data-orient="landscape"><i class="fa fa-arrows-h" style="margin-right:8px;color:#48C0D8;"></i>가로 (Landscape)</button>'
            + '<button data-orient="portrait"><i class="fa fa-arrows-v" style="margin-right:8px;color:#48C0D8;"></i>세로 (Portrait)</button>';
        parentDoc.body.appendChild(menu);

        // 메뉴 토글
        btn.onclick = function(e) {
            e.stopPropagation();
            menu.style.display = (menu.style.display === 'block') ? 'none' : 'block';
        };

        // 메뉴 외부 클릭 시 닫기
        parentDoc.addEventListener('click', function(e) {
            if (e.target !== btn && !btn.contains(e.target) && !menu.contains(e.target)) {
                menu.style.display = 'none';
            }
        });

        // 방향 선택 → 동적으로 @page 주입 후 인쇄
        function doPrint(orientation) {
            var old = parentDoc.getElementById('pdf-orient-style');
            if (old) old.remove();
            var s = parentDoc.createElement('style');
            s.id = 'pdf-orient-style';
            s.media = 'print';
            s.textContent = '@page { size: A4 ' + orientation + '; margin: 10mm; }';
            parentDoc.head.appendChild(s);
            menu.style.display = 'none';
            setTimeout(function() { window.parent.print(); }, 60);
        }

        menu.querySelectorAll('button').forEach(function(b) {
            b.addEventListener('click', function(e) {
                e.stopPropagation();
                doPrint(b.getAttribute('data-orient'));
            });
        });
    })();
    </script>
    """, height=0)
