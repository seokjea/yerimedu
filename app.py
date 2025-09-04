# app.py
import streamlit as st
import time, json, random, os, hashlib
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# 0) 부팅 & 공용 유틸
# =========================
load_dotenv()

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_event(kind, payload=None):
    st.session_state["events"].append({
        "t": now(),
        "kind": kind,
        "payload": payload or {}
    })

def init_state():
    defaults = {
        "book_title": "",
        "book_text": "",
        "outline": {"intro": "", "body": "", "concl": ""},
        "draft": "",
        "events": [],
        "n_sugs": 3,
        "enable_hints": True,
        "enable_questions": True,
        "role": "학생",
        "ai_suggestions_cache": {},
        "current_questions": [],
        "use_chat_mode": True,            # Enter 전송 채팅 모드
        "selected_book_prev": "",
        "book_index": "",                 # LLM 요약 원문(텍스트)
        "book_index_json": {},            # 파싱된 요약/장면/키워드
        "focus_kw": "",                   # 선택된 키워드
        "saved_versions": [],
        "help_open": False,
        "model_name": "gpt-5",             # 기본 모델명
        "spelling_feedback": [],   # 맞춤법/표현 피드백 보존
        "question_history": [],   # 지금까지 나온 질문들 저장해서 중복 방지
        "question_nonce": 0      # 매 요청마다 달라지는 다양화 토큰
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================
# 1) OpenAI 래퍼
# =========================
def call_openai_api(messages, max_tokens=500, model=None):
    """Chat Completions 호출 간단 래퍼."""
    try:
        response = client.chat.completions.create(
            model=model or st.session_state.get("model_name", "gpt-5"),
            messages=messages,
            max_completion_tokens=max_tokens,  # 일부 SDK는 max_completion_tokens가 아닌 max_tokens 사용
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI API 오류: {str(e)}")
        return None

# =========================
# 2) 책 인덱싱(요약/장면/키워드)
# =========================
def index_book_text():
    txt = st.session_state.get("book_text", "").strip()
    if not txt:
        st.warning("인덱싱할 책 내용이 없습니다.")
        return

    prompt = f"""
너는 초등학생 독서감상문 도우미다. 아래 책 내용을 읽고 JSON으로만 답하라.
형식:
{{
  "summary": ["문장1","문장2","문장3"],     // 3문장 요약
  "key_scenes": ["장면1","장면2","장면3"],  // 3개
  "keywords": ["키워드1","키워드2","키워드3","키워드4","키워드5"] // 5개
}}
원문(일부 또는 전체):
{txt[:4000]}
"""
    res = call_openai_api([{"role": "user", "content": prompt}], max_tokens=600)
    if not res:
        st.warning("책 인덱싱에 실패했습니다.")
        return

    st.session_state["book_index"] = res

    # JSON 파싱 시도
    try:
        data = json.loads(res)
        # 최소 필드 보정
        st.session_state["book_index_json"] = {
            "summary": data.get("summary", []) if isinstance(data.get("summary", []), list) else [],
            "key_scenes": data.get("key_scenes", []) if isinstance(data.get("key_scenes", []), list) else [],
            "keywords": data.get("keywords", []) if isinstance(data.get("keywords", []), list) else [],
        }
    except Exception:
        # 파싱 실패 시 텍스트만 유지
        st.session_state["book_index_json"] = {}
    log_event("book_indexed", {
        "summary_len": len(st.session_state.get("book_index_json", {}).get("summary", [])),
        "keywords_len": len(st.session_state.get("book_index_json", {}).get("keywords", []))
    })
    st.success("📚 책 인덱싱(요약/장면/키워드)을 완료했습니다.")

# =========================
# 3) 제안/질문/맞춤법 등
# =========================
def _stable_cache_key(parts: list) -> str:
    s = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def generate_ai_suggestions(context, block, n=3):
    """AI를 활용한 작문 제안 생성 (서론/본론/결론)"""
    key_fields = [
        context.get("book_title", ""),
        context.get("book_text", "")[:800],
        json.dumps(context.get("outline", {}), ensure_ascii=False, sort_keys=True),
        context.get("draft", "")[-500:],   # 최근 문맥을 더 반영
        block,
        n,
        st.session_state.get("focus_kw", ""),
        st.session_state.get("book_index", "")[:1200]
    ]
    cache_key = _stable_cache_key(key_fields)
    cache = st.session_state["ai_suggestions_cache"]
    if cache_key in cache:
        return cache[cache_key]

    block_prompts = {
        "intro": "독서감상문의 서론 부분으로, 책을 읽게 된 계기나 첫인상에 대한",
        "body":  "독서감상문의 본론 부분으로, 인상 깊은 장면과 느낀 점에 대한",
        "concl": "독서감상문의 결론 부분으로, 배운 점과 추천 이유에 대한",
    }
    focus_kw = st.session_state.get("focus_kw", "")
    book_index = st.session_state.get("book_index", "")

    prompt = f"""
당신은 초등학교 4-6학년 학생의 독서감상문 작성을 돕는 AI 교사입니다.
학생 정보:
- 책 제목: {context.get('book_title', '미정')}
- 책 내용(요약/장면/키워드): {book_index[:1200] or '(없음)'}
- 현재 개요: {json.dumps(context.get('outline', {}), ensure_ascii=False)}
- 현재 초안(마지막 500자): {context.get('draft', '')[-500:] or '(없음)'}
- 선택된 키워드: {focus_kw or '(없음)'}

요청: {block_prompts[block]} 문장 제안을 {n}개 만들어주세요.

조건:
1) 초등학생 수준의 쉽고 자연스러운 표현
2) 각 제안은 한 문장으로 완성
3) 구체적이고 실용적인 내용
4) 학생이 선택·수정하여 바로 사용할 수 있는 형태
5) 1인칭 시점("나는", "내가")
6) 번호나 특수문자 없이 문장만 제시

예시: 이 책을 읽게 된 이유는 표지가 예뻐서 호기심이 생겼기 때문이다.

제안 {n}개:
"""
    with st.spinner("🤖 AI가 제안을 만들고 있어요..."):
        response = call_openai_api([{"role": "user", "content": prompt}], max_tokens=400)
    if response:
        # 응답 라인 정리
        lines = []
        for line in response.strip().split("\n"):
            s = line.strip().lstrip("0123456789.- ").strip()
            if s and len(s) > 8:
                lines.append(s)
        lines = lines[:n]
        cache[cache_key] = lines
        log_event("ai_suggestions_generated", {"block": block, "count": len(lines)})
        return lines

    # 실패 시 기본 제안
    return get_fallback_suggestions(block, n)

def get_fallback_suggestions(block, n=3):
    fallback = {
        "intro": [
            "이 책을 읽게 된 이유는 친구가 재미있다고 추천했기 때문이다.",
            "처음 제목을 보았을 때 어떤 이야기일지 궁금했다.",
            "도서관에서 우연히 이 책을 발견하고 기대가 생겼다.",
        ],
        "body": [
            "가장 인상 깊었던 장면은 주인공이 어려움을 극복하는 부분이었다.",
            "등장인물들의 우정을 보며 진정한 친구의 의미를 생각하게 되었다.",
            "만약 내가 주인공이었다면 어떤 선택을 했을지 상상해 보았다.",
        ],
        "concl": [
            "이 책을 통해 포기하지 않는 태도의 중요함을 배웠다.",
            "친구들에게도 꼭 추천하고 싶은 책이라고 느꼈다.",
            "앞으로 이 책에서 배운 교훈을 생활에서 실천해 보고 싶다.",
        ],
    }
    return random.sample(fallback[block], min(n, len(fallback[block])))

def generate_guiding_questions(context):
    """
    상황에 맞는 유도 질문 3개 (중복 회피 + 매번 다르게)
    - 서로 다른 시작어 사용: 왜/어떻게/만약
    - 이전에 나왔던 질문들과 의미/표현 중복 금지
    - 선택된 키워드/책 요약을 살짝 반영
    """
    # 히스토리/토큰 준비
    hist = st.session_state.get("question_history", [])
    st.session_state["question_nonce"] = st.session_state.get("question_nonce", 0) + 1
    nonce = st.session_state["question_nonce"]

    focus_kw = st.session_state.get("focus_kw", "")
    idx_json = st.session_state.get("book_index_json", {})
    summary_txt = " / ".join(idx_json.get("summary", [])[:3])
    scenes_txt  = " / ".join(idx_json.get("key_scenes", [])[:3])

    # 최근 초안 꼬리
    draft_tail = (context.get("draft", "") or "")[-200:]

    system = (
        "너는 초등학생이 이해하기 쉬운 열린 질문을 만드는 한국어 교사야. "
        "반드시 서로 다른 질문 시작어를 사용하고(왜/어떻게/만약), 각 질문은 1줄, 물음표(?)로 끝나야 해. "
        "이미 했던 질문들과 표현/의미가 겹치지 않게 만들어."
    )

    user = f"""
[학생 정보]
- 책 제목: {context.get('book_title','미정')}
- 선택된 키워드(있으면 반영): {focus_kw or '(없음)'}
- 책 요약: {summary_txt or '(없음)'}
- 핵심 장면: {scenes_txt or '(없음)'}
- 현재 초안(마지막 200자): {draft_tail or '(없음)'}

[이미 했던 질문들(중복 금지)]
{chr(10).join('• '+q for q in hist[-20:]) if hist else '(없음)'}

[요청]
- 서로 다른 시작어로 3개: ①왜..., ②어떻게..., ③만약...
- 각 질문은 한 줄, 반드시 '?'로 끝내기
- 번호/불릿 없이 질문만 3줄
- 다양화토큰: {nonce}

질문 3개:
"""

    resp = call_openai_api(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=300
    )
    if not resp:
        # 안정적 폴백(질문 스타일 다양화)
        base = [
            "왜 이 장면이 특히 중요한지 스스로 설명할 수 있나요?",
            "어떻게 이 책의 메시지를 일상에서 실천할 수 있을까요?",
            "만약 당신이 주인공이었다면 어떤 결정을 내렸을까요?"
        ]
        st.session_state["question_history"] = (hist + base)[-50:]
        return base

    # 후처리: 라인 정리 + 중복 제거 + 포맷 강제
    raw_lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    cleaned, seen = [], set(hist[-100:])  # 최근 100개와의 중복 회피
    for ln in raw_lines:
        q = ln.lstrip("0123456789.-•* ").strip()
        # 반드시 물음표로 끝나게
        if not q.endswith("?"):
            q = q.rstrip(".!…") + "?"
        # 히스토리/이번 결과 중복 제거(문자열 기준)
        if q in seen:
            continue
        seen.add(q)
        cleaned.append(q)
        if len(cleaned) == 3:
            break

    # 만약 3개가 안 채워지면 폴백으로 채우기(시작어별 강제)
    starters = ["왜", "어떻게", "만약"]
    while len(cleaned) < 3:
        s = starters[len(cleaned) % 3]
        # 키워드를 섞어서 단순 생성
        kw_part = f" '{focus_kw}'" if focus_kw else ""
        filler = f"{s} 이 책을 통해{kw_part} 내가 배우거나 바꿀 수 있는 점은 무엇일까요?"
        if not filler.endswith("?"):
            filler += "?"
        if filler not in seen:
            cleaned.append(filler)
            seen.add(filler)

    # 히스토리 업데이트(최대 50개 유지)
    st.session_state["question_history"] = (hist + cleaned)[:50]

    return cleaned[:3]


def check_spelling_and_grammar(text):
    """맞춤법/표현 피드백 최대 3개.
    1차: 맞춤법/띄어쓰기/조사/어미 등 규범 위반만.
    2차: 1차 결과가 없으면 '표현 다듬기' 1~2개 제안."""
    if not text or not text.strip():
        return []

    # --- 1차: 규범 위반만 요구 (없으면 '없음' 명시) ---
    system = (
        "너는 한국어 교정 교사다. 국립국어원 한글 맞춤법/띄어쓰기/외래어 표기법 기준으로만 판단한다. "
        "출력은 각 줄 하나, 총 2~3줄. 각 줄은 '원래 표현 → 고친 표현 (이유)' 형식을 반드시 지켜라. "
        "진짜 규범 위반이 없으면 정확히 '없음' 한 줄만 출력해라."
    )
    user = f"""다음 글에서 규범 위반(맞춤법, 띄어쓰기, 조사·어미)만 2~3개 고쳐줘.
            형식: "원래 표현 → 고친 표현 (이유)"
    글:
        {text[:1000]}"""

    resp = call_openai_api(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        max_tokens=300
    ) or ""

    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    # '없음'이면 2차로 표현 팁 제공
    if any(ln == "없음" for ln in lines) or not lines:
        style_sys = (
            "너는 한국어 글쓰기 코치다. 규범 위반이 없을 때만 표현 개선 팁을 준다. "
            "각 줄 하나, 최대 2줄. 형식: '표현) 제안 문장 (이유)'"
        )
        style_user = f"""다음 글의 표현을 자연스럽게 다듬을 수 있는 개선 제안을 1~2개만 제시해줘.
형식: "표현) 제안 문장 (이유)"
글:
{text[:800]}"""
        style = call_openai_api(
            [{"role": "system", "content": style_sys},
             {"role": "user", "content": style_user}],
            max_tokens=160
        ) or ""
        tips = []
        for ln in style.splitlines():
            s = ln.strip().lstrip("0123456789.-•* ").strip()
            if s and s.startswith("표현)"):
                tips.append(s)
                if len(tips) == 2:
                    break
        return tips  # 표현 팁(선택적)

    # 1차 결과 정제 (최대 3개)
    out = []
    for ln in lines:
        s = ln.lstrip("0123456789.-•* ").strip()
        if "→" in s and "(" in s and ")" in s:
            out.append(s)
            if len(out) == 3:
                break
    return out


def suggest_next_sentences(context, n=3):
    """막힐 때 다음 문장 추천"""
    draft_tail = context.get("draft", "")[-500:]
    book_index = st.session_state.get("book_index", "")
    prompt = f"""
학생 초안의 마지막 부분을 자연스럽게 이어갈 **한 문장** 제안을 {n}개 생성해줘.
조건: 초등학생이 이해하기 쉬운 표현, 각 제안은 한 문장만.

지금까지 초안(마지막 500자):
{draft_tail}

책 지식(요약/장면/키워드):
{book_index[:600] or '(없음)'}
"""
    resp = call_openai_api([{"role": "user", "content": prompt}], max_tokens=300) or ""
    cands = [x.strip().lstrip("0123456789.- ").strip() for x in resp.split("\n") if x.strip()]
    return cands[:n]

# =========================
# 4) 추천 도서(기본)
# =========================
RECOMMENDED_BOOKS = {
    "소리없는 아이들 - 황선미": {
        "summary": "특별한 아이들의 소통과 이해에 관한 이야기",
        "key_scenes": [
            "주인공이 처음 특별한 친구를 만나는 장면",
            "서로 다른 소통 방식을 이해하게 되는 순간",
            "편견을 극복하고 진정한 우정을 나누는 결말",
        ],
    },
    "나와 조금 다를 뿐이야 - 이금이": {
        "summary": "다름을 인정하고 받아들이는 성장 이야기",
        "key_scenes": [
            "주인공이 자신과 다른 친구를 처음 만나는 장면",
            "차이점 때문에 생기는 갈등과 오해",
            "서로의 다름을 이해하고 받아들이는 화해",
        ],
    },
    "여름과 가을 사이 - 박슬기": {
        "summary": "계절의 변화처럼 성장하는 아이의 마음",
        "key_scenes": [
            "여름 방학 동안 겪은 특별한 경험",
            "새 학기를 앞두고 느끼는 복잡한 감정",
            "성장을 받아들이며 새로운 시작을 준비하는 모습",
        ],
    },
    "인어 소녀 - 차율이": {
        "summary": "꿈과 현실 사이에서 고민하는 소녀의 이야기",
        "key_scenes": [
            "주인공이 자신만의 특별한 꿈을 갖게 되는 순간",
            "꿈을 이루기 위해 노력하면서 겪는 어려움",
            "꿈을 향한 의지를 다지며 성장하는 결말",
        ],
    },
}

# =========================
# 5) UI 구성
# =========================
def render_sidebar():
    st.sidebar.header("⚙️ 설정")

    # 모델명 변경 가능 (기본 gpt-5)
    st.session_state["model_name"] = st.sidebar.text_input("모델명", st.session_state.get("model_name", "gpt-5"))
    api_status = "🟢 연결됨" if os.getenv("OPENAI_API_KEY") else "🔴 미연결"
    st.sidebar.caption(f"AI 상태: {api_status}")

    st.session_state["role"] = st.sidebar.selectbox("사용자 모드", ["학생", "교사"], index=0)
    st.session_state["n_sugs"] = st.sidebar.slider("AI 제안 개수", 1, 5, st.session_state.get("n_sugs", 3))
    st.session_state["enable_hints"] = st.sidebar.checkbox("맞춤법 검사", st.session_state.get("enable_hints", True))
    st.session_state["enable_questions"] = st.sidebar.checkbox("AI 유도 질문", st.session_state.get("enable_questions", True))

    st.sidebar.divider()
    st.sidebar.subheader("📚 책 선택")

    book_options = ["직접 입력"] + list(RECOMMENDED_BOOKS.keys())
    selected_book = st.sidebar.selectbox("책 선택", book_options)

    # 선택이 바뀐 경우만 반영
    if selected_book != st.session_state.get("selected_book_prev", ""):
        st.session_state["selected_book_prev"] = selected_book
        if selected_book != "직접 입력":
            st.session_state["book_title"] = selected_book
            info = RECOMMENDED_BOOKS[selected_book]
            st.session_state["book_text"] = (
                f"{info['summary']}\n\n주요 장면들:\n" + "\n".join([f"- {s}" for s in info["key_scenes"]])
            )
            st.sidebar.success(f"'{selected_book}' 정보가 로드되었습니다!")
            # 자동 인덱싱
            index_book_text()

    if selected_book == "직접 입력":
        st.session_state["book_title"] = st.sidebar.text_input("책 제목", st.session_state["book_title"])
        uploaded = st.sidebar.file_uploader("책 요약/중요 부분(.txt)", type=["txt"])
        if uploaded:
            try:
                st.session_state["book_text"] = uploaded.read().decode("utf-8")
            except Exception:
                st.session_state["book_text"] = uploaded.read().decode("utf-8", errors="ignore")
            log_event("book_uploaded", {"name": uploaded.name, "chars": len(st.session_state["book_text"])})
            # 업로드 후 자동 인덱싱
            index_book_text()

        if st.sidebar.button("📑 직접 입력한 내용 인덱싱/요약"):
            index_book_text()

    st.sidebar.divider()
    if st.sidebar.button("📊 활동 로그 준비"):
        data = {
            "book_title": st.session_state["book_title"],
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
            "events": st.session_state["events"],
        }
        st.sidebar.download_button(
            "events.json 저장",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"독서감상문_로그_{now().replace(':', '-').replace(' ', '_')}.json",
        )

def render_quality_panel():
    if st.session_state["enable_hints"] and st.session_state["draft"].strip():
        st.markdown("**✍️ AI 맞춤법 및 표현 도움**")
        if st.button("🔍 맞춤법 검사하기"):
            with st.spinner("AI가 글을 검토하고 있어요..."):
                feedback = check_spelling_and_grammar(st.session_state["draft"])
                if feedback:
                    for tip in feedback:
                        if tip.strip():
                            st.info(tip.strip())
                else:
                    st.success("훌륭해요! 특별한 문제점이 없어 보입니다.")

    if st.session_state["enable_questions"]:
        st.markdown("**🤔 AI가 제안하는 생각 질문**")
        if st.button("💡 새로운 질문 받기"):
            context = {"book_title": st.session_state["book_title"], "draft": st.session_state["draft"]}
            with st.spinner("AI가 질문을 만들고 있어요..."):
                questions = generate_guiding_questions(context)
                st.session_state["current_questions"] = questions

        if st.session_state.get("current_questions"):
            for i, q in enumerate(st.session_state["current_questions"], 1):
                st.write(f"{i}. {q}")

def render_outline():
    st.subheader("📖 Step 1. 책 내용 확인 및 정리")

    if st.session_state["book_title"]:
        st.success(f"선택된 책: **{st.session_state['book_title']}**")

    st.text_area(
        "책 요약/중요 부분",
        key="book_text",
        height=120,
        placeholder="책에서 중요한 장면이나 문장을 입력해주세요. 위에서 추천 도서를 선택하면 자동으로 채워집니다.",
    )

    st.subheader("📝 Step 2. 글의 뼈대 만들기")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**서론 (책 소개/읽게 된 이유)**")
        st.text_area(
            "서론 입력",                              # ← 라벨 채우기
            key="outline_intro",
            height=120,
            placeholder="• 책을 읽게 된 계기\n• 첫인상이나 기대\n• 간단한 책 소개",
            label_visibility="collapsed"             # ← 화면에선 숨기기
        )
    with cols[1]:
        st.markdown("**본론 (인상 깊은 장면/느낀 점)**")
        st.text_area(
            "본론 입력",
            key="outline_body",
            height=120,
            placeholder="• 가장 기억에 남는 장면\n• 그 이유와 느낀 점\n• 나의 경험과 연결",
            label_visibility="collapsed"
        )
    with cols[2]:
        st.markdown("**결론 (배운 점/추천 이유)**")
        st.text_area(
            "결론 입력",
            key="outline_concl",
            height=120,
            placeholder="• 책에서 배운 것\n• 추천하고 싶은 이유\n• 앞으로의 다짐",
            label_visibility="collapsed"
        )

    st.session_state["outline"]["intro"] = st.session_state.get("outline_intro", "")
    st.session_state["outline"]["body"]  = st.session_state.get("outline_body", "")
    st.session_state["outline"]["concl"] = st.session_state.get("outline_concl", "")

def render_keyword_pills():
    """인덱싱된 키워드로 클릭형 태그 UI"""
    kws = st.session_state.get("book_index_json", {}).get("keywords", [])
    if not kws:
        return
    st.markdown("**🔖 핵심 키워드:**")
    cols = st.columns(min(5, len(kws)))
    for i, kw in enumerate(kws):
        with cols[i % len(cols)]:
            if st.button(f"#{kw}", key=f"kw_{i}"):
                st.session_state["focus_kw"] = kw
                st.success(f"키워드 '{kw}'에 맞춰 제안을 생성합니다.")

def render_suggestion_block(label, key_block, icon):
    st.markdown(f"### {icon} {label} 작성 도움")

    if st.button(f"🤖 AI {label} 제안 받기", key=f"generate_{key_block}"):
        ctx = {
            "book_title": st.session_state["book_title"],
            "book_text": st.session_state["book_text"],
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
        }
        suggestions = generate_ai_suggestions(ctx, key_block, st.session_state["n_sugs"])
        st.session_state[f"suggestions_{key_block}"] = suggestions

    current_suggestions = st.session_state.get(f"suggestions_{key_block}", [])
    if current_suggestions:
        st.markdown("**🎯 AI가 만든 작문 제안들**")
        for i, sug in enumerate(current_suggestions):
            with st.container(border=True):
                st.write(f"**제안 {i+1}:** {sug}")
                edited = st.text_area("수정해서 추가", value=sug, height=70, key=f"edit_{key_block}_{i}")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("✅ 수정 본문에 추가", key=f"accept_edit_{key_block}_{i}"):
                        text_to_add = (edited or sug).strip()
                        cur = st.session_state["draft"].strip()
                        st.session_state["draft"] = (cur + "\n\n" + text_to_add) if cur else text_to_add
                        st.session_state["use_chat_mode"] = False  # 바로 편집 모드로 전환
                        log_event("ai_suggestion_accepted", {"block": key_block, "text": text_to_add, "edited": edited != sug})
                        st.success("본문에 추가했어요!")
                        time.sleep(0.3)
                        st.rerun()
                with c2:
                    if st.button("❌ 패스", key=f"reject_{key_block}_{i}"):
                        log_event("ai_suggestion_rejected", {"block": key_block, "text": sug})
                        st.info("다른 제안을 확인해보세요!")

def detect_stage(draft: str, outline: dict) -> str:
    """간단 휴리스틱 단계 감지"""
    if not draft.strip():
        return "intro"
    if len(draft) < 150:
        return "intro"
    if any(x in draft for x in ["가장 인상 깊었던", "기억에 남는", "느꼈", "교훈"]):
        return "body"
    return "concl"

def detect_stage_llm(draft, outline):
    """LLM 기반 단계 감지 (intro/body/concl 중 하나만)"""
    prompt = f"""
학생 초안을 보고 현재 어느 단계인지 intro/body/concl 중 하나로만 답해줘. 다른 말은 하지 마.
초안(마지막 800자):
{draft[-800:]}
개요:
{json.dumps(outline, ensure_ascii=False)}
"""
    resp = call_openai_api([{"role": "user", "content": prompt}], max_tokens=5) or "body"
    r = resp.lower()
    if "intro" in r: return "intro"
    if "concl" in r: return "concl"
    return "body"

def render_dynamic_suggestions():
    st.markdown("### 🧭 진행도 기반 자동 제안")
    if st.button("🤖 지금 단계에 맞는 제안 받기"):
        stage = detect_stage_llm(st.session_state["draft"], st.session_state["outline"])
        ctx = {
            "book_title": st.session_state["book_title"],
            "book_text": st.session_state["book_text"],
            "outline": st.session_state["outline"],
            "draft": st.session_state["draft"],
        }
        st.session_state[f"suggestions_{stage}"] = generate_ai_suggestions(ctx, stage, st.session_state["n_sugs"])
        st.success(f"현재 단계 추정: {stage} (자동 제안 생성)")

def save_snapshot():
    if not st.session_state.get("draft", "").strip():
        st.warning("저장할 내용이 없습니다.")
        return
    st.session_state["saved_versions"].append({"t": now(), "text": st.session_state["draft"]})
    st.success("💾 임시 저장 완료!")

def render_snapshot_bar():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.button("💾 임시 저장", on_click=save_snapshot)
    with c2:
        versions = st.session_state.get("saved_versions", [])
        if versions:
            idx = st.selectbox("저장본 불러오기", list(range(len(versions))), format_func=lambda i: versions[i]["t"])
            if st.button("↩️ 이 버전으로 되돌리기"):
                st.session_state["draft"] = versions[idx]["text"]
                st.success("해당 시점 버전으로 복구했습니다.")
                st.rerun()

def render_help_hub():
    if st.button("🆘 도움이 필요해요"):
        st.session_state["help_open"] = not st.session_state["help_open"]

    if st.session_state["help_open"]:
        with st.container(border=True):
            st.markdown("**도움말 허브**")
            c1, c2, c3 = st.columns(3)

            # ✅ 맞춤법 검사: 결과를 세션에 저장
            with c1:
                if st.button("🔍 맞춤법 검사", key="help_spelling"):
                    if not st.session_state["draft"].strip():
                        st.warning("검사할 글이 아직 없어요. 초안을 조금만 써주세요!")
                    else:
                        with st.spinner("AI가 글을 검토하고 있어요..."):
                            feedback = check_spelling_and_grammar(st.session_state["draft"]) or []
                        st.session_state["spelling_feedback"] = feedback
                        st.success(f"맞춤법/표현 제안 {len(feedback)}개를 가져왔어요.")

            with c2:
                if st.button("💡 생각 유도 질문"):
                    qs = generate_guiding_questions({"book_title": st.session_state["book_title"], "draft": st.session_state["draft"]})
                    st.session_state["current_questions"] = qs
            with c3:
                if st.button("➡️ 다음 문장 추천"):
                    ctx = {"book_title": st.session_state["book_title"], "draft": st.session_state["draft"], "book_text": st.session_state["book_text"]}
                    st.session_state["next_sugs"] = suggest_next_sentences(ctx, 3)

        # === 결과 표시 (리런 후에도 유지) ===
        if st.session_state.get("spelling_feedback"):
            st.markdown("#### ✍️ 맞춤법/표현 제안")
            for tip in st.session_state["spelling_feedback"]:
                st.info(tip)

        # 기존 결과 표시
        for i, q in enumerate(st.session_state.get("current_questions", []), 1):
            st.write(f"{i}. {q}")
        for i, s in enumerate(st.session_state.get("next_sugs", []), 1):
            c1, c2 = st.columns([6,1])
            with c1:
                st.write(f"• {s}")
            with c2:
                if st.button("추가", key=f"add_next_{i}"):
                    st.session_state["draft"] = (st.session_state["draft"] + ("\n" if st.session_state["draft"].strip() else "") + s).strip()
                    st.rerun()

def render_editor():
    st.subheader("✏️ Step 3. 초안 작성 및 다듬기")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.toggle(
        "Enter로 전송(채팅 입력 모드)",
        key="use_chat_mode",
        help="켜면 Enter로 전송되고, 줄바꿈은 Shift+Enter입니다. 끄면 자유 편집 모드입니다."
        )

        if st.session_state["use_chat_mode"]:
            st.text_area("현재 초안 (읽기 전용)", value=st.session_state["draft"], key="draft_view", height=300, disabled=True)
            new_line = st.chat_input("여기에 입력하고 Enter를 눌러 추가 (Shift+Enter 줄바꿈)")
            if new_line is not None and new_line.strip() != "":
                if st.session_state["draft"].strip():
                    st.session_state["draft"] += "\n" + new_line
                else:
                    st.session_state["draft"] = new_line
                log_event("draft_appended", {"source": "chat_input", "chars": len(new_line)})
                st.rerun()
        else:
            st.text_area(
                "내가 쓰고 있는 독서감상문",
                key="draft",
                height=300,
                placeholder="위에서 AI 제안을 선택하거나 직접 작성해보세요.\n\n💡 팁: AI 제안은 시작점일 뿐이에요. 여러분의 생각과 경험을 더해서 자신만의 독서감상문을 만들어보세요!",
            )

    with col2:
        render_quality_panel()

    st.subheader("💾 Step 4. 저장/완료")
    render_snapshot_bar()

    if st.session_state["draft"].strip():
        word_count = len(st.session_state["draft"])
        st.info(f"현재 글자 수: {word_count}자")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 텍스트 파일로 저장",
                data=st.session_state["draft"],
                file_name=f"{st.session_state['book_title']}_독서감상문.txt" if st.session_state["book_title"] else "독서감상문.txt",
            )
        with c2:
            if st.button("🎉 작성 완료!"):
                log_event("writing_completed", {"word_count": word_count, "book": st.session_state["book_title"]})
                st.balloons()
                st.success("독서감상문 작성을 완료했습니다! 수고하셨어요!")
    else:
        st.warning("아직 작성된 내용이 없습니다. 위에서 AI 제안을 선택하거나 직접 작성해보세요.")

# =========================
# 6) 메인
# =========================
def main():
    st.set_page_config(
        page_title="AI 독서감상문 작문 도우미",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    render_sidebar()

    st.title("📚 AI와 함께 쓰는 독서감상문")
    st.caption("🤖 AI 제안 → 🤔 선택/수정 → ✨ 완성! AI 선생님과 함께 특별한 독서감상문을 만들어보세요.")

    # 진행 상황
    steps_completed = [
        bool(st.session_state["book_text"].strip()),
        any(st.session_state["outline"].values()),
        bool(st.session_state["draft"].strip()),
    ]
    progress = sum(steps_completed) / 3
    st.progress(progress)
    st.caption(f"진행 상황: {int(progress * 100)}% 완료")

    render_outline()
    st.divider()

    # 키워드 태그 / 도움 허브 / 단계 감지 제안
    render_keyword_pills()
    render_help_hub()
    render_dynamic_suggestions()

    st.markdown("## 🎯 AI와 함께하는 단계별 작문")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_suggestion_block("서론", "intro", "🌟")
    with c2:
        render_suggestion_block("본론", "body", "💭")
    with c3:
        render_suggestion_block("결론", "concl", "🎭")

    st.divider()
    render_editor()

if __name__ == "__main__":
    main()
