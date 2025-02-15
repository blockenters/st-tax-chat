import os
import streamlit as st
from llama_index.llms.together import TogetherLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from huggingface_hub import snapshot_download

# API 키 검증
if 'api_keys' not in st.secrets:
    st.error('API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.')
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="세무 상담 챗봇",
    page_icon="💼",
    layout="centered"
)

# 세션 상태 초기화
if 'index' not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # TogetherLLM 설정
    llm = TogetherLLM(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=st.secrets["api_keys"]["together"],  # secrets에서 키 가져오기
        temperature=0,
        max_tokens=2048,
        system_prompt="너는 한국인 세무사야. 질문에 대한 답변은 최대한 자세하고 상세하게 답변해줘."
    )

    # 임베딩 모델 설정
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200

    # 허깅페이스에서 인덱스 다운로드
    repo_id = "blockenters/tax-index-dztax"
    local_dir = "tax-index-dztax"

    with st.spinner('데이터를 로딩중입니다...'):
        # 허깅페이스에 있는 데이터를 로컬에 다운로드
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type='dataset',
            token=st.secrets["api_keys"]["huggingface"]  # secrets에서 키 가져오기
        )

        # 다운로드한 폴더를 메모리에 로드
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        st.session_state.index = load_index_from_storage(storage_context)

# 채팅 기록 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 헤더
st.header("💼 세무 상담 챗봇")
st.markdown("---")

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("세무 관련 질문을 입력하세요."):
    # 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 쿼리 엔진 설정 및 응답 생성
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=True,
        temperature=0
    )

    # 어시스턴트 응답 표시
    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})

# 사이드바에 도움말 추가
with st.sidebar:
    st.markdown("""
    ### 💡 사용 방법
    1. 세무 관련 질문을 입력해주세요
    2. 예시 질문:
        - 소득세 신고 기한이 언제인가요?
        - 부가가치세 신고는 어떻게 하나요?
        - 연말정산 필요 서류는 무엇인가요?
        - 세무사 신고대행 수수료가 궁금합니다
    """)