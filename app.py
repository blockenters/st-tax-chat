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

@st.cache_resource
def initialize_llm():
    """LLM 모델 초기화"""
    return TogetherLLM(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=st.secrets["api_keys"]["together"],
        temperature=0,
        max_tokens=2048,
        system_prompt="너는 한국인 세무사야. 질문에 대한 답변은 최대한 자세하고 상세하게 답변해줘."
    )

@st.cache_resource
def initialize_embedding_model():
    """임베딩 모델 초기화"""
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        token=st.secrets["api_keys"]["huggingface"]
    )

@st.cache_data
def download_index_data():
    """허깅페이스에서 인덱스 데이터 다운로드"""
    repo_id = "blockenters/tax-index-dztax"
    local_dir = "tax-index-dztax"
    
    # 로컬에 파일이 없을 때만 다운로드
    if not os.path.exists(local_dir):
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type='dataset',
            token=st.secrets["api_keys"]["huggingface"]
        )
    return local_dir

@st.cache_resource
def load_index(_local_dir):
    """인덱스 로드"""
    storage_context = StorageContext.from_defaults(persist_dir=_local_dir)
    return load_index_from_storage(storage_context)

# 페이지 설정
st.set_page_config(
    page_title="세무 상담 챗봇",
    page_icon="💼",
    layout="centered"
)

# 세션 상태 초기화
if 'index' not in st.session_state:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 모델과 데이터 초기화
    llm = initialize_llm()
    embed_model = initialize_embedding_model()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200

    with st.spinner('데이터를 로딩중입니다...'):
        local_dir = download_index_data()
        index = load_index(local_dir)
        st.session_state.index = index
    
    # 쿼리 엔진 설정 및 응답 생성
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=True,
        temperature=0
    )
else:
    index = st.session_state.index
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
        streaming=True,
        temperature=0
    )   


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
    with st.spinner('데이터를 로딩중입니다...'):
        # 사용자 메시지 표시
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

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