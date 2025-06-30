#!/usr/bin/env python3
# Instagram 감상평 생성기 (완전버전)
# 필요 패키지:
# pip install streamlit openai pillow opencv-python numpy

import sys
import subprocess
import os
import tempfile
import io
import base64
from typing import Tuple, Optional

# 자동 패키지 설치
required = [
    ("streamlit", "streamlit"),
    ("openai", "openai"),
    ("pillow", "PIL"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
]

for pkg_name, module_name in required:
    try:
        __import__(module_name)
    except ImportError:
        print(f"{pkg_name} 패키지가 설치되어 있지 않습니다. 설치를 진행합니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        except subprocess.CalledProcessError:
            print(f"{pkg_name} 설치에 실패했습니다.")

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from openai import OpenAI
import tempfile

# OpenAI 클라이언트 초기화
def get_openai_client(api_key):
    """OpenAI 클라이언트를 초기화합니다."""
    try:
        client = OpenAI(api_key=api_key)
        # API 키 유효성 간단 테스트
        client.models.list()
        return client
    except Exception as e:
        st.error(f"API 키가 유효하지 않습니다: {str(e)}")
        return None

def check_api_key():
    """API 키를 확인하고 입력받습니다."""
    # 1. Streamlit secrets에서 확인
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    # 2. 환경변수에서 확인  
    env_api_key = os.environ.get("OPENAI_API_KEY")
    if env_api_key:
        return env_api_key
    
    # 3. 세션 스테이트에서 확인
    if "api_key" in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    
    return None

def detect_faces_opencv(image: Image.Image) -> list:
    """OpenCV를 사용해 얼굴을 감지합니다."""
    try:
        # 이미지를 OpenCV 형식으로 변환
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        
        # 얼굴 검출 (Haar Cascade 사용)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
        
        return faces.tolist() if len(faces) > 0 else []
    except Exception as e:
        st.warning(f"얼굴 감지 중 오류가 발생했습니다: {e}")
        return []

def calculate_image_cost(image: Image.Image) -> dict:
    """이미지 처리 비용을 계산합니다."""
    # OpenAI Vision API 가격 (2024년 기준)
    # GPT-4o-mini: $0.00015 per 1K input tokens, $0.0006 per 1K output tokens
    
    width, height = image.size
    
    # 이미지 토큰 계산 (OpenAI 공식 방식)
    # 저해상도 모드: 85 토큰
    # 고해상도 모드: 85 + (타일 수 × 170) 토큰
    
    # 고해상도 모드 판정 (짧은 변이 768px 이하이고 긴 변이 2048px 이하)
    if min(width, height) <= 768 and max(width, height) <= 2048:
        # 타일 개수 계산
        if max(width, height) > 512:
            # 긴 변을 512로 맞춰서 리사이즈
            if width > height:
                new_width = 512
                new_height = int(height * 512 / width)
            else:
                new_height = 512
                new_width = int(width * 512 / height)
            
            # 512x512 타일로 나누기
            tiles_x = (new_width + 511) // 512
            tiles_y = (new_height + 511) // 512
            num_tiles = tiles_x * tiles_y
        else:
            num_tiles = 1
        
        image_tokens = 85 + (num_tiles * 170)
        detail_mode = "고해상도"
    else:
        # 저해상도 모드
        image_tokens = 85
        detail_mode = "저해상도"
        num_tiles = 1
    
    # 텍스트 토큰 예상 (프롬프트 + 응답)
    estimated_prompt_tokens = 200  # 시스템 프롬프트 + 사용자 프롬프트
    estimated_output_tokens = 150  # 감상평 + 해시태그
    
    total_input_tokens = image_tokens + estimated_prompt_tokens
    total_output_tokens = estimated_output_tokens
    
    # GPT-4o-mini 가격 계산
    input_cost = (total_input_tokens / 1000) * 0.00015  # $0.00015 per 1K tokens
    output_cost = (total_output_tokens / 1000) * 0.0006  # $0.0006 per 1K tokens
    total_cost_usd = input_cost + output_cost
    
    # 원화 환산 (1 USD = 1300 KRW 가정)
    total_cost_krw = total_cost_usd * 1300
    
    return {
        "image_size": f"{width} × {height}",
        "detail_mode": detail_mode,
        "num_tiles": num_tiles,
        "image_tokens": image_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost_usd,
        "total_cost_krw": total_cost_krw
    }

def crop_image_face_center(image: Image.Image) -> Image.Image:
    """얼굴 위치를 중심으로 4:5 비율로 이미지를 크롭합니다."""
    try:
        width, height = image.size
        faces = detect_faces_opencv(image)
        
        if faces:
            # 첫 번째 얼굴을 중심으로 설정
            x, y, w, h = faces[0]
            center_x = x + w // 2
            center_y = y + h // 2
        else:
            # 얼굴이 감지되지 않으면 이미지 중앙을 사용
            center_x, center_y = width // 2, height // 2
        
        # 4:5 비율로 크롭 영역 계산
        target_ratio = 4/5
        
        if width/height > target_ratio:
            # 가로가 더 긴 경우
            crop_h = height
            crop_w = int(crop_h * target_ratio)
        else:
            # 세로가 더 긴 경우
            crop_w = width
            crop_h = int(crop_w / target_ratio)
        
        # 크롭 영역이 이미지 범위를 벗어나지 않도록 조정
        left_crop = max(0, min(center_x - crop_w//2, width - crop_w))
        top_crop = max(0, min(center_y - crop_h//2, height - crop_h))
        right_crop = left_crop + crop_w
        bottom_crop = top_crop + crop_h
        
        return image.crop((left_crop, top_crop, right_crop, bottom_crop))
    
    except Exception as e:
        st.error(f"이미지 크롭 중 오류가 발생했습니다: {e}")
        return image

def image_to_base64(image: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환합니다."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def format_cost_info(cost_info: dict) -> str:
    """비용 정보를 보기 좋게 포맷팅합니다."""
    return f"""
**💰 예상 API 비용**
- 이미지 크기: {cost_info['image_size']}
- 처리 모드: {cost_info['detail_mode']} ({cost_info['num_tiles']}개 타일)
- 이미지 토큰: {cost_info['image_tokens']:,}개
- 총 입력 토큰: {cost_info['total_input_tokens']:,}개  
- 총 출력 토큰: {cost_info['total_output_tokens']:,}개
- **예상 비용: ${cost_info['total_cost_usd']:.6f} (약 {cost_info['total_cost_krw']:.2f}원)**
    """

def generate_caption_and_hashtags(image: Image.Image, style: str, api_key: str, custom_prompt: str = None) -> Tuple[str, str, dict]:
    """이미지에 대한 감상평과 해시태그를 생성하고 실제 사용된 토큰 정보를 반환합니다."""
    try:
        client = get_openai_client(api_key)
        if not client:
            return "API 키가 유효하지 않습니다.", "", {}
        
        # 스타일별 프롬프트 설정
        style_prompts = {
            "친근하고 귀엽게": """친근하고 귀여운 톤으로 Instagram 캡션을 작성해주세요. 
            이모티콘을 적절히 사용하고 따뜻한 말투를 사용해주세요. 
            한국어로 작성하되, 자연스럽고 SNS에 어울리는 문체로 써주세요.""",
            
            "진지하고 감동적인 말투": """진지하고 감동적인 톤으로 Instagram 캡션을 작성해주세요. 
            깊이 있고 마음에 와닿는 내용으로, 감정을 자극하는 문장을 사용해주세요. 
            한국어로 작성하되, 문학적이고 아름다운 표현을 사용해주세요.""",
            
            "장난스럽고 위트있게": """장난스럽고 위트있는 톤으로 Instagram 캡션을 작성해주세요. 
            유머를 가미하고 재치있는 표현을 사용해주세요. 
            한국어로 작성하되, 밝고 경쾌한 느낌으로 써주세요.""",
        }
        
        if style == "직접 프롬프트 작성":
            system_prompt = custom_prompt or "Instagram 캡션과 해시태그를 생성해주세요."
        else:
            system_prompt = style_prompts.get(style, style_prompts["친근하고 귀엽게"])
        
        system_prompt += """
        
        응답 형식:
        Caption: [여기에 캡션 작성]
        
        Hashtags: [여기에 해시태그 작성, #으로 시작하는 관련 해시태그들을 공백으로 구분]
        """
        
        # 이미지를 base64로 인코딩
        img_base64 = image_to_base64(image)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        
        # 실제 사용된 토큰 정보
        usage = response.usage
        actual_cost = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_cost_usd": (usage.prompt_tokens / 1000) * 0.00015,
            "output_cost_usd": (usage.completion_tokens / 1000) * 0.0006,
            "total_cost_usd": ((usage.prompt_tokens / 1000) * 0.00015) + ((usage.completion_tokens / 1000) * 0.0006),
            "total_cost_krw": (((usage.prompt_tokens / 1000) * 0.00015) + ((usage.completion_tokens / 1000) * 0.0006)) * 1300
        }
        
        # 응답에서 캡션과 해시태그 분리
        if "Hashtags:" in content:
            parts = content.split("Hashtags:", 1)
            caption = parts[0].replace("Caption:", "").strip()
            hashtags = parts[1].strip()
        else:
            caption = content.replace("Caption:", "").strip()
            hashtags = ""
        
        return caption, hashtags, actual_cost
        
    except Exception as e:
        st.error(f"감상평 생성 중 오류가 발생했습니다: {e}")
        return "감상평 생성에 실패했습니다.", "", {}

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Instagram 감상평 생성기", 
        page_icon="📸", 
        layout="centered"
    )
    
    st.title("📸 Instagram 감상평 생성기")
    st.markdown("사진을 업로드하면 AI가 자동으로 인스타그램 감상평과 해시태그를 생성해드립니다!")
    
    # API 키 확인 및 입력
    current_api_key = check_api_key()
    
    if not current_api_key:
        st.warning("🔑 OpenAI API 키가 필요합니다!")
        
        # API 키 입력 탭
        tab1, tab2 = st.tabs(["🔑 API 키 입력", "📖 API 키 발급 방법"])
        
        with tab1:
            st.subheader("OpenAI API 키를 입력하세요")
            
            # API 키 입력 필드
            api_key_input = st.text_input(
                "API 키", 
                type="password",
                placeholder="sk-proj-...",
                help="OpenAI API 키를 입력하세요. 입력한 키는 브라우저에만 저장되며 서버에 저장되지 않습니다."
            )
            
            if st.button("API 키 확인", type="primary"):
                if api_key_input.strip():
                    # API 키 유효성 검사
                    test_client = get_openai_client(api_key_input.strip())
                    if test_client:
                        st.session_state.api_key = api_key_input.strip()
                        st.success("✅ API 키가 유효합니다! 페이지를 새로고침합니다.")
                        st.rerun()
                    else:
                        st.error("❌ API 키가 유효하지 않습니다. 다시 확인해주세요.")
                else:
                    st.error("API 키를 입력해주세요.")
        
        with tab2:
            st.subheader("📋 OpenAI API 키 발급 방법")
            st.markdown("""
            1. **OpenAI 계정 생성**
               - [platform.openai.com](https://platform.openai.com) 접속
               - 계정 생성 또는 로그인
            
            2. **API 키 생성**
               - 로그인 후 우측 상단 프로필 클릭
               - "API keys" 메뉴 선택  
               - "Create new secret key" 버튼 클릭
               - 키 이름 입력 (예: "Instagram Caption Generator")
               - 생성된 키 복사 (⚠️ 한 번만 보여주므로 반드시 복사!)
            
            3. **결제 설정**
               - API 사용을 위해 결제 방법 등록 필요
               - "Billing" 메뉴에서 카드 등록
               - 보통 $5 정도 충전하면 충분함
            
            4. **주의사항**
               - API 키는 절대 다른 사람과 공유하지 마세요
               - 사용하지 않을 때는 키를 비활성화하세요
               - 사용량을 정기적으로 확인하세요
            """)
            
            st.info("💡 API 키는 이 브라우저 세션에서만 사용되며, 페이지를 새로고침하면 다시 입력해야 합니다.")
        
        return  # API 키가 없으면 여기서 종료
    
    # API 키 상태 표시
    with st.sidebar:
        st.success("🔑 API 키 연결됨")
        if st.button("API 키 변경"):
            del st.session_state.api_key
            st.rerun()
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "📁 사진을 업로드하세요", 
        type=["png", "jpg", "jpeg"],
        help="PNG, JPG, JPEG 형식의 이미지 파일을 업로드해주세요."
    )
    
    # 스타일 선택
    st.subheader("✨ 스타일 선택")
    style = st.radio(
        "어떤 스타일로 감상평을 작성할까요?",
        ["친근하고 귀엽게", "진지하고 감동적인 말투", "장난스럽고 위트있게", "직접 프롬프트 작성"],
        help="원하는 톤앤매너를 선택해주세요."
    )
    
    custom_prompt = None
    if style == "직접 프롬프트 작성":
        custom_prompt = st.text_area(
            "프롬프트를 입력하세요:",
            placeholder="예: 카페에서 찍은 사진에 대해 감성적이고 따뜻한 분위기로 캡션을 작성해주세요.",
            help="AI에게 어떤 스타일로 감상평을 작성해달라고 요청할지 구체적으로 작성해주세요."
        )
    
    if uploaded_file is not None:
        try:
            # 이미지 로드 및 표시
            image = Image.open(uploaded_file)
            
            # 이미지 크기 확인 및 조정
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 원본 이미지")
                st.image(image, use_column_width=True)
            
            # 얼굴 중심으로 4:5 비율 크롭
            cropped_image = crop_image_face_center(image)
            
            with col2:
                st.subheader("✂️ 크롭된 이미지 (4:5 비율)")
                st.image(cropped_image, use_column_width=True)
                st.caption("Instagram 피드에 최적화된 비율입니다.")
            
            # 비용 계산 및 표시
            cost_info = calculate_image_cost(cropped_image)
            
            # 비용 정보를 접을 수 있는 섹션으로 표시
            with st.expander("💰 예상 API 비용 확인", expanded=False):
                st.markdown(format_cost_info(cost_info))
                st.info("💡 실제 비용은 생성되는 텍스트 길이에 따라 약간 달라질 수 있습니다.")
                
                # 이미지 최적화 팁
                if cost_info['detail_mode'] == "고해상도" and cost_info['num_tiles'] > 4:
                    st.warning("⚡ 이미지가 큽니다! 비용을 줄이려면 이미지를 작게 리사이즈해서 다시 업로드해보세요.")
                elif cost_info['detail_mode'] == "저해상도":
                    st.success("✅ 이미지가 최적 크기입니다!")
            
            # 간단한 비용 요약을 항상 표시
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("예상 비용 (USD)", f"${cost_info['total_cost_usd']:.6f}")
            with col4:
                st.metric("예상 비용 (KRW)", f"{cost_info['total_cost_krw']:.2f}원")
            with col5:
                st.metric("토큰 수", f"{cost_info['total_input_tokens']:,}")
            
            # 감상평 생성 버튼
            if st.button("🎨 감상평 생성하기", type="primary", use_container_width=True):
                if style == "직접 프롬프트 작성" and not custom_prompt:
                    st.warning("직접 프롬프트를 작성해주세요.")
                else:
                    with st.spinner("AI가 멋진 감상평을 작성하고 있어요... ✨"):
                        caption, hashtags, actual_cost = generate_caption_and_hashtags(cropped_image, style, current_api_key, custom_prompt)
                    
                    if caption and "API 키가 유효하지 않습니다" not in caption:
                        st.success("감상평이 생성되었습니다! 🎉")
                        
                        # 실제 비용 표시
                        if actual_cost:
                            col6, col7, col8 = st.columns(3)
                            with col6:
                                st.metric("실제 사용 비용 (USD)", f"${actual_cost['total_cost_usd']:.6f}")
                            with col7:
                                st.metric("실제 사용 비용 (KRW)", f"{actual_cost['total_cost_krw']:.2f}원")
                            with col8:
                                st.metric("실제 토큰 수", f"{actual_cost['total_tokens']:,}")
                        
                        # 결과 표시
                        st.subheader("📝 감상평")
                        st.text_area(
                            "생성된 캡션",
                            caption,
                            height=150,
                            help="이 텍스트를 복사해서 Instagram에 붙여넣기 하세요."
                        )
                        
                        if hashtags:
                            st.subheader("🏷️ 해시태그")
                            st.text_area(
                                "생성된 해시태그",
                                hashtags,
                                height=100,
                                help="관련 해시태그들입니다. 필요에 따라 수정해서 사용하세요."
                            )
                        
                        # 복사 안내
                        st.info("💡 텍스트 영역을 클릭한 후 Ctrl+A로 전체 선택, Ctrl+C로 복사할 수 있습니다.")
                        
                        # 상세 비용 분석
                        if actual_cost:
                            with st.expander("📊 상세 비용 분석", expanded=False):
                                col9, col10 = st.columns(2)
                                with col9:
                                    st.markdown("**예상 vs 실제 비교**")
                                    comparison_data = {
                                        "구분": ["입력 토큰", "출력 토큰", "총 비용(USD)", "총 비용(KRW)"],
                                        "예상": [
                                            f"{cost_info['total_input_tokens']:,}",
                                            f"{cost_info['total_output_tokens']:,}",
                                            f"${cost_info['total_cost_usd']:.6f}",
                                            f"{cost_info['total_cost_krw']:.2f}원"
                                        ],
                                        "실제": [
                                            f"{actual_cost['prompt_tokens']:,}",
                                            f"{actual_cost['completion_tokens']:,}",
                                            f"${actual_cost['total_cost_usd']:.6f}",
                                            f"{actual_cost['total_cost_krw']:.2f}원"
                                        ]
                                    }
                                    st.table(comparison_data)
                                
                                with col10:
                                    st.markdown("**비용 절약 팁**")
                                    st.markdown("""
                                    - 이미지 크기를 512x512 이하로 줄이면 저해상도 모드 사용
                                    - 프롬프트를 간결하게 작성
                                    - 불필요한 재생성 자제
                                    - 배치 처리로 여러 이미지 한번에 처리
                                    """)
                    else:
                        st.error("감상평 생성에 실패했습니다. API 키를 확인해주세요.")
        
        except Exception as e:
            st.error(f"이미지 처리 중 오류가 발생했습니다: {e}")
    
    # 사용법 안내
    with st.expander("📋 사용법 안내"):
        st.markdown("""
        1. **API 키 입력**: OpenAI API 키를 입력하세요. (한 번만 입력하면 됩니다)
        2. **사진 업로드**: PNG, JPG, JPEG 형식의 이미지를 업로드하세요.
        3. **비용 확인**: 업로드 후 예상 API 비용을 확인할 수 있습니다.
        4. **스타일 선택**: 원하는 톤앤매너를 선택하거나 직접 프롬프트를 작성하세요.
        5. **감상평 생성**: 버튼을 클릭하면 AI가 자동으로 감상평과 해시태그를 생성합니다.
        6. **복사 및 사용**: 생성된 텍스트를 복사해서 Instagram에 사용하세요.
        
        **팁**: 
        - 얼굴이 포함된 사진의 경우 자동으로 얼굴을 중심으로 크롭됩니다.
        - 이미지 크기가 클수록 비용이 높아집니다. 512x512 이하로 줄이면 비용을 절약할 수 있습니다.
        - 실제 비용은 생성되는 텍스트 길이에 따라 예상 비용과 약간 다를 수 있습니다.
        
        **보안**: API 키는 브라우저에서만 사용되며 서버에 저장되지 않습니다.
        """)
    
    # 비용 정보 섹션
    with st.expander("💰 API 비용 정보"):
        st.markdown("""
        ### OpenAI GPT-4o-mini 가격 (2024년 기준)
        - **입력 토큰**: $0.00015 per 1,000 tokens
        - **출력 토큰**: $0.0006 per 1,000 tokens
        - **이미지 처리**: 해상도에 따라 85~수백 토큰
        
        ### 이미지 토큰 계산 방식
        - **저해상도 모드**: 85 토큰 (이미지가 512px 이하)
        - **고해상도 모드**: 85 + (타일 수 × 170) 토큰
        
        ### 예상 비용 예시
        - **일반적인 사진 1장**: 약 $0.0001~0.0005 (0.13~0.65원)
        - **고해상도 사진 1장**: 약 $0.0005~0.002 (0.65~2.6원)
        
        실제 사용해보시면 생각보다 저렴하다는 것을 느끼실 수 있을 거예요! 🎉
        """)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Made with ❤️ using Streamlit & OpenAI | "
        "<a href='https://github.com' target='_blank'>GitHub</a>"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()