# 이미지 빌드를 위한 베이스 이미지 설정
FROM ubuntu:20.04 AS builder

# 이미지 관리자 정보 추가
MAINTAINER Gu Lim <lim9gu@gmail.com>
#LABEL description="Hotpants Docker Image"

# 필요한 패키지 설치와 hotpants 소스코드 빌드
# 컨테이너의 /home/hotpants 디렉토리에 hotpants를 다운받아 compile
RUN apt update && apt install -y libccfits-dev git make build-essential
RUN git clone https://github.com/acbecker/hotpants.git /home/hotpants
WORKDIR /home/hotpants
RUN git reset --hard 63e126081509a52e96260acdcf2f77b31096c173  # 버전 5.1.11으로 리셋 
RUN make clean && make


# 세 번째 스테이지: 최종 이미지 구성하고, 의존성 패키지 추가 설치
FROM ubuntu:20.04
RUN apt update && apt install -y libccfits-dev 

# Global 사용을 위한 파일 복사
COPY --from=builder /home/hotpants/extractkern /home/hotpants/hotpants /home/hotpants/maskim /usr/local/bin

