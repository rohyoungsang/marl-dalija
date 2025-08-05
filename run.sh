#!/bin/bash
GPU=$1
COMMAND=${@:2}

# 컨테이너 이름 규칙
NAME="pymarl_gpu_${GPU}"

# GPU 출력
echo "Target GPU: $GPU"
echo "Container Name: $NAME"

# nvidia-docker 또는 docker 선택
if hash nvidia-docker 2>/dev/null; then
    CMD=nvidia-docker
else
    CMD=docker
fi

# 컨테이너 상태 확인
EXISTING=$(docker ps -a --filter "name=^/${NAME}$" --format "{{.Status}}")

if [[ "$EXISTING" == "" ]]; then
    echo "[INFO] 컨테이너가 없으므로 새로 생성합니다."
    NV_GPU="$GPU" $CMD run \
        --name $NAME \
        --gpus device=$GPU \
        -v "$(pwd)":/pymarl \
        -dit pymarl:1.0 ${COMMAND:-/bin/bash}

elif [[ "$EXISTING" == Exited* ]]; then
    echo "[INFO] 컨테이너가 꺼져 있으므로 다시 시작합니다."
    docker start -ai $NAME

elif [[ "$EXISTING" == Up* ]]; then
    echo "[INFO] 컨테이너가 이미 실행 중입니다. 접속합니다."
    docker exec -it $NAME ${COMMAND:-/bin/bash}
fi

