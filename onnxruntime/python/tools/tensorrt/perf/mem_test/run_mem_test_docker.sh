#!/bin/bash

set -x

# Run Valgrind to check if there's memory leak
valgrind --leak-check=yes --show-leak-kinds=all ./code/onnxruntime/build/Linux/Release/onnxruntime_perf_test -e tensorrt -r 1 -i /code/onnxruntime/build/Linux/Release/testdata/squeezenet/model.onnx 2> /code/onnxruntime/build/Linux/Release/valgrind.log 
found_leak_summary=false
while IFS= read -r line
do
  if echo $line | grep -q 'LEAK SUMMARY:'; then
    found_leak_summary=true
  elif $found_leak_summary && echo $line | grep -q 'definitely lost:'; then
    bytes_lost=$(echo $line | grep -o -E '[0-9,]+ bytes')
    blocks_lost=$(echo $line | grep -o -E '[0-9]+ blocks')
    echo "Bytes lost: $bytes_lost"
    echo "Blocks lost: $blocks_lost"
    if [ "$blocks_lost" != "0 blocks" ]; then
      echo 'Memory leak happened when testing squeezenet model!'
    fi
    found_leak_summary=false
  fi
done < "/code/onnxruntime/build/Linux/Release/valgrind.log"

# Parse Arguments
while getopts w:d:p:l: parameter
do case "${parameter}"
in 
w) WORKSPACE=${OPTARG};; # workspace folder of onnxruntime
d) DOCKER_IMAGE=${OPTARG};; # docker image:"trt-ep-mem-test" docker image is already pre-built on perf machine
p) MEM_TEST_DIR=${OPTARG};; # mem test dir
l) BUILD_ORT_LATEST=${OPTARG};; # whether to build latest ORT
esac
done 

# Variables
DOCKER_MEM_TEST_DIR='/mem_test/'
DOCKER_ORT_SOURCE=$WORKSPACE'onnxruntime'
DOCKER_ORT_LIBS=$DOCKER_ORT_SOURCE'/build/Linux/Release/' # This is the path on container where all ort libraries (aka libonnxruntime*.so) reside.


if [ -z ${BUILD_ORT_LATEST} ]
then
    BUILD_ORT_LATEST="true"
fi

docker run --rm --gpus all -v $MEM_TEST_DIR:$DOCKER_MEM_TEST_DIR $DOCKER_IMAGE /bin/bash $DOCKER_MEM_TEST_DIR'run.sh' -p $DOCKER_MEM_TEST_DIR -o $DOCKER_ORT_LIBS -s $DOCKER_ORT_SOURCE -l $BUILD_ORT_LATEST
