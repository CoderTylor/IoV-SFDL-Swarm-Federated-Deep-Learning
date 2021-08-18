# IoV-SFDL: Swarm-Federated-Deep-Learning Framework
LSTM network to verify trajector prediction on the NGSIM dataset based on IoV-SFDL framework. The work is based on our paper:xxxxxxxxxxxx

This work is based on the Swarm Learning Framework to achieve efficient and privacy protected vehicle trajectory prediction task in the IoV system.

![微信截图_20210818101303](https://user-images.githubusercontent.com/55383755/129825713-41d69ecf-5813-4c6e-95bb-a55405ef2e9a.png)


<!-- First, please fellow the Swarm Learning @ https://github.com/HewlettPackard/swarm-learning to download and setup docker images and evaluation licenses.

Step 1, please open a terminal, change directory to ./SL-IoV/SL_file/

bash ./swarm-learning/bin/run-spire-server --name=spire-server -p 8081:8081

Step 2, please open another two terminal which represent two swarm node, the directory is ./SL-IoV/SL_file/

bash ./swarm-learning/bin/run-sn  \
    --name=sn-1              \
    --host-ip=172.1.1.1      \
    --sentinel-ip=172.1.1.1  \
    --sn-p2p-port=10000      \
    --sn-api-port=11000      \
    --sn-fs-port=12000       \
    --apls-ip 172.7.7.7      \
    -serverAddress 172.8.8.8 \
    -genJoinToken
    
 
bash ./swarm-learning/bin/run-sn  \
    --name=sn-2              \
    --host-ip=172.4.4.4      \
    --sentinel-ip=172.1.1.1  \
    --sn-p2p-port=13000      \
    --sn-api-port=14000      \
    --sn-fs-port=15000       \
    --sentinel-fs-port=12000 \
    --apls-ip 172.7.7.7      \
    -serverAddress 172.8.8.8 \
    -genJoinToken
    
Step3: Start the four Swarm Learning nodes on 172.2.2.2, 172.3.3.3, 172.5.5.5 and 172.6.6.6 respectively. Specify --sl-platform=PYT 

bash ./swarm-learning/bin/run-sl        \
    --name=sl-1                         \
    --sl-platform=PYT                   \
    --host-ip=172.2.2.2                 \
    --sn-ip=172.1.1.1                   \
    --sn-api-port=11000                 \
    --sl-fs-port=16000                  \
    --data-dir=examples/mnist/app-data  \
    --model-dir=examples/mnist/model    \
    --model-program=mnist_pyt.py        \
    --gpu=0                             \
    --apls-ip 172.7.7.7                 \
    -serverAddress 172.8.8.8            \
    -genJoinToken

,

bash ./swarm-learning/bin/run-sl        \
    --name=sl-2                         \
    --sl-platform=PYT                   \
    --host-ip=172.3.3.3                 \
    --sn-ip=172.1.1.1                   \
    --sn-api-port=11000                 \
    --sl-fs-port=17000                  \
    --data-dir=examples/mnist/app-data  \
    --model-dir=examples/mnist/model    \
    --model-program=mnist_pyt.py        \
    --gpu=3                             \
    --apls-ip 172.7.7.7                 \
    -serverAddress 172.8.8.8            \
    -genJoinToken
,

bash ./swarm-learning/bin/run-sl        \
    --name=sl-3                         \
    --sl-platform=PYT                   \
    --host-ip=172.5.5.5                 \
    --sn-ip=172.4.4.4                   \
    --sn-api-port=14000                 \
    --sl-fs-port=18000                  \
    --data-dir=examples/mnist/app-data  \
    --model-dir=examples/mnist/model    \
    --model-program=mnist_pyt.py        \
    --gpu=5                             \
    --apls-ip 172.7.7.7                 \
    -serverAddress 172.8.8.8            \
    -genJoinToken
,

bash ./swarm-learning/bin/run-sl        \
    --name=sl-4                         \
    --sl-platform=PYT                   \
    --host-ip=172.6.6.6                 \
    --sn-ip=172.4.4.4                   \
    --sn-api-port=14000                 \
    --sl-fs-port=19000                  \
    --data-dir=examples/mnist/app-data  \
    --model-dir=examples/mnist/model    \
    --model-program=mnist_pyt.py        \
    --gpu=7                             \
    --apls-ip 172.7.7.7                 \
    -serverAddress 172.8.8.8            \
    -genJoinToken
 -->

Then, change the directory to ./SL-IoV/FL_file/ with commond

python FL_aggregation.py


Model information

VPTLSTM(
(cell): LSTMCell(64, 32)
(input_embedding_layer): Linear(in_features=9, out_features=32, bias=True)
(social_tensor_conv1): Conv2d(32, 16, kernel_size=(5, 3), stride=(2, 1))
(social_tensor_conv2): Conv2d(16, 8, kernel_size=(5, 3), stride=(1, 1))
(social_tensor_embed): Linear(in_features=32, out_features=32, bias=True)
(output_layer): Linear(in_features=32, out_features=5, bias=True)
(relu): ReLU()
(dropout): Dropout(p=0, inplace=False)
)
