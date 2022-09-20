# IoV-SFDL: Swarm-Federated-Deep-Learning Framework
LSTM network to verify trajectory prediction on the NGSIM dataset based on IoV-SFDL framework.

Paper Link:   https://arxiv.org/abs/2108.03981

## Framework Scheme



<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="./Fig/scheme.png"
         alt="Fig not Found"
         style="zoom:30%"/>
    <br>		<!--换行-->
  IoV SFDL Framework.	<!--标题-->
    </center>
</div>
This Project proposes a Swarm-Federated Deep Learning framework in the IoV system (IoV-SFDL) that integrates SL into the FDL framework. The IoV-SFDL organizes vehicles to generate local SL models with adjacent vehicles based on the blockchain empowered SL, then aggregates the global FDL model among different SL groups with a proposed credibility weights prediction algorithm.

**If you have any questions during the usage of this repository, feel free to open a new issue or contact email: tyrlorwang@bupt.edu.cn (and cc it to tylor.wang@kcl.ac.uk)**

## Please cite our paper if the source code can help you.

```
@article{wang2021credibility,
  title={A Credibility-aware Swarm-Federated Deep Learning Framework in Internet of Vehicles},
  author={Wang, Zhe and Li, Xinhang and Wu, Tianhao and Xu, Chen and Zhang, Lin},
  journal={arXiv preprint arXiv:2108.03981},
  year={2021}
}
```

## Getting Started

* **[Prerequisites](https://github.com/HewlettPackard/swarm-learning/blob/v0.3.0/docs/Prerequisites.md) for Swarm Learning**

The SFDL is based on the Swarm Learning framework. The specific prerequisites is the same as Swarm Learning framework, which could be shown as above.

* **Clone this repository**

```shell
git clone https://github.com/CoderTylor/IoV-SFDL-Swarm-Federated-Deep-Learning.git
```

* **[Download and setup](https://github.com/HewlettPackard/swarm-learning/blob/v0.3.0/docs/setup.md) docker images and evaluation licenses**

As the same as the Swarm Learning framework have a maximum of 16 sn nodes and other identity authentication nodes.

* **The key packages and their versions used in our algorithm implementation are listed as follows**

- python==3.7.0
- keras==2.2.4
- pytorch==1.7
- numpy==1.14.5
- pandas==0.23.4
- scipy==1.1.0



* **Execute Steps**

**1. Start the License Server**

Do not stop the License Server once the licenses are installed

```shell
#Start the APLS container using swarm-learning-install-dir/swarm-learning/bin/run-apls --apls-port=5814
sudo chmod +x swarm-learning-install-dir/swarm-learning/bin/run-apls
./swarm-learning/bin/run-apls

open chrom with url 127.0.0.1:5814
Default Username:admin, Password:password
Install Liscense
Notice: The Swarm Learning Framework Liscense has a maximam of 16 sn nodes and 4 other nodes.
```

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="./Fig/Install Liscense.png"
         alt="Fig not Found"
         style="zoom:30%"/>
    <br>		<!--换行-->
  Install Swarm Learning Liscense.	<!--标题-->
    </center>
</div>


**2. Init Circle-1 workspace**

```shell
cd .~/swarm-learning-install-dir
APLS_IP=<License Host Server IP>
EXAMPLE=mnist-keras
WORKSPACE_DIR=$PWD
./SL_trainfile/mnist-keras/bin/init-workspace -e $EXAMPLE -i $APLS_IP -d $WORKSPACE_DIR
```

**3. Init Circle-2 workspace**

```shell
cd .~/swarm-learning-install-dir
EXAMPLE=mnist-pytorch
WORKSPACE_DIR=$PWD
./SL_trainfile/mnist-pytorch/bin/init-workspace_2 -e $EXAMPLE -i $APLS_IP -d $WORKSPACE_DIR
```

<font color=red>Tips: </font> 

<font color=red>1. The self-designed deep learning program is placed in the mini-PyTorch and mini-Keras folders to replace the original deep learning program. This way can solve the problem of the program error.</font> 

<font color=red>2. Use docker network ls to check whether the two different docker images have connected to the same docker netork</font> 

**4.1 (Recommend) Execute Using Bash Files**

```
# A 3 nodes 2 circles sample could be executed through the command 
bash main.sh
```

**4.2 Execute Using Command Line**

```shell
#Execute Circle-1
#！/user/bin/env bash

APLS_IP=<License Host Server IP>
EXAMPLE=mnist-keras
WORKSPACE_DIR=$PWD

TRAINING_NODE=node1
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_tf.py --sl-platform TF

TRAINING_NODE=node2
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_tf.py --sl-platform TF

TRAINING_NODE=node3
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_tf.py --sl-platform TF
...

#Execute Circle-2
#！/user/bin/env bash
APLS_IP=<License Host Server IP>
EXAMPLE=mnist-keras
WORKSPACE_DIR=$PWD

TRAINING_NODE=node1
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform TF

TRAINING_NODE=node2
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform TF
	
TRAINING_NODE=node3
../swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=5 --apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform TF
...
```

**5. Execute global federated Learning**

```shell
cd .~/swarm-learning-install-dir
python federated.py
```

**6.Dataset**

The dataset used in the experiment is the Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data, which can be downloaded from the [URL](https://www.fhwa.dot.gov/publications/research/operations/06137/index.cfm) and saved and generated in the following [format](https://github.com/CoderTylor/IoV-SFDL-Swarm-Federated-Deep-Learning/blob/main/data/test.csv)


<<<<<<< HEAD
=======

>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
## References

- [Papers](https://arxiv.org/pdf/2108.03981.pdf)
- [Videos]()

![image](https://github.com/CoderTylor/IoV-SFDL-Swarm-Federated-Deep-Learning/blob/main/Fig/Trim.gif)
<<<<<<< HEAD


=======
>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
