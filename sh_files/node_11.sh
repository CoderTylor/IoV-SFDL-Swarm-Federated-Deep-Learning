APLS_IP=192.168.0.6
EXAMPLE=mnist-pytorch
TRAINING_NODE=node1
WORKSPACE_DIR=$PWD
bash ./swarm-learning/bin/run-sl --name $TRAINING_NODE-sl --network $EXAMPLE-net --host-ip $TRAINING_NODE-sl --sn-ip node-sn -e MAX_EPOCHS=1000^C-apls-ip $APLS_IP --serverAddress node-spire -genJoinToken --data-dir $WORKSPACE_DIR/wx-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform PYT