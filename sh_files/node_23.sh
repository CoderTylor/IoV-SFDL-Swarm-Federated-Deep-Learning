<<<<<<< HEAD
APLS_IP=192.168.0.6
EXAMPLE=mnist-keras
TRAINING_NODE=node3
WORKSPACE_DIR=$PWD
bash ../swarm-learning/bin/run-sl --name $TRAINING_NODE-sn --network $EXAMPLE-net --host-ip $TRAINING_NODE-sn --sn-ip node-sn -e MAX_EPOCHS=1000 --apls-ip $APLS_IP --serverAddress node-spire_1 -genJoinToken --data-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/ws-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform PYT
=======
APLS_IP=10.128.233.41
EXAMPLE=mnist-keras
TRAINING_NODE=node3
WORKSPACE_DIR=$PWD
bash ../swarm-learning/bin/run-sl --name $TRAINING_NODE-sn --network $EXAMPLE-net --host-ip $TRAINING_NODE-sn --sn-ip node-sn -e MAX_EPOCHS=1000 --apls-ip $APLS_IP --serverAddress node-spire_1 -genJoinToken --data-dir $WORKSPACE_DIR/wx-$EXAMPLE/$TRAINING_NODE/app-data --model-dir $WORKSPACE_DIR/wx-$EXAMPLE/$TRAINING_NODE/model --model-program mnist_pyt.py --sl-platform PYT
>>>>>>> c570179daa53c7cd2cafe6e9b4d95446932832f4
