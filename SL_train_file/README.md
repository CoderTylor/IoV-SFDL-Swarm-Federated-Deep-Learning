## Swarm Federated Learning

## System setup for the multi-node examples
1. The instructions in these examples assume that Swarm Learning will run on 8 to 10 systems.
    - These 10 systems have IP addresses 172.1.1.1, 172.2.2.2, 172.3.3.3, 172.4.4.4, 172.5.5.5, 172.6.6.6, 172.7.7.7, 172.8.8.8, 172.9.9.9 and 172.10.10.10.
    - 172.1.1.1 will run the Sentinel node.
    - 172.4.4.4 will run a Swarm Network node.
    - 172.2.2.2 and 172.3.3.3 will run one Swarm Learning node each. These nodes will register themselves with the Sentinel node that is running on 172.1.1.1.
    - 172.5.5.5 and 172.6.6.6 will also run one Swarm Learning node each. These nodes will register themselves with the Swarm Network node that is running on 172.4.4.4.
    - 172.7.7.7 will run the License Server.
    - 172.8.8.8, 172.9.9.9 and 172.10.10.10 will run one SPIRE server node each, with different configurations. The sample program might use one, two or all three of these servers.

2. Further, these instructions also assume that each of the 4 hosts meant for running Swarm Learning nodes (172.2.2.2, 172.3.3.3, 172.5.5.5 and 172.6.6.6) have 8 NVIDIA GPUs each.

3. Finally, these instructions assume swarm-learning to be the current working directory on all 8 systems:
    `cd swarm-learning`

4. The scripts supplied with the Swarm Learning package do not have the capability to work across systems. So, the instructions must be issued on the right systems:
    - Commands targetting the Sentinel node should be issued on 172.1.1.1.
    - Commands targetting the Swarm Network node should be issued on 172.4.4.4
    - Commands targetting the Swarm Learning nodes that register with the Sentinel node should be issued on 172.2.2.2 and 172.3.3.3.
    - Commands targetting the Swarm Learning nodes that register with the Swarm Network node should be issued on 172.5.5.5 and 172.6.6.6.
    - Commands targetting the License Server node should be issued on 172.7.7.7.
    - Commands targetting the SPIRE server node should be issued on 172.8.8.8, 172.9.9.9 and 172.10.10.10.
