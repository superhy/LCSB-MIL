'''
@author: Yang Hu
'''
import sys


'''
for different task id, with different meanning!

2: algorithm attention MIL
5. algorithm LCSB MIL
7. algorithm reLCSB MIL

For algorithm 2 (attention-MIL):
    20: Attentional Pool MIL (with ReLU activation)
    21: Gated Attentional Pool MIL (with ReLU activation)
For algorithm 5 (LCSB MIL)
    50: LCSB with Attention Pool, encoder: ResNet18
    51: LCSB with Gated Attention Pool, encoder: ResNet18
For algorithm 7 (reversed gradient LCSB MIL)
    70: reversed LCSB with Attention Pool, encoder: ResNet18
    71: reversed LCSB with Gated Attention Pool, encoder: ResNet18
For algorithm 0 (pre-training of aggregator in MIL)
    01(1): pre-training the Attention Pool
    02(2): pre-training the Gated Attention Pool
'''

class Logger(object):

    def __init__(self, filename='running_log-default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    pass