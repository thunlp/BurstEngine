import sys
import numpy as np
import argparse
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse log file and extract TFLOPs lines.')
    parser.add_argument('-f', '--file', required=True, help='Path to the log file')
    args = parser.parse_args()

    with open(args.file, 'r') as file:
        whole_content = file.read()
        for c in whole_content.split("after training is done"):
            arr1 = []
            arr2 = []
            for line in c.split('\n'):
                if "TFLOPs" in line:
                    res = line.split("|")
                    tgs, tflops = res[-3], res[-2]
                    arr1.append(float(tgs.split(":")[-1]))
                    arr2.append(float(tflops.split(":")[-1]))
            if len(arr1) > 0:
                print(arr1[2:])
                print(np.std(arr1[2:]))
                print(np.mean(arr1[2:]))
                # print("mean tgs: {:.2f}".format( sum(arr1[:2])/len(arr1[:2])))
                # print("std tgs: {:.2f}".format( np.std(arr1[:2])))
                # print("mean tflops: {:.2f}".format( sum(arr2[:2])/len(arr2[:2])))
                # print("std tflops: {:.2f}".format( np.std(arr2[:2])))

                    # tgs = tgs.split(" ")[-1]
                    # tflops = tflops.split(" ")[-1]
            print("******************")
