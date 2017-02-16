from googlenet import TestDataGenerator
import argparse
import os
import numpy as np
from scipy import misc


class Generator:
    def __init__(self, number, output_path):
        self.number = number
        self.output_path = os.path.abspath(output_path)
        if not os.path.isdir(self.output_path):
            try:
                os.mkdir(self.output_path)
            except:
                print("%s not existed and can not be created." % self.output_path)
                exit(1)
        if not os.path.isdir("%s/%s" % (self.output_path, 0)):
            try:
                os.mkdir("%s/%s" % (self.output_path, 0))
            except:
                print("%s/%s not existed and can not be created." % (self.output_path, 0))
                exit(1)
        if not os.path.isdir("%s/%s" % (self.output_path, 1)):
            try:
                os.mkdir("%s/%s" % (self.output_path, 1))
            except:
                print("%s/%s not existed and can not be created." % (self.output_path, 1))
                exit(1)
        print("Will generate %s pic in %s." % (self.number, self.output_path))
        self.data = TestDataGenerator(self.number * 2, batch_size=128, test_data_percent=0.1).train_generator()

    def generate(self):
        count = 0
        while True:
            result_x, result_y = next(self.data)
            for i in range(0, 128):
                if count % 1000 == 0:
                    print("%s/%s" % (count, self.number))
                if count > self.number:
                    break
                count += 1
                picture = result_x[i]
                picture_type = np.argmax(result_y[0][i])
                misc.toimage(picture).save("%s/%s/%s.png" % (self.output_path, picture_type, count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=100000, help="Number of generate pictures.")
    parser.add_argument("-o", "--output", default="./data", help="Output path.")
    args = parser.parse_args()
    generator = Generator(number=args.number, output_path=args.output)
    generator.generate()
