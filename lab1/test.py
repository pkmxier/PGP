import random
import sys

tests = 1
max_n = int(sys.argv[1])

for index in range(tests):
    with open("{}.t".format(max_n), "w") as test, \
         open("{}.a".format(max_n), "w") as ans:
        n = max_n #random.randint(1, max_n)
        test.write("{}\n".format(n))
        
        print("n = {}".format(n))
        
        for i in range(n):
            number = random.uniform(-1e8, 1e8)
            test.write("{} ".format(number))
            ans.write("{:.10e} ".format(abs(number)))
        test.write("\n")
        ans.write("\n")
