#for the submission uncomment the submission statements
#so submission.README

from math import *
import numpy as np
from submission import *

def seven_segment(pattern):

    def to_bool(a):
        if a==1:
            return True
        return False


    def hor(d):
        if d:
            print(" _ ")
        else:
            print("   ")

    def vert(d1,d2,d3):
        word=""

        if d1:
            word="|"
        else:
            word=" "

        if d3:
            word+="_"
        else:
            word+=" "

        if d2:
            word+="|"
        else:
            word+=" "

        print(word)



    pattern_b=list(map(to_bool,pattern))

    hor(pattern_b[0])
    vert(pattern_b[1],pattern_b[2],pattern_b[3])
    vert(pattern_b[4],pattern_b[5],pattern_b[6])

    number=0
    for i in range(0,4):
        if pattern_b[7+i]:
            number+=pow(2,i)
    print(int(number))


def evolution( weight_matrix, test, N ):

    seven_segment(test)
    submission.seven_segment(test)
    submission.qquad()

    test_old = test.copy()
    test_new = np.zeros((N))
    iteration = 0

    while True:
        iteration += 1
        #unstable pattern if exceeds 20 iterations and will continue till infinity
        #if iteration == 20:
        #    print("unstable")
        #    break

        #applying McCulloch- Pitts formula to evolve synchronously
        test_new = weight_matrix.dot(test_old)
        #thresholding
        for i in range(N):
            if test_new[i] <= 0:
                test_new[i] = -1
            else :
                test_new[i] = 1
        #checks if there is no change in pattern and exits
        if np.array_equal(test_new,test_old):
            break
        #here the network should run printing at each step
        seven_segment(test_new)
        #for the final submission it should also output to submission on each step
        submission.seven_segment(test_new)
        submission.qquad()
        #updating the test_old with current version
        test_old = test_new.copy()


submission=Submission("ya17227")
submission.header("Yash Agarwal")

six=[1,1,-1,1,1,1,1,-1,1,1,-1]
three=[1,-1,1,1,-1,1,1,1,1,-1,-1]
one=[-1,-1,1,-1,-1,1,-1,1,-1,-1,-1]

seven_segment(three)
seven_segment(six)
seven_segment(one)


##learning all the above patterns and storing knowledge in "weight_matrix"
weight_matrix = np.zeros((11,11))
for i in range(0,11):
    for j in range(0,11):
        if i != j:
            weight_matrix[i][j] = ((three[i] * three[j]) + (six[i] * six[j]) + (one[i] * one[j]))/3

##this assumes you have called your weight matrix "weight_matrix"
submission.section("Weight matrix")
submission.matrix_print("W",weight_matrix)



print("test1")
submission.section("Test 1")
test=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
evolution( weight_matrix, test, len(test))

print("test2")
submission.section("Test 1")
test=[1,1,1,1,1,1,1,-1,-1,-1,-1]
evolution( weight_matrix, test, len(test))


submission.bottomer()
