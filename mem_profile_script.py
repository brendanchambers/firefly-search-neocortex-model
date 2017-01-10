from run_firefly1 import run_firefly1
from memory_profiler import profile

# UPDATE don't need to do it this way
# instead, just from memory_profiler import profile
#    and add the @profile above functions you want to see mem usage for


@profile
def runFireflyForMemoryTest():
    run_firefly1()

if __name__ == '__main__':
    runFireflyForMemoryTest()