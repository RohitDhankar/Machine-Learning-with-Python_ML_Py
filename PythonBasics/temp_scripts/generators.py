ints = [1,2,3,4,5,]
results = []

def sqrGen(ints):
    for k in range(len(ints)):
        results.append(k*k)
    return results
        #yield(k*k)

results = sqrGen(ints)
print(results)
#
def sqrGen(ints):
    for k in range(len(ints)):
        results.append(k*k)
