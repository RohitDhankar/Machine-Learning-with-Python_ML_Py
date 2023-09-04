ls_ints = [1,2,3,4,5,]
results = []

def sqrGen(ls_ints):
    for k in range(len(ls_ints)):
        results.append(k*k)
    return results
        #yield(k*k)

results = sqrGen(ls_ints)
print(results)
#
def sqrGen(ls_ints):
    for k in range(len(ls_ints)):
        results.append(k*k)
