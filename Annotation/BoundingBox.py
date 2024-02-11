def filter_tooNear(objects):
    pre_point = None
    distance = None
    removeList = []
    for i in range(0, len(objects)):
        for j in range(0, len(objects)):
            if j != i:
                if abs(objects[j][0][0] - objects[i][0][0]) <= 6 and abs(objects[j][0][1] - objects[i][0][1]) <= 2 and abs(objects[j][0][2] - \
                        objects[i][0][2]) <= 1.8:
                    if objects[j][1]>objects[i][1]:
                        removeList.append(i)
                    else:
                        removeList.append(j)
    removeList=list(set(removeList))
    removeList.sort(reverse=True)
    for index in removeList:
        objects.pop(index)
    return objects