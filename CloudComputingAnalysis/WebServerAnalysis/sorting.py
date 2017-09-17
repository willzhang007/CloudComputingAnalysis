__author__ = 'lichaozhang'
import time



start_time = time.clock()


def readfile(fname):
    content =[]
    with open(fname) as f:
        for line in f:
            content.append(int(line))
    return content


def mergeSort(alist):
    #print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    #print("Merging ",alist)

for i in range(0,10):


    prev_time1 = time.clock()

    path = '/Users/lichaozhang/Desktop/ECE670Dataset/sorting/'


    random_100 = readfile(path +"random_100.txt")
#print random_100
    #print alist
    mergeSort(random_100)

#print len(random_100)
    now_time1 = time.clock()



#print "sort 100 numbers takes >>>>>>",now_time1-prev_time1


    prev_time2 = time.clock()

    random_1000 = readfile(path+"random_1000.txt")

    mergeSort(random_1000)
#print len(random_1000)

    now_time2 = time.clock()



#print "sort 1000 numbers takes >>>>>>",now_time2-prev_time2


    prev_time3 = time.clock()

    random_2000 = readfile(path+"random_2000.txt")


    mergeSort(random_2000)

#print len(random_2000)
    now_time3 = time.clock()



#print "sort 2000 numbers takes >>>>>>",now_time3-prev_time3


    prev_time4 = time.clock()

    random_5000 = readfile(path+"random_5000.txt")
    mergeSort(random_5000)

#print len(random_5000)
    now_time4 = time.clock()



#print "sort 5000 numbers takes >>>>>>",now_time4-prev_time4

    prev_time5 = time.clock()

    random_10000 = readfile(path+"random_10000.txt")
    mergeSort(random_10000)

#print len(random_10000)
    now_time5 = time.clock()



    prev_time6 = time.clock()

    random_20000 = readfile(path+"random_10000.txt")
    mergeSort(random_20000)

#print len(random_10000)
    now_time6 = time.clock()



    prev_time7 = time.clock()

    random_50000 = readfile(path+"random_50000.txt")
    mergeSort(random_50000)
#print len(random_50000)
    now_time7 = time.clock()


#print "sort 10000 numbers takes >>>>>>",now_time5-prev_time5

    prev_time8 =time.clock()

    random_100000 = readfile(path+"random_100000.txt")
    mergeSort(random_100000)
    now_time8 = time.clock()
#print len(random_100000)
    prev_time9 =time.clock()

    random_200000 = readfile(path+"random_200000.txt")
    mergeSort(random_200000)
    now_time9 = time.clock()
#print len(random_200000)

    prev_time10 =time.clock()
    random_500000 = readfile(path+"random_500000.txt")
    mergeSort(random_500000)
    now_time10 = time.clock()
#print len(random_500000)

    prev_time11 =time.clock()
    random_1000000 = readfile(path+"random_1000000.txt")
    mergeSort(random_1000000)
    now_time11 = time.clock()
#print len(random_1000000)
    print  "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n{9}\n{10}".format(now_time1-prev_time1,now_time2-prev_time2,now_time3-prev_time3,
now_time4-prev_time4,now_time5-prev_time5,now_time6-prev_time6,now_time7-prev_time7
,now_time8-prev_time8,now_time9-prev_time9,now_time10-prev_time10,now_time11-prev_time11)

