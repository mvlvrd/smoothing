class myCounter(dict):
    def __init__(self,inputList,defaultList=None):
        """
        This is a counter where all the items in defaultList are supposed to be included.
        If not defined then assume the default elements are all the keys in inputList.
        We prefer this to collections.Counter because Counter wipes out zero count elements
        when merging.
        """
        super(myCounter,self).__init__()
        if defaultList is None:
            defaultList = inputList
        for item in defaultList:
            self[item] = 0
        for item in inputList:
            self[item] += 1
