from smoothing import simple_good_turing

def _isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def test_file(fileName):
    in_dict = {}
    word = 0
    with open(fileName, "r") as inFile:
        for line in inFile:
            r, n_r = [int(_) for _ in line.split()]
            for _ in range(n_r):
                in_dict[word] = r
                word += 1

    res = simple_good_turing(in_dict)
#    for k,v in res.iteritems():
#        print("{}:{}".format(k,v))
    assert _isclose(sum(res.itervalues()), 1.), "norm_gt_dict is not properly normalized."

if __name__=="__main__":
    test_file("chinese.txt")
    test_file("prosody.txt")
