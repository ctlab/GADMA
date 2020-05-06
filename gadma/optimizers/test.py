class A(object):
    def __init__(self):
        print("A.__init__")

    def check(self):
        print("A.check")

class B(A):
    def check(self):
        print("B.check")

class C(B):
    def check(self):
        print("C.check")

class D(A):
    def __init__(self, x=0):
        self.x = x
        super(D, self).__init__()
        print("D.__init__")

    def test(self):
        self.check()

class E(D, C):
    pass

